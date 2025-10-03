from typing import List, Dict
from nestful import Catalog, API
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.mappings import (
    Mapping,
    MappingCandidate,
)
from sentence_transformers import SentenceTransformer, util

import string


def split_camel_case(text: str) -> List[str]:
    words: List[str] = []
    current_word = ""

    for i, char in enumerate(text):
        if char.isupper() and i != 0:
            words.append(current_word)
            current_word = char
        else:
            current_word += char

    words.append(current_word)
    return words


def split_variable_name(name: str) -> List[str]:
    name_without_whitespace = name.replace(" ", "")
    name_split_with_underscore = name_without_whitespace.split("_")

    split_the_dots: List[str] = []
    for item in name_split_with_underscore:
        tmp_split = item.split(".")
        split_the_dots.extend(tmp_split)

    new_split: List[str] = []
    for item in split_the_dots:
        de_camel_words = split_camel_case(item)
        new_split.extend(de_camel_words)

    return new_split


class Mapper:
    def __init__(self) -> None:
        print("Loading Sentence Transformer")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading NLTK")
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        import nltk

        nltk.download("stopwords")
        nltk.download("punkt_tab")

        self.word_tokenize = word_tokenize
        self.stop_words = set(stopwords.words("english"))

    def process_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        text_tokens = self.word_tokenize(text)
        filtered_tokens = [w for w in text_tokens if w not in self.stop_words]

        return " ".join(filtered_tokens)

    def construct_description(self, var: str, description: str, api: API) -> str:
        return self.process_text(f"{var} {description} {api.name} {api.description}")

    def cosine_similarity(self, t1: str, t2: str) -> float:
        t1 = self.model.encode(t1)
        t2 = self.model.encode(t2)

        try:
            cos_sim: float = round(util.dot_score(t1, t2)[0].cpu().tolist()[0], 2)
            return (1 + cos_sim) / 2.0

        except Exception as e:
            print(e)
            return 0.0

    def compute_maps_to_candidates(
        self,
        source_items: List[str],
        target_items: List[str],
        top_k: int = 5,
        threshold: float = 0.8,
    ) -> List[Mapping]:
        mappings: List[Mapping] = list()

        for source in source_items:
            temp_maps = [
                Mapping(
                    source_name=source,
                    target_name=reference,
                    probability=self.cosine_similarity(source, reference),
                )
                for reference in target_items
                if source != reference
            ]

            above_threshold_maps = list(
                filter(lambda x: x.probability >= threshold, temp_maps)
            )

            sorted_maps = sorted(
                above_threshold_maps, reverse=True, key=lambda x: x.probability
            )[:top_k]

            mappings.extend(sorted_maps)

        return mappings

    def compute_maps(
        self,
        catalog: Catalog,
        top_k: int = 5,
        threshold: float = 0.8,
    ) -> List[Mapping]:
        mappings: List[Mapping] = list()
        variables: Dict[str, MappingCandidate] = dict()

        for index, api in enumerate(catalog.apis):
            print(f"Processing API: {index + 1}/{len(catalog.apis)}")

            input_parameters = api.get_arguments(required=None)
            output_parameters = api.get_outputs()

            mapping_candidates = input_parameters + output_parameters

            for param in mapping_candidates:
                param_split = split_variable_name(param)
                description = self.process_text(" ".join(param_split))

                variables[param] = MappingCandidate(
                    name=param,
                    description=description,
                    type=None,
                    source=api.name,
                    is_input=param in input_parameters,
                )

        for index, var in enumerate(variables.keys()):
            print(f"Processing variable: {index + 1}/{len(variables.keys())}")
            temp_maps = [
                Mapping(
                    source_name=var,
                    target_name=reference,
                    probability=self.cosine_similarity(
                        variables[var].description,
                        variables[reference].description,
                    ),
                )
                for reference in variables
                if var != reference
                and variables[var].source != variables[reference].source
                and variables[var].type == variables[reference].type
                and (variables[reference].is_input and not variables[var].is_input)
            ]

            above_threshold_maps = list(
                filter(lambda x: x.probability >= threshold, temp_maps)
            )

            sorted_maps = sorted(
                above_threshold_maps, reverse=True, key=lambda x: x.probability
            )[:top_k]

            mappings.extend(sorted_maps)

        return mappings
