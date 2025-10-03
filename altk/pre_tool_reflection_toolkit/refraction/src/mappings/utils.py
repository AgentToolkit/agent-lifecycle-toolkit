from typing import List, Dict
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import Mapping
from nestful.utils import extract_label
from nestful import (
    Catalog,
    SequencingDataset,
    SequencingData,
    SequenceStep,
)


def cache_maps(
    sequence_data: SequencingDataset | List[SequencingData],
) -> List[Mapping]:
    mappings = []

    if isinstance(sequence_data, SequencingDataset):
        sequence_data = sequence_data.data

    for sequence in sequence_data:
        new_maps = cache_maps_from_sequence(sequence)
        mappings.extend(new_maps)

    return mappings


def cache_maps_from_sequence(sequence: SequencingData) -> List[Mapping]:
    mappings = []

    for step in sequence.output:
        new_maps = cache_maps_from_sequence_step(step)
        mappings.extend(new_maps)

    return mappings


def cache_maps_from_sequence_step(sequence_step: SequenceStep) -> List[Mapping]:
    mappings = []

    for arg in sequence_step.arguments:
        source = str(sequence_step.arguments[arg])
        label, mapping = extract_label(source)

        if label and mapping and mapping != arg:
            new_map = Mapping(source_name=mapping, target_name=arg, probability=1.0)

            mappings.append(new_map)

    return mappings


def merge_maps(mappings: List[Mapping]) -> List[Mapping]:
    map_of_mappings: Dict[str, float] = dict()
    merged_maps: List[Mapping] = list()

    for mapping in mappings:
        index = f"{mapping.source_name}_{mapping.target_name}"
        current_probability = map_of_mappings.get(index, 0.0)

        if mapping.probability > current_probability:
            map_of_mappings[index] = mapping.probability
            merged_maps.append(mapping)

    return merged_maps


def filter_maps(catalog: Catalog, mappings: List[Mapping]) -> List[Mapping]:
    list_of_variables: List[str] = list()

    for api in catalog.apis:
        parameter_dict = {**api.query_parameters, **api.output_parameters}
        list_of_variables.extend(parameter_dict.keys())

    return [
        mapping
        for mapping in mappings
        if mapping.target_name in list_of_variables
        or mapping.source_name in list_of_variables
    ]
