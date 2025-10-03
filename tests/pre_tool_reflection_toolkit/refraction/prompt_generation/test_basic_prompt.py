from nestful.data_handlers import get_nestful_catalog
from nestful import SequencingData, SequenceStep
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    PromptType,
    DebuggingResult,
)
from altk.pre_tool_reflection_toolkit.refraction.src.utils import pprint
from altk.pre_tool_reflection_toolkit.refraction.src import refract, generate_prompt
from altk.pre_tool_reflection_toolkit.refraction.src.prompt_template import (
    stringify_list_of_fixes,
)
from typing import Optional
from pathlib import Path
from inspect import stack
from tests.utils.refraction.utils import get_cached_maps

import pytest

pytestmark = pytest.mark.refract_extra
# These tests take a while, going to set as extra


class TestBasicPrompt:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)
        self.mappings = get_cached_maps()

        self.memory = {
            "var1": {
                "skyId": "foo",
                "entityId": "bar",
            },
            "var2": {
                "skyId": "baz",
                "entityId": "qux",
            },
            "var4": {
                "geoId": "BOS",
            },
        }

        self.sequence_missing_parameters = [
            'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
            ' originEntityId="$var1.entityId$", date="2024-08-15")'
        ]

        self.result_missing_parameters = refract(
            self.sequence_missing_parameters,
            catalog=self.catalog,
            mappings=self.mappings,
            memory_objects=self.memory,
        )

        self.sequence_made_up_parameter = [
            'var3 = TripadvisorSearchHotels(location="$var4.geoId",'
            ' checkIn="2024-08-15", checkOut="2024-08-18")'
        ]

        self.result_made_up_parameter = refract(
            self.sequence_made_up_parameter,
            catalog=self.catalog,
            mappings=self.mappings,
            memory_objects=self.memory,
        )

        self.multiple_results = [
            self.result_missing_parameters,
            DebuggingResult(),
            refract(
                self.sequence_missing_parameters,
                catalog=self.catalog,
                memory_objects=self.memory,
            ),
        ]

    @staticmethod
    def check_with_cached_prompt(prompt: str, file_name: Optional[str] = None) -> None:
        file_name = file_name or f"{stack()[1][3].replace('test_', '')}.txt"

        path_to_file = Path(__file__).parent.resolve()
        relative_path_to_data = f"./cached_prompts/{file_name}"

        abs_path_to_data = Path.joinpath(path_to_file, relative_path_to_data).resolve()

        pprint(prompt)

        cached_prompt = open(abs_path_to_data).read()
        assert cached_prompt == prompt

    def test_no_help(self) -> None:
        prompt = generate_prompt(
            self.result_missing_parameters,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
        )

        self.check_with_cached_prompt(prompt)

    def test_basic_equality(self) -> None:
        sequence_missing_and_made_up_parameters = [
            'var3 = SkyScrapperFlightSearch(originSkyAyeId="$var1.skyId$",'
            ' originEntityId="$var1.entityId$", date="2024-08-15")'
        ]

        result = refract(
            sequence_missing_and_made_up_parameters,
            catalog=self.catalog,
            mappings=self.mappings,
            memory_objects=self.memory,
        )

        assert generate_prompt(
            self.result_missing_parameters,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
        ) == generate_prompt(
            result,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
        )

    def test_with_fix(self) -> None:
        prompt = generate_prompt(
            self.result_missing_parameters,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_FIX,
        )

        self.check_with_cached_prompt(prompt)

    def test_sanity_check_missing_parameter(self) -> None:
        prompt = generate_prompt(
            self.result_missing_parameters,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.SANITY_CHECK,
        )

        self.check_with_cached_prompt(prompt)

    def test_sanity_check_missing_parameter_fix_only(self) -> None:
        fixes = stringify_list_of_fixes(
            self.sequence_missing_parameters,
            self.result_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.SANITY_CHECK,
        )

        self.check_with_cached_prompt("\n".join(fixes))

    def test_sanity_check_made_up_parameter(self) -> None:
        prompt = generate_prompt(
            self.result_made_up_parameter,
            self.sequence_made_up_parameter,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.SANITY_CHECK,
        )

        self.check_with_cached_prompt(prompt)

    def test_sanity_check_multiple(self) -> None:
        prompt = generate_prompt(
            self.multiple_results,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.SANITY_CHECK,
        )

        self.check_with_cached_prompt(prompt)

    def test_with_suggestions_missing_parameter(self) -> None:
        prompt = generate_prompt(
            self.result_missing_parameters,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        self.check_with_cached_prompt(prompt)

    def test_with_suggestions_made_up_parameter(self) -> None:
        prompt = generate_prompt(
            self.result_made_up_parameter,
            self.sequence_made_up_parameter,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        self.check_with_cached_prompt(prompt)

    def test_with_suggestions_made_up_parameter_fix_only(self) -> None:
        fixes = stringify_list_of_fixes(
            self.sequence_made_up_parameter,
            self.result_made_up_parameter,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        self.check_with_cached_prompt("\n".join(fixes))

    def test_with_suggestions_recovery_call(self) -> None:
        del self.memory["var4"]

        sequence = SequencingData(
            input="Book a hotel in Boston.",
            output=[
                SequenceStep(
                    name="TripadvisorSearchHotels",
                    label="var3",
                    arguments={
                        "checkIn": "2024-08-15",
                        "checkOut": "2024-08-18",
                    },
                )
            ],
        )

        result = refract(
            sequence,
            catalog=self.catalog,
            mappings=self.mappings,
            memory_objects=self.memory,
        )

        prompt = generate_prompt(
            result,
            sequence,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        self.check_with_cached_prompt(prompt)

    def test_with_suggestions_slot_fill(self) -> None:
        del self.memory["var4"]

        result = refract(
            self.sequence_made_up_parameter,
            catalog=self.catalog,
            mappings=self.mappings,
            memory_objects=self.memory,
        )

        prompt = generate_prompt(
            result,
            self.sequence_made_up_parameter,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        self.check_with_cached_prompt(prompt)

    def test_with_suggestions_multiple_fixes(self) -> None:
        prompt = generate_prompt(
            self.multiple_results,
            self.sequence_missing_parameters,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        self.check_with_cached_prompt(prompt)

    def test_with_failed_refraction(self) -> None:
        prompt_1 = generate_prompt(
            result=self.result_made_up_parameter,
            sequence=self.sequence_made_up_parameter,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.SANITY_CHECK,
        )

        prompt_2 = generate_prompt(
            result=[DebuggingResult(), self.result_made_up_parameter],
            sequence=self.sequence_made_up_parameter,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.SANITY_CHECK,
        )

        assert prompt_1 == prompt_2

    def test_sanity_check_sequence(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.locationId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog, mappings=self.mappings)

        prompt = generate_prompt(
            result,
            tokens,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.SANITY_CHECK,
        )

        self.check_with_cached_prompt(prompt)

    def test_with_suggestions_sequence(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.locationId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog, mappings=self.mappings)

        prompt = generate_prompt(
            result,
            tokens,
            catalog=self.catalog,
            memory_objects=self.memory,
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        self.check_with_cached_prompt(prompt)
