import pytest
from typing import List, Dict, Any
from tests.utils.refraction.mellea.prompt import PROMPT_TEMPLATE
from tests.utils.refraction.tools.sample_tool_specs import tools
from altk.pre_tool_reflection_toolkit.refraction.src import refract
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
from altk.pre_tool_reflection_toolkit.refraction.src.prompt_template import (
    generate_prompt,
    PromptType,
)

try:
    from mellea import start_session
    from mellea.stdlib.requirement import ValidationResult
    from mellea.stdlib.sampling import RejectionSamplingStrategy
    from mellea.stdlib.base import Context
    from altk.pre_tool_reflection_toolkit.refraction.src.integration.mellea_requirement import (
        RefractionRequirement,
    )
except ImportError:
    pytest.mark.skip(reason="mellea not available")
    ValidationResult = None
    RejectionSamplingStrategy = None
    Context = None
    RefractionRequirement = None  # type: ignore

# TODO: No need for a mellea integration to be a test. Remove this and move it to examples maybe
# For now, set as extra test
pytestmark = pytest.mark.refract_extra


class TestMellea:
    def setup_method(self) -> None:
        self.mellea_session = start_session()
        self.refractor_req = RefractionRequirement(tools=tools)

        # NOTE: This step is optional and done once up front, if you want the refractor
        # to guess mappings between input and output fields of available tools
        # self.refractor_req.initialize_maps(mapping_threshold=0.85)

    @pytest.mark.ignore("To clear integrations after first refactor.")
    def test_basic_req_pass(self) -> None:
        inference_result = self.mellea_session.instruct(
            description=PROMPT_TEMPLATE,
            requirements=[
                self.refractor_req,
            ],
            strategy=RejectionSamplingStrategy(loop_budget=1),
            user_variables={
                "query": (
                    "I need a travel approval to present my conference papers."
                    " My email is tchakra2@ibm.com"
                ),
                "tools": tools,
                "memory": {},
            },
            return_sampling_results=True,
        )

        assert inference_result.success is True
        print(inference_result.result)

    @pytest.mark.ignore("To clear integrations after first refactor.")
    def test_basic_req_fail(self) -> None:
        tmp_refractor_req = RefractionRequirement(
            tools=tools, validation_fn=corrupted_refract
        )

        tmp_refractor_req.initialize_maps(mapping_threshold=0.65)

        inference_result = self.mellea_session.instruct(
            description=PROMPT_TEMPLATE,
            requirements=[
                tmp_refractor_req,
            ],
            user_variables={
                "query": (
                    "I need a travel approval to present my conference papers."
                    " My email is tchakra2@ibm.com"
                ),
                "tools": tools,
                "memory": {},
            },
            strategy=RejectionSamplingStrategy(loop_budget=1),
            return_sampling_results=True,
        )

        assert inference_result.success is False
        print(f"Expect sub-par result. {inference_result.sample_generations[0].value}")

    @pytest.mark.ignore("No longer accepts repair function :-/.")
    def test_req_with_sampling(self) -> None:
        tmp_refractor_req = RefractionRequirement(
            tools=tools, validation_fn=corrupted_refract
        )
        tmp_refractor_req.initialize_maps(mapping_threshold=0.65)

        inference_result = self.mellea_session.instruct(
            description=PROMPT_TEMPLATE,
            requirements=[
                tmp_refractor_req,
            ],
            user_variables={
                "query": (
                    "I need a travel approval to present my conference papers."
                    " My email is tchakra2@ibm.com"
                ),
                "tools": tools,
                "memory": {},
            },
            strategy=RejectionSamplingStrategy(loop_budget=2),
            return_sampling_results=True,
        )

        assert inference_result.success is False

        corrupted_tool_calls = make_corrupted_call()
        refraction_result = refract(
            corrupted_tool_calls, tools, mappings=tmp_refractor_req.mappings
        )

        reason = generate_prompt(
            refraction_result,
            corrupted_tool_calls,
            catalog=tools,
            memory_objects={},
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )

        assert reason == inference_result.sample_validations[0][0][1].reason


def make_corrupted_call() -> List[Dict[str, Any]]:
    from tests.utils.refraction.tools.sample_tool_specs import tool_calls

    corrupted_tool_calls = []

    for og_call in tool_calls:
        if og_call.get("name") == "concur":
            # NOTE: Remove one of the parameters to test tool calling error
            if "employee_info" in og_call["arguments"]:
                del og_call["arguments"]["employee_info"]  # type: ignore

        corrupted_tool_calls.append(og_call)

    return corrupted_tool_calls


def corrupted_refract(context: Context, refractor_class: Refractor) -> ValidationResult:
    _ = context
    # NOTE: In the actual requirement this is extracted from the response
    tool_calls = make_corrupted_call()

    refraction_result = refractor_class.refract(tool_calls, memory_objects={})
    reason = generate_prompt(
        refraction_result,
        tool_calls,
        catalog=refractor_class.catalog,
        memory_objects={},
        prompt_type=PromptType.WITH_SUGGESTIONS,
    )

    return ValidationResult(
        result=refraction_result.report.determination is True, reason=reason
    )
