from tests.utils.refraction.tools.sample_tool_specs_untyped import (
    tool_calls,
    tool_calls_internal,
)
from tests.utils.refraction.tools.sample_tool_specs import (
    tool_calls as tool_calls_execution,
)
from tests.utils.refraction.mellea.prompt import generate_response

from altk.pre_tool_reflection_toolkit.refraction.src.integration.utils import (
    extract_tool_calls,
)
from nestful.schemas.tools import ToolCall, OpenAIToolCall


class TestExtraction:
    def test_extract_tool_calls(self) -> None:
        for test_case in [
            tool_calls,
            tool_calls_internal,
            tool_calls_execution,
        ]:
            response = generate_response(test_case)
            extracted_calls = extract_tool_calls(response)

            for item in extracted_calls:
                (
                    OpenAIToolCall.model_validate(item)
                    if test_case == tool_calls
                    else ToolCall.model_validate(item)
                )

            assert len(extracted_calls) == len(test_case)
