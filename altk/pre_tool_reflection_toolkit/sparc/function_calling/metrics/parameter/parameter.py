from typing import Any, Dict, List, Union
from altk.pre_tool_reflection_toolkit.sparc.function_calling.metrics.base import (
    FunctionMetricsPrompt,
)

_parameter_system: str = (
    "### Task Description:\n\n"
    "{{ task_description }}\n\n"
    "Your output must conform to the following JSON schema, in the same order as the fields appear in the schema:\n"
    "{{ metric_jsonschema }}"
)

_parameter_user: str = (
    "Conversation context:\n"
    "{{ conversation_context }}\n\n"
    "Tool Specification:\n"
    "{{ tool_inventory }}\n\n"
    "Proposed tool call:\n"
    "{{ tool_call }}\n\n"
    "Parameter name:\n"
    "{{ parameter_name }}\n\n"
    "Parameter value:\n"
    "{{ parameter_value }}\n\n"
    "Return a JSON object as specified in the system prompt. You MUST keep the same order of fields in the JSON object as provided in the JSON schema and examples."
)


class ParameterMetricsPrompt(FunctionMetricsPrompt):
    """Prompt builder for parameter-level metrics."""

    system_template = _parameter_system
    user_template = _parameter_user


def get_parameter_metrics_prompt(
    prompt: ParameterMetricsPrompt,
    conversation_context: Union[str, List[Dict[str, str]]],
    tool_inventory: List[Dict[str, Any]],
    tool_call: Dict[str, Any],
    parameter_name: str,
    parameter_value: Any,
) -> List[Dict[str, str]]:
    """
    Build the messages for a parameter-level evaluation.
    """
    return prompt.build_messages(
        user_kwargs={
            "conversation_context": conversation_context,
            "tool_inventory": tool_inventory,
            "tool_call": tool_call,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
        }
    )
