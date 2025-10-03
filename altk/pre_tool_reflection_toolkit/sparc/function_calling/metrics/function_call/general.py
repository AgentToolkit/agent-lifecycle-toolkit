from typing import Any, Dict, List, Union
from altk.pre_tool_reflection_toolkit.sparc.function_calling.metrics.base import (
    FunctionMetricsPrompt,
)

_general_system = (
    "### Task Description and Role:\n\n"
    "{{ task_description }}\n\n"
    "Your output must conform to the following JSON schema, in the same order as the fields appear in the schema:\n"
    "{{ metric_jsonschema }}"
)

_general_user: str = (
    "Conversation context:\n"
    "{{ conversation_context }}\n\n"
    "Tool Specification:\n"
    "{{ tool_inventory }}\n\n"
    "Proposed tool call:\n"
    "{{ tool_call }}\n\n"
    "Return a JSON object as specified in the system prompt. You MUST keep the same order of fields in the JSON object as provided in the JSON schema and examples."
)


class GeneralMetricsPrompt(FunctionMetricsPrompt):
    """Prompt builder for general tool-call semantic metrics."""

    system_template = _general_system
    user_template = _general_user


def get_general_metrics_prompt(
    prompt: GeneralMetricsPrompt,
    conversation_context: Union[str, List[Dict[str, str]]],
    tool_inventory: List[Dict[str, Any]],
    tool_call: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Build the messages for a general semantic evaluation.

    Returns the list of chat messages (system -> [few-shot] -> user).
    """
    return prompt.build_messages(
        user_kwargs={
            "conversation_context": conversation_context,
            "tool_inventory": tool_inventory,
            "tool_call": tool_call,
        }
    )
