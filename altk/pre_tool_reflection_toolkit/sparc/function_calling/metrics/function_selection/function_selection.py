from altk.pre_tool_reflection_toolkit.sparc.function_calling.metrics.base import (
    FunctionMetricsPrompt,
)

_function_system = (
    "### Task Description:\n\n"
    "{{ task_description }}\n\n"
    "Your output must conform to the following JSON schema, in the same order as the fields appear in the schema:\n"
    "{{ metric_jsonschema }}"
)

_function_user = (
    "Conversation context:\n"
    "{{ conversation_context }}\n\n"
    "Tools Inventory:\n"
    "{{ tools_inventory }}\n\n"
    "Proposed function call:\n"
    "{{ proposed_tool_call }}\n\n"
    "Function name:\n"
    "{{ selected_function }}\n\n"
    "Return a JSON object as specified in the system prompt. You MUST keep the same order of fields in the JSON object as provided in the JSON schema and examples."
)


class FunctionSelectionPrompt(FunctionMetricsPrompt):
    """Prompt builder for function-selection metrics."""

    system_template = _function_system
    user_template = _function_user
