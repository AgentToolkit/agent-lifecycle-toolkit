from jinja2 import meta, Environment
from typing import Dict, Any


def remove_threshold_fields(schema: dict) -> dict:
    """
    Recursively removes 'threshold_low' and 'threshold_high' fields from a JSON schema.
    """
    if isinstance(schema, dict):
        # Remove the threshold fields if present
        schema.pop("threshold_low", None)
        schema.pop("threshold_high", None)
        # Recurse into nested dictionaries and lists
        for key, value in schema.items():
            if isinstance(value, dict):
                schema[key] = remove_threshold_fields(value)
            elif isinstance(value, list):
                schema[key] = [
                    remove_threshold_fields(item) if isinstance(item, dict) else item
                    for item in value
                ]
    return schema


def validate_template_context(
    env: Environment,
    template_str: str,
    context: Dict[str, Any],
    template_name: str = "",
):
    parsed = env.parse(template_str)
    required_vars = meta.find_undeclared_variables(parsed)

    # Allow custom_instructions, tool_specification, and custom_schema to be optional since they're in conditional blocks
    optional_vars = {"custom_instructions", "tool_specification", "custom_schema"}

    missing_or_empty = [
        var
        for var in required_vars
        if var not in optional_vars
        and (var not in context or context[var] in (None, [], {}, ()))
    ]
    if missing_or_empty:
        raise ValueError(
            f"Missing or empty variables in template '{template_name or 'unnamed'}': {missing_or_empty}"
        )
