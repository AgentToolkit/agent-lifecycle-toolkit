from typing import List, Dict, Any, Optional
from jsonschema import (
    Draft7Validator,
)
import copy

from altk.pre_tool_reflection_toolkit.sparc.function_calling.pipeline.types import (
    ToolCall,
    ToolSpec,
    StaticResult,
    StaticMetricResult,
)

# ----------------------------------------
# Human-readable descriptions for checks
# ----------------------------------------
_STATIC_CHECKS: Dict[str, str] = {
    "non_existent_function": "Function name not found in the provided API specification.",
    "non_existent_parameter": "One or more parameters are not defined for the specified function.",
    "incorrect_parameter_type": "One or more parameters have values whose types don't match the expected types.",
    "missing_required_parameter": "One or more required parameters are missing from the call.",
    "allowed_values_violation": "One or more parameters have values outside the allowed enumeration.",
    "json_schema_validation": "The API call does not conform to the provided JSON Schema.",
    "empty_api_spec": "There are no API specifications provided or they are invalid.",
    "invalid_api_spec": "The API specifications provided are not valid Tool or ToolSpec instances.",
    "invalid_tool_call": "The provided ToolCall is not a valid instance of ToolCall.",
}


def evaluate_static(apis_specs: List[ToolSpec], api_call: ToolCall) -> StaticResult:
    """
    Perform static validation on a single tool call.

    Args:
        apis_specs: Non-empty list of ToolSpec instances (OpenAI spec for ToolCall)
        api_call: Single call to validate: ToolCall instance (OpenAI tool call)

    Returns:
        StaticResult(metrics=..., final_decision=bool)
    """
    if not isinstance(apis_specs, list) or not apis_specs:
        return StaticResult(
            metrics={
                "empty_api_spec": StaticMetricResult(
                    description=_STATIC_CHECKS["empty_api_spec"],
                    valid=False,
                    explanation="No API specifications provided.",
                    correction=None,
                )
            },
            final_decision=False,
        )

    if not all(isinstance(spec, ToolSpec) for spec in apis_specs):
        return StaticResult(
            metrics={
                "invalid_api_spec": StaticMetricResult(
                    description=_STATIC_CHECKS["invalid_api_spec"],
                    valid=False,
                    explanation="Invalid API specifications provided; expected ToolSpec instances (List of ToolSpec).",
                    correction=None,
                )
            },
            final_decision=False,
        )

    if not isinstance(api_call, ToolCall):
        return StaticResult(
            metrics={
                "invalid_tool_call": StaticMetricResult(
                    description=_STATIC_CHECKS["invalid_tool_call"],
                    valid=False,
                    explanation="Invalid ToolCall provided; expected ToolCall instance.",
                    correction=None,
                )
            },
            final_decision=False,
        )

    errors, correction = _check_tool_call(specs=apis_specs, call=api_call)

    # Build metrics results: missing key => valid
    metrics: Dict[str, StaticMetricResult] = {}
    for check_name, desc in _STATIC_CHECKS.items():
        valid = check_name not in errors
        # Add correction only to the incorrect_parameter_type metric if it exists
        metric_correction = (
            correction
            if check_name in ["incorrect_parameter_type", "non_existent_parameter"]
            and correction
            else None
        )
        metrics[check_name] = StaticMetricResult(
            description=desc,
            valid=valid,
            explanation=None if valid else errors.get(check_name),
            correction=metric_correction,
        )
    final_decision = all(m.valid for m in metrics.values())
    return StaticResult(metrics=metrics, final_decision=final_decision)


def _attempt_type_conversion(value: Any, expected_type: str) -> Optional[Any]:
    """
    Attempt to convert a value to the expected type.
    Returns the converted value if successful, None if conversion fails.
    """
    try:
        # Handle None/null values - only convert to string
        if value is None:
            if expected_type == "string":
                return "None"
            else:
                return None  # Cannot convert None to other types

        if expected_type == "string":
            return str(value)
        elif expected_type == "integer":
            if isinstance(value, str):
                # Skip empty or whitespace-only strings
                if not value or value.isspace():
                    return None
                # Try to convert string to int
                if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                    return int(value)
                # Try float to int conversion (including scientific notation)
                try:
                    float_val = float(value)
                    if float_val.is_integer():
                        return int(float_val)
                except ValueError:
                    pass
            elif isinstance(value, float) and value.is_integer():
                return int(value)
            elif isinstance(value, bool):
                return int(value)
        elif expected_type == "number":
            if isinstance(value, str):
                # Skip empty or whitespace-only strings
                if not value or value.isspace():
                    return None
                try:
                    return float(value)
                except ValueError:
                    pass
            elif isinstance(value, int):
                return float(value)
            elif isinstance(value, bool):
                return float(value)
        elif expected_type == "boolean":
            if isinstance(value, str):
                # Skip empty strings
                if not value:
                    return None
                lower_val = value.lower().strip()
                if lower_val in ("true", "1", "yes", "on"):
                    return True
                elif lower_val in ("false", "0", "no", "off"):
                    return False
            elif isinstance(value, (int, float)):
                return bool(value)
        elif expected_type == "array" and not isinstance(value, list):
            # Only convert simple values to arrays, not complex objects
            if not isinstance(value, dict):
                return [value]
    except (ValueError, TypeError, AttributeError, OverflowError):
        pass

    return None


def _check_tool_call(
    specs: List[ToolSpec], call: ToolCall
) -> tuple[Dict[str, str], Optional[Dict[str, Any]]]:
    """
    Static checks for OpenAI ToolCall + ToolSpec list.
    Returns tuple of (failed check keys -> explanation, correction if type conversion succeeded).
    """
    errors: Dict[str, str] = {}
    correction: Optional[Dict[str, Any]] = None

    # 1) Function existence
    spec = next((s for s in specs if s.function.name == call.function.name), None)
    if not spec:
        errors["non_existent_function"] = (
            f"Function '{call.function.name}' does not exist in the provided API specifications:"
            f" {', '.join(s.function.name for s in specs)}."
        )
        return errors, None

    params_schema = spec.function.parameters
    properties = params_schema.get("properties", params_schema)
    parsed_arguments = call.function.parsed_arguments

    # 2) Parameter existence check
    if non_existent_params := set(parsed_arguments.keys()) - set(properties.keys()):
        errors["non_existent_parameter"] = (
            f"Parameters not defined in function '{call.function.name}': "
            f"{', '.join(sorted(non_existent_params))}. "
            f"Possible parameters are: {', '.join(sorted(properties.keys()))}."
        )
        corrected_parameters = {
            param: parsed_arguments[param]
            for param in parsed_arguments.keys()
            if param not in non_existent_params
        }
        correction = {
            "corrected_arguments": corrected_parameters,
            "tool_call": {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": corrected_parameters,
                },
            },
        }

    # 3) JSON Schema validation with type conversion attempt
    validator = Draft7Validator(params_schema)
    corrected_arguments = copy.deepcopy(parsed_arguments)
    type_corrections_made = False

    missing_required = []
    incorrect_types = []
    invalid_enum = []
    other_errors = []

    for error in validator.iter_errors(parsed_arguments):
        field = ".".join(str(x) for x in error.path) if error.path else "unknown"
        if error.validator == "required":
            missing_required.append(error.message)
        elif error.validator == "type":
            # Attempt type conversion for type errors
            param_name = error.path[0] if error.path else None
            if param_name and param_name in parsed_arguments:
                incorrect_types.append(f"{field}: {error.message}")
                current_value = parsed_arguments[param_name]
                expected_type = error.schema.get("type")

                if expected_type:
                    converted_value = _attempt_type_conversion(
                        current_value, expected_type
                    )
                    if converted_value is not None:
                        corrected_arguments[param_name] = converted_value
                        type_corrections_made = True
                        # Don't add to incorrect_types if conversion succeeded
                        continue

        elif error.validator == "enum":
            invalid_enum.append(f"{field}: {error.message}")
        else:
            other_errors.append(f"{field}: {error.message}")

    # If type corrections were made, validate the corrected arguments
    if type_corrections_made:
        corrected_errors = list(validator.iter_errors(corrected_arguments))
        # Filter out type errors that were successfully corrected
        remaining_type_errors = []
        for error in corrected_errors:
            field = ".".join(str(x) for x in error.path) if error.path else "unknown"
            if error.validator == "type":
                remaining_type_errors.append(f"{field}: {error.message}")

        # If all type errors were resolved, create correction
        if not remaining_type_errors:
            correction = {
                "corrected_arguments": corrected_arguments,
                "tool_call": {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": corrected_arguments,
                    },
                },
            }

    if missing_required:
        errors["missing_required_parameter"] = (
            "Missing required parameter(s): " + "; ".join(missing_required)
        )
    if incorrect_types:
        errors["incorrect_parameter_type"] = (
            "Incorrect parameter type(s): " + "; ".join(incorrect_types)
        )
    if invalid_enum:
        errors["allowed_values_violation"] = "Invalid parameter value(s): " + "; ".join(
            invalid_enum
        )
    if other_errors:
        errors["json_schema_validation"] = "Other validation error(s): " + "; ".join(
            other_errors
        )

    return errors, correction
