import json
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Tuple,
    Union,
)

from pydantic import ValidationError

from altk.pre_tool_reflection_toolkit.sparc.metrics import Metric, MetricPrompt
from altk.pre_tool_reflection_toolkit.sparc.function_calling.metrics.function_call.general import (
    GeneralMetricsPrompt,
)
from altk.pre_tool_reflection_toolkit.sparc.function_calling.metrics.function_selection.function_selection import (
    FunctionSelectionPrompt,
)
from altk.pre_tool_reflection_toolkit.sparc.function_calling.metrics.parameter.parameter import (
    ParameterMetricsPrompt,
)

PromptType = Union[
    GeneralMetricsPrompt,
    FunctionSelectionPrompt,
    ParameterMetricsPrompt,
]


# Enum for prompt kinds
class PromptKind(str, Enum):
    GENERAL = "general"
    FUNCTION_SELECTION = "function_selection"
    PARAMETER = "parameter"


# Map enum → Prompt class
_PROMPT_CLASS_MAP: Dict[PromptKind, Any] = {
    PromptKind.GENERAL: GeneralMetricsPrompt,
    PromptKind.FUNCTION_SELECTION: FunctionSelectionPrompt,
    PromptKind.PARAMETER: ParameterMetricsPrompt,
}


class LoaderError(Exception):
    """Raised when prompt loading fails."""


def load_prompts_from_jsonl(
    path: Union[str, Path],
    kind: PromptKind,
) -> List[PromptType]:
    """
    Load prompts from a JSONL file.

    Args:
        path: .jsonl file path.
        kind: PromptKind value.

    Returns:
        List of PromptType, each with its examples loaded.

    Raises:
        LoaderError on I/O, parse, or validation errors.
    """
    PromptCls = _PROMPT_CLASS_MAP.get(kind)
    if PromptCls is None:
        raise LoaderError(f"Unknown PromptKind: {kind}")

    p = Path(path)
    if not p.is_file():
        raise LoaderError(f"File not found: {path}")

    prompts: List[PromptType] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LoaderError(f"{path}:{lineno} invalid JSON: {e}") from e

        # Extract
        try:
            schema = rec["jsonschema"]
            examples = rec.get("examples", [])
            description = rec.get("description", schema.get("description", ""))
        except KeyError as e:
            raise LoaderError(f"{path}:{lineno} missing key {e}") from e

        # Build metric
        try:
            metric = Metric.from_jsonschema(schema)
            metric.description = description
        except Exception as e:
            raise LoaderError(f"{path}:{lineno} invalid schema: {e}") from e

        # Instantiate prompt
        prompt: MetricPrompt
        try:
            prompt = PromptCls(metric=metric, task_description=metric.description)
        except TypeError:
            prompt = PromptCls(metric=metric)

        # Load examples
        for ex_idx, ex in enumerate(examples, start=1):
            try:
                user_kwargs = ex["user_kwargs"]
                output = ex["output"]
            except KeyError as e:
                raise LoaderError(
                    f"{path}:{lineno}, example {ex_idx} missing {e}"
                ) from e
            try:
                prompt.add_example(user_kwargs, output)
            except (ValidationError, ValueError) as e:
                raise LoaderError(
                    f"{path}:{lineno}, example {ex_idx} invalid: {e}"
                ) from e

        prompts.append(prompt)

    return prompts


def load_prompts_from_list(
    records: Iterable[Dict[str, Any]], kind: PromptKind
) -> List[PromptType]:
    """
    Load prompts from an in-memory list of dicts, same structure as JSONL.

    Args:
        records: Iterable of dicts with keys {schema, thresholds, examples, description}.
        kind: PromptKind value.

    Returns:
        List of PromptType.

    Raises:
        LoaderError on missing data or validation failures.
    """
    PromptCls = _PROMPT_CLASS_MAP.get(kind)
    if PromptCls is None:
        raise LoaderError(f"Unknown PromptKind: {kind}")

    prompts: List[PromptType] = []
    for idx, rec in enumerate(records, start=1):
        # same logic as JSONL loader
        try:
            schema = rec["jsonschema"]
            examples = rec.get("examples", [])
            description = schema.get("description", rec.get("name", ""))
        except KeyError as e:
            raise LoaderError(f"Record {idx} missing key {e}") from e

        try:
            metric = Metric.from_jsonschema(schema)
            metric.description = description
        except Exception as e:
            raise LoaderError(f"Record {idx} invalid schema: {e}") from e

        try:
            prompt = PromptCls(metric=metric, task_description=rec["task_description"])
        except TypeError:
            prompt = PromptCls(metric=metric)

        for ex_idx, ex in enumerate(examples, start=1):
            try:
                # Handle both old format (user_kwargs/output) and new format (trajectory/expected_output)
                if "user_kwargs" in ex and "output" in ex:
                    user_kwargs = ex["user_kwargs"]
                    output = ex["output"]
                elif "trajectory" in ex and "expected_output" in ex:
                    # Convert new format to expected format for trajectory metrics
                    user_kwargs = {
                        "trajectory": ex["trajectory"],
                        "tool_inventory": [],  # Default empty, will be provided at runtime
                    }
                    output = ex["expected_output"]
                else:
                    raise KeyError("user_kwargs/output or trajectory/expected_output")
            except KeyError as e:
                raise LoaderError(f"Record {idx}, example {ex_idx} missing {e}") from e
            try:
                prompt.add_example(user_kwargs, output)
            except (ValidationError, ValueError) as e:
                raise LoaderError(f"Record {idx}, example {ex_idx} invalid: {e}") from e

        prompts.append(prompt)

    return prompts


def load_prompts_from_metrics(
    metrics_with_examples: Iterable[Tuple[Metric, List[Dict[str, Any]]]],
    kind: PromptKind,
) -> List[PromptType]:
    """
    Instantiate prompts directly from Metric objects and example data.

    Args:
        metrics_with_examples: An iterable of (Metric instance, examples) tuples.
            Each examples list item must be a dict with:
              - "user_kwargs": Dict[str, Any]
              - "output": Dict[str, Any]
        kind: Which PromptKind to use (GENERAL, FUNCTION_SELECTION, PARAMETER).

    Returns:
        A list of PromptType, each with its few-shot examples loaded.

    Raises:
        LoaderError: on missing data or validation errors.
    """
    PromptCls = _PROMPT_CLASS_MAP.get(kind)
    if PromptCls is None:
        raise LoaderError(f"Unknown PromptKind: {kind}")

    prompts: List[PromptType] = []
    for idx, (metric, examples) in enumerate(metrics_with_examples, start=1):
        if not isinstance(metric, Metric):
            raise LoaderError(
                f"Item {idx}: expected a Metric instance, got {type(metric)}"
            )

        # Instantiate prompt with the metric's description as task_description
        try:
            prompt = PromptCls(metric=metric, task_description=metric.description)
        except TypeError:
            # Fallback if constructor signature differs
            prompt = PromptCls(metric=metric)

        # Add each provided example
        for ex_idx, ex in enumerate(examples or [], start=1):
            if "user_kwargs" not in ex or "output" not in ex:
                raise LoaderError(
                    f"Metric {metric.name}, example {ex_idx}: "
                    "each example must include 'user_kwargs' and 'output'."
                )
            user_kwargs = ex["user_kwargs"]
            output = ex["output"]
            try:
                prompt.add_example(user_kwargs, output)
            except (ValidationError, ValueError) as e:
                raise LoaderError(
                    f"Metric {metric.name}, example {ex_idx} invalid: {e}"
                ) from e

        prompts.append(prompt)

    return prompts
