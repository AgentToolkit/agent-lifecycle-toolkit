from __future__ import annotations
import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Literal,
)
from types import NoneType
from typing_extensions import Self

from pydantic import BaseModel, Field, model_validator, ValidationError

from altk.pre_tool_reflection_toolkit.sparc.metrics import MetricRunResult

# ----------------------------------------------------------------------
# 1. Function-Call Metric Models
# ----------------------------------------------------------------------


class FunctionCallMetric(BaseModel):
    """
    Function-call metric: a single metric name, schema, and examples.
    """

    name: str = Field(
        ..., description="Name of the metric (e.g. 'function_selection')."
    )
    task_description: str = Field(
        ..., description="Description of the task this metric is evaluating."
    )
    jsonschema: Dict[str, Any] = Field(
        ..., description="JSON Schema dict for this metric's output."
    )
    examples: Optional[List[Dict[Literal["user_kwargs", "output"], Any]]] = Field(
        None,
        description=(
            "List of example inputs and outputs for this metric; "
            "each example is a dict with 'user_kwargs' and 'output' keys."
        ),
    )


# ----------------------------------------------------------------------
# 2. Static-Check Models (Optional)
# ----------------------------------------------------------------------


class StaticMetricResult(BaseModel):
    """
    Result of a single static (schema-based) check.
    """

    description: str = Field(
        ..., description="Human-readable description of this static validation check."
    )
    valid: bool = Field(
        ..., description="True if this static check passed; False otherwise."
    )
    explanation: Optional[str] = Field(
        None,
        description=(
            "If valid==False, a detailed explanation of why the check failed; "
            "otherwise None."
        ),
    )
    correction: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "If type conversion was successful, contains the corrected tool call; "
            "otherwise None."
        ),
    )


class StaticResult(BaseModel):
    """
    Aggregated results of static (schema-based) checks for one function call.
    """

    metrics: Dict[str, StaticMetricResult] = Field(
        ...,
        description=("Mapping from each static-check name to its StaticMetricResult."),
    )
    final_decision: bool = Field(
        ...,
        description=(
            "Overall outcome: False if any metric.valid is False; True only if all pass."
        ),
    )


# ----------------------------------------------------------------------
# 3. Semantic Metric Result Models
# ----------------------------------------------------------------------


class SemanticMetricResult(BaseModel):
    """
    Wraps a single metric evaluation result returned by MetricRunner.
    """

    metric_name: str = Field(
        ..., description="Identifier (name) of the evaluated metric."
    )
    jsonschema: Dict[str, Any] = Field(
        ..., description="JSON Schema dict that was used to validate output."
    )
    prompt: Union[str, List[Dict[str, str]]] = Field(
        ...,
        description=(
            "The actual prompt sent to the LLM—either a plain string "
            "or a list of {'role','content'} messages."
        ),
    )
    raw_response: Any = Field(
        ..., description="Raw response returned by the LLM client."
    )
    numeric_thresholds_checks: Dict[str, bool] = Field(
        ...,
        description=(
            "For every numeric field in the metric, a boolean indicating "
            "whether the parsed value fell within its [low, high] thresholds."
        ),
    )
    is_important: bool = Field(
        ...,
        description=(
            "True if the metric's confidence field met its importance threshold; "
            "False otherwise."
        ),
    )
    importance_reason: Optional[str] = Field(
        None,
        description=(
            "If is_important==False, a textual reason (e.g. 'confidence too low'); "
            "otherwise None."
        ),
    )
    error: Optional[str] = Field(
        None,
        description=(
            "Error message if prompt generation or parsing failed; otherwise None."
        ),
    )
    is_correct: bool = Field(
        ...,
        description=(
            "True if both importance and the metric's primary value field "
            "fell within thresholds; False otherwise."
        ),
    )
    correctness_reason: Optional[str] = Field(
        None,
        description=(
            "If is_correct==False, a textual reason why the value or confidence "
            "fell outside thresholds; otherwise None."
        ),
    )
    is_issue: bool = Field(
        ...,
        description=(
            "True if is_correct==False and is_important==True; False otherwise."
        ),
    )

    @classmethod
    def from_runner(cls, rr: MetricRunResult) -> "SemanticMetricResult":
        """
        Construct from an internal MetricRunResult instance.
        """
        # first construct the object from what MetricRunner gave us
        data = rr.model_dump()
        inst: SemanticMetricResult = cls(**data)

        return inst

    @property
    def output_value(self) -> Optional[float]:
        """
        Convenience accessor for the metric's primary 'output' numeric field,
        if present and parsed successfully.
        """
        if self.raw_response and isinstance(
            self.raw_response.get("output"), (int, float)
        ):
            return float(self.raw_response["output"])
        return None

    @property
    def normalized_output(self) -> Optional[float]:
        """
        Linearly scale 'output' into [0,1] according to its schema min/max.
        """
        out = self.output_value
        subs = self.jsonschema.get("properties", {}).get("output", {})
        low = subs.get("minimum", 0.0)
        high = subs.get("maximum", 1.0)
        if out is None or high == low:
            return None
        return (out - low) / (high - low)


class SemanticCategoryResult(BaseModel):
    """
    Collection of SemanticMetricResults for a single category:
      - general
      - function_selection
      - parameter
    """

    metrics: Optional[Dict[str, SemanticMetricResult]] = Field(
        None,
        description=("Mapping metric_name -> SemanticMetricResult for this category."),
    )
    avg_score: Optional[float] = Field(
        None,
        description=(
            "Average of the 'output' values across all metrics whose "
            "confidence was within thresholds (is_important==True)."
        ),
    )

    @classmethod
    def from_results(cls, results: List[MetricRunResult]) -> "SemanticCategoryResult":
        """
        Build a category result from a list of MetricRunResult objects.
        """
        # 1) build per-metric results
        mapping: Dict[str, SemanticMetricResult] = {
            r.metric_name: SemanticMetricResult.from_runner(r) for r in results
        }

        # 2) compute normalized‐output average over 'important' metrics only
        norms: List[float] = []
        for m in mapping.values():
            norm = m.normalized_output
            if norm is not None and m.is_important:
                norms.append(norm)

        avg = (sum(norms) / len(norms)) if norms else None
        return cls(metrics=mapping, avg_score=avg)


class SemanticResult(BaseModel):
    """
    Aggregated semantic metrics across all categories for one function call.
    """

    general: Optional[SemanticCategoryResult] = Field(
        None,
        description=("Results of general tool-call metrics, if any; otherwise None."),
    )
    function_selection: Optional[SemanticCategoryResult] = Field(
        None,
        description=("Results of function-selection metrics, if any; otherwise None."),
    )
    parameter: Optional[Dict[str, SemanticCategoryResult]] = Field(
        None,
        description=(
            "Parameter-level results, keyed by parameter name, each with its metrics."
        ),
    )
    transform: Optional[Dict[str, TransformResult]] = Field(
        None,
        description=(
            "Optional per-parameter transformation results: "
            "mapping parameter_name -> TransformResult."
        ),
    )


# ----------------------------------------------------------------------
# 4. Transformation Result Model
# ----------------------------------------------------------------------


class TransformResult(BaseModel):
    """
    Result of unit-extraction and code-based transformation checks for one parameter.
    """

    units: Dict[str, Any] = Field(
        ...,
        description=(
            "Extracted unit info: keys 'user_units', 'user_value', and 'spec_units'."
        ),
    )
    generated_code: str = Field(
        ...,
        description="The Python code snippet returned by the LLM for unit conversion.",
    )
    execution_success: bool = Field(
        ...,
        description="True if generated_code executed without error and matched values.",
    )
    correct: bool = Field(
        ...,
        description=(
            "False if execution_success is True but the transformation "
            "was incorrect; True if the transformation was correct or was not executed."
        ),
    )
    execution_output: Any = Field(
        None, description="The actual output of executing the transformation code."
    )
    correction: Optional[str] = Field(
        None,
        description="Correction explanation if execution succedded but the transformation was incorrect.",
    )
    error: Optional[str] = Field(
        None,
        description=(
            "Error message if code generation or execution failed; otherwise None."
        ),
    )


# ----------------------------------------------------------------------
# 5. Pipeline I/O Models
# ----------------------------------------------------------------------


class FunctionCallInput(BaseModel):
    """
    Input bundle for the function-calling pipeline.
    """

    conversation_context: Union[str, List[Dict]] = Field(
        ...,
        description=(
            "Either a single user text string or a list of chat messages "
            "with {'role','content'}."
        ),
    )
    tools_inventory: List[ToolSpec] = Field(
        ...,
        description=(
            "List of available tools; each entry must at least include "
            "'name' and argument schema."
        ),
    )
    tool_call: ToolCall = Field(
        ...,
        description=(
            "Proposed function call dict: {\n"
            "  'name': '<function_name>',\n"
            "  'args': {<param>:<value>, ...}\n"
            "}."
        ),
    )


class TrajectoryPipelineInput(BaseModel):
    """
    Input bundle for the trajectory evaluation pipeline.
    """

    trajectory: Union[str, List[Dict]] = Field(
        ...,
        description="The full conversation trajectory, including all user, assistant, and tool messages.",
    )
    tool_inventory: List[ToolSpec] = Field(
        ...,
        description="List of available tools; each entry must at least include 'name' and argument schema.",
    )


class TrajectoryPipelineResult(BaseModel):
    """
    Final output of the trajectory evaluation pipeline.
    """

    inputs: TrajectoryPipelineInput = Field(
        ..., description="Echo of the pipeline inputs."
    )
    result: Optional[SemanticCategoryResult] = Field(
        None, description="Trajectory evaluation results, if any; otherwise None."
    )


class PipelineResult(BaseModel):
    """
    Final output of the function-calling pipeline for one tool call.
    """

    inputs: FunctionCallInput = Field(..., description="Echo of the pipeline inputs.")
    static: Optional[StaticResult] = Field(
        None, description="Static schema-validation results, if enabled."
    )
    semantic: SemanticResult = Field(
        ..., description="All semantic metric results by category."
    )
    overall_valid: bool = Field(
        ...,
        description=(
            "True if all semantic metrics passed (is_correct==True) "
            "and, if present, all transformations succeeded."
        ),
    )
    overall_avg_score: Optional[float] = Field(
        None,
        description=(
            "Average of the three category avg_scores "
            "(general, function_selection, parameter) where available."
        ),
    )

    @model_validator(mode="after")
    def compute_overall(self) -> Self:
        """
        After validation, compute overall_valid as AND of:
          • all semantic is_correct flags
          • if transform exists: all execution_success flags
        """
        static: StaticResult = self.static
        if static:
            # static checks
            ok = static.final_decision

        sem: SemanticResult = self.semantic
        if sem:
            # semantic checks
            if sem.general and sem.general.metrics:
                for m in sem.general.metrics.values():
                    if not m.is_correct:
                        ok = False
            if sem.function_selection and sem.function_selection.metrics:
                for m in sem.function_selection.metrics.values():
                    if not m.is_correct:
                        ok = False
            if sem.parameter:
                for cat in sem.parameter.values():
                    if cat and cat.metrics:
                        for m in cat.metrics.values():
                            if not m.is_correct:
                                ok = False

        # transformation checks (if any)
        trans: Optional[Dict[str, TransformResult]] = sem.transform
        if trans:
            for tr in trans.values():
                if not tr.correct:
                    ok = False

        # compute overall_avg_score from category averages
        cat_avgs: List[float] = []
        for cat in (sem.general, sem.function_selection):
            if cat and cat.avg_score is not None:
                cat_avgs.append(cat.avg_score)
        # for parameters, average the per‐param avg_scores
        if sem.parameter:
            param_avgs = [
                cat.avg_score
                for cat in sem.parameter.values()
                if cat.avg_score is not None
            ]
            if param_avgs:
                cat_avgs.append(sum(param_avgs) / len(param_avgs))

        self.overall_avg_score = sum(cat_avgs) / len(cat_avgs) if cat_avgs else None
        self.overall_valid = ok
        return self


# ----------------------------------------------------------------------
# 6. API Specification & Call Models
# ----------------------------------------------------------------------


# Map primitive spec-types to Python types (optional helper)
SPEC_TYPES: Dict[str, Any] = {
    "any": str,
    "array": list,
    "bigint": int,
    "boolean": bool,
    "byte": int,
    "char": str,
    "dict": dict,
    "double": float,
    "float": float,
    "hashtable": dict,
    "hashmap": dict,
    "integer": int,
    "int": int,
    "list": list,
    "long": int,
    "number": float,
    "null": NoneType,
    "object": dict,
    "string": str,
    "tuple": tuple,
    "uint": int,
    "ulong": int,
    "unsigned": int,
    "void": NoneType,
}


class FunctionDefinition(BaseModel):
    """
    Wraps an OpenAI-style function definition for function-calling clients.
    """

    name: str = Field(..., description="Function name as expected by the LLM.")
    description: Optional[str] = Field(
        None, description="Human-readable description of the function."
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description=(
            "JSON-Schema object describing all parameters; either a dict "
            "or a FunctionParameter model."
        ),
    )


class ToolSpec(BaseModel):
    """
    OpenAI tool specification wrapper, matching function-calling API.
    """

    type: Literal["function"] = Field(
        "function",
        description="Must be 'function' for OpenAI function-calling.",
    )
    function: FunctionDefinition = Field(
        ..., description="Underlying function definition or raw dict."
    )


class ToolFunctionCall(BaseModel):
    """
    Parsed representation of an LLM's function call response.
    """

    name: str = Field(..., description="Name of the function the LLM chose to call.")
    arguments: str = Field(
        ..., description="JSON-encoded string of the call's arguments."
    )
    parsed_arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed JSON arguments, available after validation.",
    )

    @model_validator(mode="after")
    def _parse_arguments(self) -> Self:
        """
        After model construction, parse the `arguments` JSON string
        into `parsed_arguments`, or raise a ValidationError.
        """
        try:
            raw = self.arguments
            self.parsed_arguments = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in arguments: {e}") from e
        return self


class ToolCall(BaseModel):
    """
    Full OpenAI function call object (for v1 function-calling API).
    """

    id: Optional[str] = Field(
        None,
        description=("Optional unique identifier for this function call."),
    )
    type: Literal["function"] = Field(
        "function",
        description="Must be 'function' for OpenAI function calls.",
    )
    function: ToolFunctionCall = Field(
        ..., description="Nested function name+arguments object or raw dict."
    )
