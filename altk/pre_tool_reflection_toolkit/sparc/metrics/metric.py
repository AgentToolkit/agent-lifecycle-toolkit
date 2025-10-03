from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

from altk.pre_tool_reflection_toolkit.sparc.metrics.field import (
    BaseField,
    ExplanationField,
    EvidenceField,
    CorrectionField,
    NumericField,
)

TMetric = TypeVar("TMetric", bound="Metric")


class Metric:
    """
    Abstract representation of an evaluation metric composed of multiple fields.
    """

    def __init__(
        self,
        name: str,
        description: str,
        fields: Optional[List[BaseField]] = None,
        required: Optional[List[str]] = None,
        additional_properties: bool = True,
    ) -> None:
        """
        Args:
            name: Unique metric identifier.
            description: Full description of what this metric measures.
            fields: List of BaseField instances composing this metric.
            required: List of field names that must appear in results.
                      Defaults to all provided fields.
        """
        self.name = name
        self.description = description
        self.fields: List[BaseField] = fields or []
        self.additional_properties = additional_properties
        # Determine required fields
        if required is not None:
            self.required_fields: Set[str] = set(required)
        else:
            self.required_fields: Set[str] = {f.name for f in self.fields}

        # Validate required_fields
        known = {f.name for f in self.fields}
        missing = self.required_fields - known
        if missing:
            raise ValueError(
                f"Required fields {missing} not among metric fields {known}"
            )

    def to_jsonschema(self) -> Dict[str, Any]:
        """
        Build a JSONSchema representation of this metric.

        Returns:
            A dict with keys:
              - title: self.name
              - description: self.description
              - type: "object"
              - properties: mapping field.name → field.to_jsonschema()
              - required: list of required field names
        """
        props: Dict[str, Any] = {f.name: f.to_jsonschema() for f in self.fields}
        return {
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": props,
            "required": sorted(self.required_fields),
            "additionalProperties": self.additional_properties,
        }

    def add_field(self, field: BaseField, required: bool = True) -> None:
        """
        Add a new field to this metric.

        Args:
            field: BaseField instance.
            required: Whether this field must appear in results.
        """
        if any(f.name == field.name for f in self.fields):
            raise ValueError(f"Field '{field.name}' already defined")
        self.fields.append(field)
        if required:
            self.required_fields.add(field.name)

    def remove_field(self, name: str) -> None:
        """
        Remove a field by name.

        Args:
            name: Name of field to remove.
        """
        self.fields = [f for f in self.fields if f.name != name]
        self.required_fields.discard(name)

    @classmethod
    def from_jsonschema(cls: Type[TMetric], schema: Dict[str, Any]) -> Metric:
        """
        Reconstruct a Metric from a JSONSchema dict.

        Args:
            schema: dict with 'title', 'description', 'properties', 'required'.

        Returns:
            Metric instance with fields populated.
        """
        name: str = schema.get("title", "")
        description: str = schema.get("description", "")
        props: Dict[str, Any] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])
        additional_props: bool = schema.get("additionalProperties", True)
        fields: List[BaseField] = []
        for fname, fschema in props.items():
            # If type is number or integer, use NumericField
            if fschema.get("type") in ("number", "integer"):
                field = NumericField.from_jsonschema(fname, fschema)
            else:
                field = BaseField.from_jsonschema(fname, fschema)
            fields.append(field)
        return cls(
            name=name,
            description=description,
            fields=fields,
            required=required,
            additional_properties=additional_props,
        )

    def is_important(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        A result is 'important' if its confidence lies within the defined confidence thresholds.

        Args:
            result: Parsed metric result with at least 'confidence'.

        Returns:
            (important: bool, reason: Optional[str])
        """
        try:
            conf = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            return False, "Invalid confidence value"
        # locate the confidence field
        conf_field = next((f for f in self.fields if f.name == "confidence"), None)
        if isinstance(conf_field, NumericField):
            ok = conf_field.is_within_threshold(conf)
            reason = (
                None
                if ok
                else f"Confidence {conf} outside [{conf_field.threshold_low},{conf_field.threshold_high}]"
            )
            return ok, reason
        return False, "Confidence field not defined"

    def is_correct(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        A result is 'correct' if it is important AND its output lies within thresholds.

        Args:
            result: Parsed metric result with 'output' and 'confidence'.

        Returns:
            (correct: bool, reason: Optional[str])
        """
        important, imp_reason = self.is_important(result)
        if not important:
            return True, f"Not important: {imp_reason}"
        # check output
        try:
            val = float(result.get("output", 0.0))
        except (TypeError, ValueError):
            return False, "Invalid output value"
        out_field = next((f for f in self.fields if f.name == "output"), None)
        if isinstance(out_field, NumericField):
            ok = out_field.is_within_threshold(val)
            reason = (
                None
                if ok
                else f"Output {val} outside [{out_field.threshold_low},{out_field.threshold_high}]"
            )
            return ok, reason
        return False, "Output field not defined"

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse a raw response string into a structured dict.

        Args:
            response: Raw response string.

        Returns:
            Parsed response as a dict.
        """
        # Default implementation: assume JSON string
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse response: {e}") from e


class StandardMetric(Metric):
    """
    A standard metric with common fields:
      - explanation: string, detailed reasoning.
      - evidence: string, supporting quote or reference.
      - output: numeric value within specified range.
      - confidence: numeric confidence within specified range.
      - correction: object, structured suggestion for improvement.
    Also provides convenience methods `is_important` and `is_correct`.
    """

    def __init__(
        self,
        name: str,
        description: str,
        *,
        output_range: Tuple[float, float] = (0.0, 1.0),
        confidence_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """
        Args:
            name: Metric identifier.
            description: Explanation of what the metric measures.
            output_range: (min, max) allowed for the 'output' field.
            confidence_range: (min, max) for the 'confidence' field.

        Fields created:
          - explanation: "A detailed, step-by-step explanation of the reasoning."
          - evidence: "The exact quote or evidence supporting the reasoning."
          - output: numeric in output_range
          - confidence: numeric in confidence_range
          - correction: structured suggestion if output below threshold
        """
        # Prepare fields
        min_out, max_out = output_range
        min_conf, max_conf = confidence_range

        explanation = ExplanationField(
            name="explanation",
            json_type="string",
            description="A detailed, step-by-step explanation of the reasoning behind the output value.",
        )
        evidence = EvidenceField(
            name="evidence",
            json_type="string",
            description="The exact quote or reference that supports the output value.",
        )
        output = NumericField(
            name="output",
            json_type=(
                "number"
                if isinstance(min_out, float) or isinstance(max_out, float)
                else "integer"
            ),
            description=f"Primary numeric score for this metric (range {min_out} to {max_out}).",
            jsonschema_extra={"minimum": min_out, "maximum": max_out},
            extra_params={"threshold_low": min_out, "threshold_high": max_out},
        )
        confidence = NumericField(
            name="confidence",
            json_type="number",
            description=f"Confidence in the output value (range {min_conf} to {max_conf}).",
            jsonschema_extra={"minimum": min_conf, "maximum": max_conf},
            extra_params={"threshold_low": min_conf, "threshold_high": max_conf},
        )
        correction = CorrectionField(
            name="correction",
            json_type="object",
            description="Structured suggestion for how to correct or improve the output if needed.",
        )

        fields = [explanation, evidence, output, confidence, correction]
        super().__init__(name=name, description=description, fields=fields)

    def is_important(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        A result is 'important' if its confidence lies within the defined confidence thresholds.

        Args:
            result: Parsed metric result with at least 'confidence'.

        Returns:
            (important: bool, reason: Optional[str])
        """
        try:
            conf = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            return False, "Invalid confidence value"
        # locate the confidence field
        conf_field = next((f for f in self.fields if f.name == "confidence"), None)
        if isinstance(conf_field, NumericField):
            ok = conf_field.is_within_threshold(conf)
            reason = (
                None
                if ok
                else f"Confidence {conf} outside [{conf_field.threshold_low},{conf_field.threshold_high}]"
            )
            return ok, reason
        return False, "Confidence field not defined"

    def is_correct(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        A result is 'correct' if it is important AND its output lies within thresholds.

        Args:
            result: Parsed metric result with 'output' and 'confidence'.

        Returns:
            (correct: bool, reason: Optional[str])
        """
        important, imp_reason = self.is_important(result)
        if not important:
            return True, f"Not important: {imp_reason}"
        # check output
        try:
            val = float(result.get("output", 0.0))
        except (TypeError, ValueError):
            return False, "Invalid output value"
        out_field = next((f for f in self.fields if f.name == "output"), None)
        if isinstance(out_field, NumericField):
            ok = out_field.is_within_threshold(val)
            reason = (
                None
                if ok
                else f"Output {val} outside [{out_field.threshold_low},{out_field.threshold_high}]"
            )
            return ok, reason
        return False, "Output field not defined"
