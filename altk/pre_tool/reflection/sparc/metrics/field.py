from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field as PydanticField, model_validator

JSONType = Literal["integer", "number", "string", "boolean", "object", "array"]
TField = TypeVar("TField", bound="BaseField")
BaseFieldRegistry: List[Type[BaseField]] = []


class BaseField(BaseModel, ABC):
    """
    Abstract representation of a single metric field.

    Attributes:
        name: Identifier of the field (used as JSON key).
        json_type: JSON Schema type of the field.
        description: Human-friendly description of the field's purpose.
        jsonschema_extra: Additional JSONSchema keywords (e.g., enum, pattern).
        extra_params: Non-JSONSchema attributes (e.g., thresholds).
    """

    name: str
    json_type: JSONType
    description: str = PydanticField(
        "No description provided. Please specify what this field represents.",
        description="A clear description of this field's meaning.",
    )
    jsonschema_extra: Dict[str, Any] = PydanticField(
        default_factory=dict,
        description="Additional JSONSchema constraints for this field.",
    )
    extra_params: Dict[str, Any] = PydanticField(
        default_factory=dict,
        description="Extra parameters not included in the JSONSchema (e.g., thresholds).",
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstract__", False):
            BaseFieldRegistry.insert(0, cls)

    @classmethod
    def can_handle(cls, name: str, schema: Dict[str, Any]) -> bool:
        """Override in subclasses to signal compatibility with a JSONSchema snippet."""
        return False

    @classmethod
    def from_jsonschema(cls, name: str, schema: Dict[str, Any]) -> BaseField:
        """
        Instantiate the appropriate Field subclass from a JSONSchema property.
        The first subclass whose `can_handle` returns True is used.
        Falls back to GenericField.
        """
        for field_cls in BaseFieldRegistry:
            if field_cls.can_handle(name, schema):
                desc = schema.get("description", "")
                extra = {
                    k: v for k, v in schema.items() if k not in ("type", "description")
                }
                return field_cls(
                    name=name,
                    json_type=schema.get("type", "string"),
                    description=desc,
                    jsonschema_extra=extra,
                    extra_params={},
                )
        return GenericField(
            name=name,
            json_type=schema.get("type", "string"),
            description=schema.get("description", ""),
            jsonschema_extra={
                k: v for k, v in schema.items() if k not in ("type", "description")
            },
            extra_params={},
        )

    def to_jsonschema(self) -> Dict[str, Any]:
        return {
            "type": self.json_type,
            "description": self.description,
            **self.jsonschema_extra,
        }

    # --- Getters and Setters ---

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> None:
        self.name = name

    def get_description(self) -> str:
        return self.description

    def set_description(self, description: str) -> None:
        self.description = description

    def get_jsonschema_extra(self) -> Dict[str, Any]:
        return dict(self.jsonschema_extra)

    def set_jsonschema_extra(self, extra: Dict[str, Any]) -> None:
        self.jsonschema_extra = extra

    def get_extra_param(self, key: str) -> Any:
        return self.extra_params.get(key)

    def set_extra_param(self, key: str, value: Any) -> None:
        self.extra_params[key] = value


class NumericField(BaseField):
    """
    Numeric field (integer or number) with optional thresholds.
    The `extra_params` dict may include:
      - threshold_low: minimal acceptable value (for validation)
      - threshold_high: maximal acceptable value
    """

    threshold_low: Optional[float] = PydanticField(
        None, description="Lower bound for correctness checks (not in JSONSchema)."
    )
    threshold_high: Optional[float] = PydanticField(
        None, description="Upper bound for correctness checks (not in JSONSchema)."
    )

    __abstract__ = False

    @model_validator(mode="before")
    def extract_thresholds(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        extra = values.get("jsonschema_extra", {})
        if "threshold_low" in extra:
            values["threshold_low"] = extra["threshold_low"]
        if "threshold_high" in extra:
            values["threshold_high"] = extra["threshold_high"]
        return values

    @classmethod
    def can_handle(cls, name: str, schema: Dict[str, Any]) -> bool:
        return schema.get("type") in ("integer", "number")

    @classmethod
    def from_jsonschema(cls, name: str, schema: Dict[str, Any]) -> NumericField:
        """
        Create a NumericField from a JSONSchema property.
        """
        return NumericField(
            name=name,
            json_type=schema.get("type", "number"),
            description=schema.get("description", ""),
            jsonschema_extra={
                k: v for k, v in schema.items() if k not in ("type", "description")
            },
            extra_params={},
        )

    def to_jsonschema(self) -> Dict[str, Any]:
        return super().to_jsonschema()

    def is_within_threshold(self, value: float) -> bool:
        if self.threshold_low is not None and value < self.threshold_low:
            return False
        if self.threshold_high is not None and value > self.threshold_high:
            return False
        return True


class EnumField(BaseField):
    """
    Field whose value must be one of a fixed set of options.
    Expects `jsonschema_extra["enum"]` to be a list of allowed values.
    """

    __abstract__ = False

    @classmethod
    def can_handle(cls, name: str, schema: Dict[str, Any]) -> bool:
        return "enum" in schema


class ExplanationField(BaseField):
    """
    Free-form explanation of the metric's reasoning.
    """

    __abstract__ = False

    def __init__(self, **data: Any):
        data.setdefault(
            "description",
            "A detailed, step-by-step explanation of the reasoning behind the metric's value.",
        )
        super().__init__(**data)

    @classmethod
    def can_handle(cls, name: str, schema: Dict[str, Any]) -> bool:
        return name.lower() == "explanation" and schema.get("type") == "string"


class EvidenceField(BaseField):
    """
    The specific quote or reference that supports the metric's evaluation.
    """

    __abstract__ = False

    def __init__(self, **data: Any):
        data.setdefault(
            "description",
            "The exact quote or reference from the input or context that justifies the metric's value.",
        )
        super().__init__(**data)

    @classmethod
    def can_handle(cls, name: str, schema: Dict[str, Any]) -> bool:
        return name.lower() == "evidence" and schema.get("type") == "string"


class CorrectionField(BaseField):
    """
    A structured suggestion (as JSON) for correcting or improving the output.
    """

    __abstract__ = False

    def __init__(self, **data: Any):
        data.setdefault(
            "description",
            "A JSON-formatted suggestion for how to correct or improve the output if needed.",
        )
        super().__init__(**data)

    @classmethod
    def can_handle(cls, name: str, schema: Dict[str, Any]) -> bool:
        return name.lower() == "correction" and schema.get("type") == "object"


class GenericField(BaseField):
    """
    Fallback field type for any property not handled by other classes.
    """

    __abstract__ = False

    def __init__(self, **data: Any):
        data.setdefault(
            "description",
            f"A generic field named '{data.get('name')}' of type {data.get('json_type')}.",
        )
        super().__init__(**data)

    @classmethod
    def can_handle(cls, name: str, schema: Dict[str, Any]) -> bool:
        return True
