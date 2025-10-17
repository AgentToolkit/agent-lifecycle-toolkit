import re
import json
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import jsonschema

from pydantic import (
    BaseModel,
    create_model,
    Field,
    ValidationError as PydanticValidationError,
)

from .base import BaseLLMClient

T = TypeVar("T")


def json_schema_to_pydantic_model(
    schema: Dict[str, Any], model_name: str = "AutoModel"
) -> Type[BaseModel]:
    fields = {}
    required_fields = set(schema.get("required", []))

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    def parse_type(type_def: Union[str, List[str]]) -> Type[T]:
        if isinstance(type_def, list):
            python_types = [type_mapping.get(t, Any) for t in type_def]
            if type(None) in python_types:
                python_types.remove(type(None))
                if len(python_types) == 1:
                    return Optional[python_types[0]]  # type: ignore
                else:
                    return Optional[Union[tuple(python_types)]]  # type: ignore
            else:
                return Union[tuple(python_types)]  # type: ignore
        else:
            return type_mapping.get(type_def, Any)

    for prop_name, prop_schema in schema.get("properties", {}).items():
        field_type: Any = parse_type(prop_schema.get("type"))
        default = ... if prop_name in required_fields else None
        description = prop_schema.get("description", None)
        field_args = {"description": description} if description else {}
        fields[prop_name] = (field_type, Field(default, **field_args))

    return create_model(model_name, **fields)  # type: ignore


class OutputValidationError(Exception):
    """Raised when LLM output cannot be validated against the provided schema."""


class ValidatingLLMClient(BaseLLMClient, ABC):
    """
    An LLMClient wrapper enforcing output structure via:
      - JSON Schema (dict),
      - Pydantic model (BaseModel subclass),
      - or Python built-in types (int, float, str, bool, list, dict).

    Features:
      - Injects a system-level prompt describing the required format.
      - Cleans raw responses (strips Markdown, extracts JSON).
      - Validates and parses the response.
      - Retries only invalid items (single or batch) up to `retries` times.
      - Falls back to single-item loops if no batch method is configured.
    """

    @classmethod
    @abstractmethod
    def provider_class(cls) -> Type[Any]:
        """Return the underlying SDK client class, e.g. openai.OpenAI."""

    @abstractmethod
    def _register_methods(self) -> None:
        """
        Register MethodConfig entries:
          self.set_method_config("text", ...),
          self.set_method_config("chat", ...),
          self.set_method_config("text_async", ...),
          self.set_method_config("chat_async", ...),
        """

    def _make_instruction(
        self, schema: Union[Dict[str, Any], Type[BaseModel], Type[Any]]
    ) -> str:
        """Produce a clear instruction describing exactly the required output format."""
        if isinstance(schema, dict):
            schema_json = json.dumps(schema, indent=2)
            return (
                "Please output ONLY a JSON object conforming exactly to the following JSON Schema:\n"
                f"{schema_json}"
            )
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            model_schema = schema.model_json_schema()
            return (
                "Please output ONLY a JSON object conforming exactly to this Pydantic model schema:\n"
                f"{model_schema}"
            )
        if isinstance(schema, type) and schema in (int, float, str, bool, list, dict):
            # For simple types, no JSON wrapper required
            return f"Please output ONLY a value of type `{schema.__name__}`."
        raise TypeError(f"Unsupported schema type: {schema!r}")

    @staticmethod
    def _extract_json(raw: str) -> str:
        """
        Extract JSON from markdown fences or inline braces.
        Falls back to returning the entire raw string.
        """
        # Code fence (```json ... ```)
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if fence:
            return fence.group(1)
        # Inline {...}
        inline = re.search(r"(\{[\s\S]*\})", raw)
        if inline:
            return inline.group(1)
        return raw

    def _clean_raw(self, raw: str) -> str:
        """Strip extraneous markdown and whitespace."""
        cleaned = self._extract_json(raw)
        return cleaned.strip()

    def _validate(
        self, raw: str, schema: Union[Dict[str, Any], Type[BaseModel], Type[Any]]
    ) -> Any:
        """
        Clean, parse, and validate raw text against the schema/type.
        Returns the parsed object or Pydantic instance.
        Raises OutputValidationError on any failure.
        """

        cleaned = self._clean_raw(raw)
        try:
            if isinstance(schema, str):
                data = cleaned
            else:
                data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                data = json.loads(cleaned.encode("unicode_escape").decode("utf-8"))
            except Exception:
                data = cleaned

        # JSON Schema validation
        if isinstance(schema, dict):
            if jsonschema is None:
                raise ImportError(
                    "jsonschema is required for JSON Schema validation. Install with: pip install jsonschema"
                )
            try:
                jsonschema.validate(instance=data, schema=schema)
            except jsonschema.ValidationError as e:
                raise OutputValidationError(
                    f"JSON Schema validation error: {e.message}"
                ) from e
            return data

        # Pydantic model validation
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                return schema.model_validate(data)
            except PydanticValidationError as e:
                raise OutputValidationError(f"Pydantic validation error: {e}") from e

        # Built-in type enforcement
        if isinstance(schema, type) and schema in (int, float, str, bool, list, dict):
            if not isinstance(data, schema):
                raise OutputValidationError(
                    f"Type mismatch: expected {schema.__name__}, got {type(data).__name__}"
                )
            return data

        raise TypeError(f"Unsupported schema type: {schema!r}")

    def _inject_system(
        self, prompt: Union[str, List[Dict[str, Any]]], instr: str
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Combine instruction and user prompt:
        - For text: prepend the instruction.
        - For chat messages: if first role=system, append instr to it;
          otherwise insert a new system message.
        """
        if isinstance(prompt, str):
            return f"{instr}\n\n{prompt}"

        msgs = prompt.copy()
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + instr
        else:
            msgs.insert(0, {"role": "system", "content": instr})
        return msgs

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        schema: Union[Dict[str, Any], Type[BaseModel], Type[Any]],
        schema_field: Optional[str] = None,
        retries: int = 3,
        include_schema_in_system_prompt: bool = False,
        **kwargs: Any,
    ) -> Union[str, Any]:
        """
        Synchronous single-item generation with validation + retries.
        """
        current = prompt
        instr = None
        if include_schema_in_system_prompt:
            instr = self._make_instruction(schema)
            current = self._inject_system(prompt, instr)
        if schema_field:
            kwargs[schema_field] = schema
            if isinstance(schema, dict):
                new_schema = json_schema_to_pydantic_model(schema)
                kwargs[schema_field] = new_schema

        last_error: Optional[str] = None
        for _ in range(1, retries + 1):
            # Filter out schema-related kwargs for the base class
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "schema",
                    "schema_field",
                    "retries",
                    "include_schema_in_system_prompt",
                ]
            }
            raw = super()._generate(**{"prompt": current, **filtered_kwargs})
            try:
                if isinstance(raw, str):
                    return self._validate(raw, schema)
                return raw
            except OutputValidationError as e:
                last_error = str(e)
                correction = (
                    f"The previous response did not conform: {last_error}\nPlease correct it."
                    " And remember to output ONLY the requested schema, without any additional text."
                )
                if isinstance(current, str):
                    if instr:
                        current = (
                            f"{instr}\n\nPrevious output:\n{raw}\n\n"
                            f"{correction}\n\n{prompt}"
                        )
                    else:
                        current = f"Previous output:\n{raw}\n\n{correction}\n\n{prompt}"
                else:
                    current = current + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": correction},
                    ]
        raise OutputValidationError(f"Failed after {retries} attempts: {last_error}")

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        schema: Union[Dict[str, Any], Type[BaseModel], Type[Any]],
        schema_field: Optional[str] = None,
        retries: int = 3,
        include_schema_in_system_prompt: bool = False,
        **kwargs: Any,
    ) -> Union[str, Any]:
        """
        Asynchronous single-item generation with validation + retries.
        """
        current = prompt
        instr = None
        if include_schema_in_system_prompt:
            instr = self._make_instruction(schema)
            current = self._inject_system(prompt, instr)
        if schema_field:
            kwargs[schema_field] = schema
            if isinstance(schema, dict):
                new_schema = json_schema_to_pydantic_model(schema)
                kwargs[schema_field] = new_schema

        last_error: Optional[str] = None
        for _ in range(1, retries + 1):
            # Filter out schema-related kwargs for the base class
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "schema",
                    "schema_field",
                    "retries",
                    "include_schema_in_system_prompt",
                ]
            }
            raw = await super()._generate_async(
                **{"prompt": current, **filtered_kwargs}
            )
            try:
                if isinstance(raw, str):
                    return self._validate(raw, schema)
                return raw
            except OutputValidationError as e:
                last_error = str(e)
                correction = (
                    f"The previous response did not conform: {last_error}\nPlease correct it."
                    " And remember to output ONLY the requested schema, without any additional text."
                )
                if isinstance(current, str):
                    if instr:
                        current = (
                            f"{instr}\n\nPrevious output:\n{raw}\n\n"
                            f"{correction}\n\n{prompt}"
                        )
                    else:
                        current = f"Previous output:\n{raw}\n\n{correction}\n\n{prompt}"
                else:
                    current = current + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": correction},
                    ]
        raise OutputValidationError(f"Failed after {retries} attempts: {last_error}")
