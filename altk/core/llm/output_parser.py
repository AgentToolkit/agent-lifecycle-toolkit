import re
import json
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Literal,
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
    schema: Dict[str, Any],
    model_name: str = "AutoModel",
    free_form_object_as_str: bool = False,
) -> Type[BaseModel]:
    """Build a Pydantic model from a JSON Schema dict.

    Args:
        schema: JSON Schema dict.
        model_name: name of the generated Pydantic model.
        free_form_object_as_str: when ``True``, any free-form ``type: object``
            property (one without its own ``properties`` sub-schema) is
            modeled as a JSON-formatted ``str`` instead of a ``dict``. This
            is the workaround for OpenAI's structured-output API, which
            requires ``additionalProperties: false`` on every object schema —
            a constraint that free-form dicts cannot meet. The caller is
            expected to use :func:`relax_freeform_object_schema` when
            validating the raw output so the JSON-string form is accepted.
            Default ``False`` preserves backward-compatible behavior.
    """
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

    def _map_object_for_prop(prop_schema: Dict[str, Any]) -> Type:
        """Return a model/dict/str for a property whose declared type is ``object``.

        A property is "free-form" if it has no ``properties`` sub-schema; the
        OpenAI workaround only applies to those. An object *with* properties is
        recursed into so nested constraints survive the conversion — a bare
        ``dict`` would erase them and let providers emit output that fails the
        original JSON Schema.
        """
        if "properties" in prop_schema:
            return json_schema_to_pydantic_model(
                prop_schema,
                model_name=f"{model_name}_{_next_nested_id()}",
                free_form_object_as_str=free_form_object_as_str,
            )
        if free_form_object_as_str:
            return str
        return dict

    _nested_count = [0]

    def _next_nested_id() -> int:
        _nested_count[0] += 1
        return _nested_count[0]

    def parse_type(
        type_def: Union[str, List[str], None],
        prop_schema: Dict[str, Any],
    ) -> Type[T]:
        def _lookup(t: str) -> Type:
            if t == "object":
                return _map_object_for_prop(prop_schema)
            if t == "array":
                return _map_array_for_prop(prop_schema)
            return type_mapping.get(t, Any)

        def _map_array_for_prop(prop_schema: Dict[str, Any]) -> Type:
            """Preserve ``items`` so array element constraints are not lost."""
            items = prop_schema.get("items")
            if not isinstance(items, dict):
                return list
            item_type = parse_type(items.get("type"), items)
            return List[item_type]  # type: ignore[valid-type]

        # ``enum`` becomes a real ``Literal`` type so the choices survive even
        # inside ``items``, where a Field-level constraint could not reach.
        enum_values = prop_schema.get("enum")
        if enum_values and all(
            isinstance(v, (str, int, bool)) or v is None for v in enum_values
        ):
            literal = Literal[tuple(enum_values)]  # type: ignore[valid-type]
            if isinstance(type_def, list) and "null" in type_def:
                return Optional[literal]  # type: ignore[return-value]
            return literal  # type: ignore[return-value]

        if isinstance(type_def, list):
            python_types = [_lookup(t) for t in type_def]
            if type(None) in python_types:
                python_types.remove(type(None))
                if len(python_types) == 1:
                    return Optional[python_types[0]]  # type: ignore
                else:
                    return Optional[Union[tuple(python_types)]]  # type: ignore
            else:
                return Union[tuple(python_types)]  # type: ignore
        if isinstance(type_def, str):
            return _lookup(type_def)
        return Any  # type: ignore[return-value]

    # JSON Schema keyword -> Pydantic ``Field`` argument. Carrying these over
    # keeps provider-native structured output faithful to the source schema
    # (a dropped ``minimum``/``enum`` shows up later as a validation failure).
    _CONSTRAINT_ARGS = {
        "minimum": "ge",
        "maximum": "le",
        "exclusiveMinimum": "gt",
        "exclusiveMaximum": "lt",
        "minLength": "min_length",
        "maxLength": "max_length",
        "minItems": "min_length",
        "maxItems": "max_length",
        "pattern": "pattern",
    }

    for prop_name, prop_schema in schema.get("properties", {}).items():
        field_type: Any = parse_type(prop_schema.get("type"), prop_schema)
        default = ... if prop_name in required_fields else None
        description = prop_schema.get("description", None)
        field_args: Dict[str, Any] = {"description": description} if description else {}
        for json_kw, field_kw in _CONSTRAINT_ARGS.items():
            if json_kw in prop_schema and field_kw not in field_args:
                field_args[field_kw] = prop_schema[json_kw]
        fields[prop_name] = (field_type, Field(default, **field_args))

    model = create_model(model_name, **fields)  # type: ignore
    # Mirror ``additionalProperties: false`` — providers with strict structured
    # output need it, and without it the model may invent extra keys that the
    # original schema then rejects.
    if schema.get("additionalProperties") is False:
        model.model_config["extra"] = "forbid"
    return model


def relax_freeform_object_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *schema* with free-form ``"type": "object"``
    properties widened to accept ``"string"`` as well.

    This is the validation-time counterpart to
    ``json_schema_to_pydantic_model(..., free_form_object_as_str=True)``: when
    the Pydantic model emits a JSON string for a free-form object field,
    ``jsonschema.validate`` against the original schema would reject it. This
    helper widens those fields so the same schema accepts both object-literal
    and stringified forms. Schemas where the object has sub-``properties`` are
    left alone.
    """
    import copy

    relaxed = copy.deepcopy(schema)
    for _prop, prop_schema in relaxed.get("properties", {}).items():
        t = prop_schema.get("type")
        if t == "object" and "properties" not in prop_schema:
            prop_schema["type"] = ["object", "string"]
    return relaxed


class OutputValidationError(Exception):
    """Raised when LLM output cannot be validated against the provided schema."""


def _is_truncated(raw: Any) -> bool:
    """Return ``True`` when *raw* was cut off by the token limit.

    A ``finish_reason`` of ``"length"`` means the model never finished writing,
    so whatever came back cannot be valid JSON. Reasoning ("thinking") models
    hit this routinely: the reasoning tokens consume the whole budget and the
    content field arrives empty.
    """
    choices = getattr(raw, "choices", None) or (
        raw.get("choices", []) if isinstance(raw, dict) else []
    )
    if not choices:
        return False
    first = choices[0]
    finish = getattr(first, "finish_reason", None) or (
        first.get("finish_reason") if isinstance(first, dict) else None
    )
    return finish == "length"


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

    Production knobs (instance-level, with class-level defaults):
      - ``free_form_object_as_str``: when ``True``, free-form ``type: object``
        schema fields are modeled in Pydantic as ``str`` (and the validation
        schema is widened at runtime to accept both object and string). Use
        this for providers that require ``additionalProperties: false`` on
        every object schema (notably OpenAI's structured-output API).
      - ``prompt_based_validation``: when ``True``, the schema is always
        injected into the system prompt and no native ``response_format``
        kwarg is forwarded. Use for providers that don't support OpenAI-style
        structured output (e.g. watsonx).
      - ``default_generation_kwargs``: dict of kwargs merged into every
        ``generate``/``generate_async`` call (e.g. ``{"max_tokens": 8096,
        "temperature": 0}``). Caller-provided kwargs override the defaults.
    """

    # Class-level defaults — override on subclasses or per instance in
    # ``configure_validation`` / constructor kwargs.
    free_form_object_as_str: bool = False
    prompt_based_validation: bool = False

    def __init__(
        self,
        *,
        free_form_object_as_str: Optional[bool] = None,
        prompt_based_validation: Optional[bool] = None,
        default_generation_kwargs: Optional[Dict[str, Any]] = None,
        **base_kwargs: Any,
    ) -> None:
        if free_form_object_as_str is not None:
            self.free_form_object_as_str = free_form_object_as_str
        if prompt_based_validation is not None:
            self.prompt_based_validation = prompt_based_validation
        self.default_generation_kwargs: Dict[str, Any] = dict(
            default_generation_kwargs or {}
        )
        # Set by the wrapped parser: was the most recent reply cut off by the
        # token limit? Retries use it to grow ``max_tokens`` instead of
        # re-asking with a budget already known to be too small.
        self._last_response_truncated: bool = False
        super().__init__(**base_kwargs)
        # Wrap the subclass's _parse_llm_response so empty / malformed LLM
        # outputs retry gracefully (the retry loop treats "" as invalid)
        # rather than raising an unrecoverable ValueError.
        # This particularly covers reasoning models that exhaust max_tokens
        # on "thinking" tokens and return finish_reason="length" with no
        # content but non-empty reasoning_content.
        orig_parse = self._parse_llm_response
        self._parse_llm_response = self._build_safe_parse(orig_parse)  # type: ignore[assignment]

    def configure_validation(
        self,
        *,
        free_form_object_as_str: Optional[bool] = None,
        prompt_based_validation: Optional[bool] = None,
        default_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "ValidatingLLMClient":
        """Update the validation knobs after construction (chainable)."""
        if free_form_object_as_str is not None:
            self.free_form_object_as_str = free_form_object_as_str
        if prompt_based_validation is not None:
            self.prompt_based_validation = prompt_based_validation
        if default_generation_kwargs is not None:
            self.default_generation_kwargs = dict(default_generation_kwargs)
        return self

    #: Budget assumed to be in play when the provider applied its own default
    #: (``max_tokens`` was never passed) and the reply came back truncated.
    ASSUMED_PROVIDER_MAX_TOKENS: int = 1024
    #: Ceiling for the retry escalation, so a stuck model cannot grow forever.
    MAX_TOKENS_ESCALATION_LIMIT: int = 16384

    def _escalate_max_tokens(self, current: Optional[int]) -> int:
        """Return a larger ``max_tokens`` for the next attempt after truncation."""
        base = current or self.ASSUMED_PROVIDER_MAX_TOKENS
        return min(base * 4, self.MAX_TOKENS_ESCALATION_LIMIT)

    def _build_safe_parse(self, orig):  # noqa: ANN001, ANN205
        """Wrap ``_parse_llm_response`` so parse failures become retry-worthy
        empty strings instead of raising. Also surfaces a targeted warning
        when a reasoning-only response exhausted the token budget."""
        import logging as _logging

        _logger = _logging.getLogger("altk.core.llm.output_parser")

        def _safe_parse(raw):  # noqa: ANN001, ANN202
            self._last_response_truncated = _is_truncated(raw)
            try:
                return orig(raw)
            except (ValueError, KeyError):
                # Detect: choice with reasoning_content but finish_reason='length'
                _choices = getattr(raw, "choices", None) or (
                    raw.get("choices", []) if isinstance(raw, dict) else []
                )
                if _choices:
                    c0 = _choices[0]
                    _msg = getattr(c0, "message", None) or (
                        c0.get("message", {}) if isinstance(c0, dict) else {}
                    )
                    _reasoning = getattr(_msg, "reasoning_content", None) or (
                        _msg.get("reasoning_content")
                        if isinstance(_msg, dict)
                        else None
                    )
                    _finish = getattr(c0, "finish_reason", None) or (
                        c0.get("finish_reason") if isinstance(c0, dict) else None
                    )
                    if _reasoning and _finish == "length":
                        _logger.warning(
                            "LLM reasoning consumed the entire token budget "
                            "(finish_reason='length'). Consider increasing "
                            "max_tokens. Will retry."
                        )
                        return ""
                _logger.debug("LLM returned empty/unparseable response; will retry.")
                return ""

        return _safe_parse

    @classmethod
    @abstractmethod
    def provider_class(cls) -> Type[Any]:
        """Return the underlying SDK client class, e.g. openai.OpenAI."""

    def supports_native_structured_output(self) -> bool:
        """Whether the target model honors a native structured-output kwarg.

        Defaults to ``True`` (previous behavior). Providers that can tell which
        models support it override this; when it returns ``False`` the schema is
        injected into the system prompt instead, because a model that ignores
        ``response_format`` cannot be constrained by it.
        """
        return True

    def _render_native_schema(
        self, schema: Union[Dict[str, Any], Type[BaseModel], Type[Any]]
    ) -> Any:
        """Render *schema* into the value this provider expects for its native
        structured-output kwarg (``schema_field``).

        The default converts a JSON Schema dict into a Pydantic model, which is
        what litellm accepts. Providers whose SDK rejects a model class
        override this — see the OpenAI/Azure clients, which need a
        ``{"type": "json_schema", ...}`` dict for ``chat.completions.create``.
        """
        if isinstance(schema, dict):
            return json_schema_to_pydantic_model(
                schema,
                free_form_object_as_str=self.free_form_object_as_str,
            )
        return schema

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
            # Widen free-form object props to also accept strings when we're
            # configured to round-trip them as JSON strings (see
            # ``free_form_object_as_str`` in the class docstring).
            effective_schema = (
                relax_freeform_object_schema(schema)
                if self.free_form_object_as_str
                else schema
            )
            try:
                jsonschema.validate(instance=data, schema=effective_schema)
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
        # Instance defaults — caller kwargs win.
        if self.default_generation_kwargs:
            merged = {**self.default_generation_kwargs}
            merged.update(kwargs)
            kwargs = merged
        # Providers that don't support native structured output switch to
        # prompt-based schema injection and drop any OpenAI-style
        # ``response_format`` field.
        if self.prompt_based_validation:
            include_schema_in_system_prompt = True
            schema_field = None
        # Models that ignore a native schema kwarg must be steered by the
        # prompt instead; sending ``response_format`` to them is at best a
        # no-op and at worst returns empty content.
        elif schema_field and not self.supports_native_structured_output():
            include_schema_in_system_prompt = True
            schema_field = None
        current = prompt
        instr = None
        if include_schema_in_system_prompt:
            instr = self._make_instruction(schema)
            current = self._inject_system(prompt, instr)
        if schema_field:
            kwargs[schema_field] = self._render_native_schema(schema)

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
            raw = ""
            try:
                # Inside the try: ``_parse_llm_response`` raises ValueError for a
                # contentless reply, and that must be retried like any other
                # invalid output rather than aborting the whole call.
                raw = super()._generate(**{"prompt": current, **filtered_kwargs})
                if isinstance(raw, str):
                    return self._validate(raw, schema)
                return raw
            except (OutputValidationError, ValueError) as e:
                # ValueError covers providers whose ``_parse_llm_response``
                # rejects an empty/contentless response ("No content or tool
                # calls found in response"). Without it, a single blank reply
                # from the backend aborts the whole call and the configured
                # ``retries`` are never used.
                last_error = str(e)
                # An empty response carries no mistake to correct: it is a
                # transient backend failure (empty content under load). Retry
                # the original prompt untouched — appending correction turns
                # would grow a conversation that several backends answer with
                # another empty response, burning every remaining attempt.
                if not (isinstance(raw, str) and raw.strip()):
                    # Truncated by the token limit? Re-asking with the same
                    # budget yields the identical truncation, so grow it. This
                    # is the common failure for reasoning models, whose
                    # "thinking" tokens can consume a small default budget
                    # (watsonx defaults to 1024) before any content is emitted.
                    if self._last_response_truncated:
                        kwargs["max_tokens"] = self._escalate_max_tokens(
                            kwargs.get("max_tokens")
                        )
                    current = self._inject_system(prompt, instr) if instr else prompt
                    continue
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
        # Instance defaults — caller kwargs win.
        if self.default_generation_kwargs:
            merged = {**self.default_generation_kwargs}
            merged.update(kwargs)
            kwargs = merged
        # Providers that don't support native structured output switch to
        # prompt-based schema injection and drop any OpenAI-style
        # ``response_format`` field.
        if self.prompt_based_validation:
            include_schema_in_system_prompt = True
            schema_field = None
        # Models that ignore a native schema kwarg must be steered by the
        # prompt instead; sending ``response_format`` to them is at best a
        # no-op and at worst returns empty content.
        elif schema_field and not self.supports_native_structured_output():
            include_schema_in_system_prompt = True
            schema_field = None
        current = prompt
        instr = None
        if include_schema_in_system_prompt:
            instr = self._make_instruction(schema)
            current = self._inject_system(prompt, instr)
        if schema_field:
            kwargs[schema_field] = self._render_native_schema(schema)

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
            raw = ""
            try:
                # Inside the try: ``_parse_llm_response`` raises ValueError for a
                # contentless reply, and that must be retried like any other
                # invalid output rather than aborting the whole call.
                raw = await super()._generate_async(
                    **{"prompt": current, **filtered_kwargs}
                )
                if isinstance(raw, str):
                    return self._validate(raw, schema)
                return raw
            except (OutputValidationError, ValueError) as e:
                # ValueError covers providers whose ``_parse_llm_response``
                # rejects an empty/contentless response ("No content or tool
                # calls found in response"). Without it, a single blank reply
                # from the backend aborts the whole call and the configured
                # ``retries`` are never used.
                last_error = str(e)
                # An empty response carries no mistake to correct: it is a
                # transient backend failure (empty content under load). Retry
                # the original prompt untouched — appending correction turns
                # would grow a conversation that several backends answer with
                # another empty response, burning every remaining attempt.
                if not (isinstance(raw, str) and raw.strip()):
                    # Truncated by the token limit? Re-asking with the same
                    # budget yields the identical truncation, so grow it. This
                    # is the common failure for reasoning models, whose
                    # "thinking" tokens can consume a small default budget
                    # (watsonx defaults to 1024) before any content is emitted.
                    if self._last_response_truncated:
                        kwargs["max_tokens"] = self._escalate_max_tokens(
                            kwargs.get("max_tokens")
                        )
                    current = self._inject_system(prompt, instr) if instr else prompt
                    continue
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
