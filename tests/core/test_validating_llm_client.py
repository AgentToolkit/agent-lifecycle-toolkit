"""Tests for ``ValidatingLLMClient`` production knobs.

Covers the behaviors migrated from CLEAR's monkey-patches:

- ``free_form_object_as_str`` changes how free-form ``type: object`` fields are
  modeled (as ``str`` in Pydantic) and loosens ``_validate`` to accept the
  stringified form. Together these make SPARC/CLEAR compatible with OpenAI's
  ``additionalProperties: false`` structured-output requirement.
- ``prompt_based_validation`` forces schema-into-system-prompt and skips
  native ``response_format``. Targeted at providers like watsonx.
- ``default_generation_kwargs`` forwards e.g. ``max_tokens``/``temperature``
  into every ``generate`` call, with caller kwargs winning.
- The wrapped ``_parse_llm_response`` returns ``""`` on parse error and
  warns when a reasoning-only response exhausted the budget.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Type

import pytest
from pydantic import BaseModel

from altk.core.llm.output_parser import (
    OutputValidationError,
    ValidatingLLMClient,
    json_schema_to_pydantic_model,
    relax_freeform_object_schema,
)

# ---------------------------------------------------------------------------
# Dummy client used throughout — no actual LLM call.
# ---------------------------------------------------------------------------


class _FakeValidating(ValidatingLLMClient):
    """Minimal concrete subclass — real ``_generate`` is stubbed via
    ``monkeypatch`` in each test, because ``super()._generate`` inside
    ``ValidatingLLMClient.generate`` bypasses subclass overrides."""

    @classmethod
    def provider_class(cls) -> Type[Any]:
        return object

    def _register_methods(self) -> None:  # noqa: D401
        pass

    def _setup_parameter_mapper(self) -> None:  # noqa: D401
        pass

    def _parse_llm_response(self, raw: Any) -> str:
        if isinstance(raw, Exception):
            raise raw
        return str(raw)

    def __init__(self, **kw):
        super().__init__(**kw)


def _install_scripted_generate(monkeypatch, observed: list, scripted: list):
    """Replace ``BaseLLMClient._generate`` so tests can intercept the call
    that happens inside ``ValidatingLLMClient.generate``'s retry loop."""
    from altk.core.llm.base import BaseLLMClient

    def fake_generate(self, **kwargs):  # noqa: ANN001
        observed.append(kwargs)
        if not scripted:
            return ""
        raw = scripted.pop(0)
        return self._parse_llm_response(raw)

    monkeypatch.setattr(BaseLLMClient, "_generate", fake_generate, raising=True)


# ---------------------------------------------------------------------------
# json_schema_to_pydantic_model — free_form_object_as_str
# ---------------------------------------------------------------------------


class TestJsonSchemaToPydantic:
    def test_default_keeps_object_as_dict(self):
        m = json_schema_to_pydantic_model(
            {"type": "object", "properties": {"a": {"type": "object"}}}
        )
        assert m.model_fields["a"].annotation is dict

    def test_freeform_object_flag_switches_to_str(self):
        m = json_schema_to_pydantic_model(
            {"type": "object", "properties": {"a": {"type": "object"}}},
            free_form_object_as_str=True,
        )
        assert m.model_fields["a"].annotation is str

    def test_freeform_flag_recurses_into_nested_objects(self):
        # Only a free-form object (no properties) becomes a str. An object that
        # *has* properties is recursed into as a nested model, so its fields
        # survive into the schema handed to the provider — a bare ``dict`` would
        # erase them and let the model emit output the real schema rejects.
        m = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "flat": {"type": "object"},
                    "structured": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            },
            free_form_object_as_str=True,
        )
        assert m.model_fields["flat"].annotation is str
        nested = m.model_fields["structured"].annotation
        assert issubclass(nested, BaseModel)
        assert set(nested.model_fields) == {"x"}

    def test_nested_models_forbid_extras_without_repeating_the_keyword(self):
        # Strict providers require ``additionalProperties: false`` on *every*
        # object schema; nested sub-schemas rarely repeat the keyword, so
        # forbid is the default. See issue #120.
        rendered = json_schema_to_pydantic_model(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sub": {"type": "object", "properties": {"x": {"type": "string"}}}
                },
            }
        ).model_json_schema()
        assert rendered["additionalProperties"] is False
        assert all(
            d["additionalProperties"] is False for d in rendered["$defs"].values()
        )

    def test_explicit_additional_properties_true_is_respected(self):
        rendered = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "sub": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {"x": {"type": "string"}},
                    }
                },
            }
        ).model_json_schema()
        assert "additionalProperties" not in next(iter(rendered["$defs"].values()))


# ---------------------------------------------------------------------------
# relax_freeform_object_schema
# ---------------------------------------------------------------------------


class TestRelaxFreeformObjectSchema:
    def test_relaxes_freeform_object(self):
        out = relax_freeform_object_schema(
            {"type": "object", "properties": {"a": {"type": "object"}}}
        )
        assert out["properties"]["a"]["type"] == ["object", "string"]

    def test_leaves_structured_objects_alone(self):
        schema = {
            "type": "object",
            "properties": {
                "sub": {"type": "object", "properties": {"x": {"type": "string"}}}
            },
        }
        out = relax_freeform_object_schema(schema)
        assert out["properties"]["sub"]["type"] == "object"

    def test_deep_copy_does_not_mutate_input(self):
        schema = {"type": "object", "properties": {"a": {"type": "object"}}}
        _ = relax_freeform_object_schema(schema)
        assert schema["properties"]["a"]["type"] == "object"

    _nested = {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"corrected_value": {"type": "object"}},
                },
            }
        },
    }

    def test_relaxes_freeform_objects_at_any_depth(self):
        # ``json_schema_to_pydantic_model`` stringifies free-form objects at
        # every depth, so widening only the top level left nested replies
        # failing validation. See issue #120.
        out = relax_freeform_object_schema(self._nested)
        inner = out["properties"]["issues"]["items"]["properties"]["corrected_value"]
        assert inner["type"] == ["object", "string"]

    def test_relaxed_validation_accepts_nested_json_string(self):
        c = _FakeValidating(free_form_object_as_str=True, client=object())
        payload = '{"issues": [{"corrected_value": "{\\"p\\": 1}"}]}'
        assert c._validate(payload, self._nested)["issues"][0]["corrected_value"] == (
            '{"p": 1}'
        )
        # The object-literal form still validates against the same schema.
        assert c._validate('{"issues": [{"corrected_value": {"p": 1}}]}', self._nested)


# ---------------------------------------------------------------------------
# ValidatingLLMClient configuration surface
# ---------------------------------------------------------------------------


class TestValidatingLLMClientConfig:
    def test_defaults(self):
        c = _FakeValidating(client=object())
        assert c.free_form_object_as_str is False
        assert c.prompt_based_validation is False
        assert c.default_generation_kwargs == {}

    def test_init_kwargs(self):
        c = _FakeValidating(
            free_form_object_as_str=True,
            prompt_based_validation=True,
            default_generation_kwargs={"max_tokens": 42},
            client=object(),
        )
        assert c.free_form_object_as_str is True
        assert c.prompt_based_validation is True
        assert c.default_generation_kwargs == {"max_tokens": 42}

    def test_configure_validation_is_chainable(self):
        c = _FakeValidating(client=object())
        out = c.configure_validation(free_form_object_as_str=True)
        assert out is c
        assert c.free_form_object_as_str is True

    def test_default_generation_kwargs_is_copied(self):
        kw = {"max_tokens": 10}
        c = _FakeValidating(default_generation_kwargs=kw, client=object())
        c.default_generation_kwargs["temperature"] = 0.0
        assert "temperature" not in kw, "caller's dict must not be mutated"


# ---------------------------------------------------------------------------
# _validate honors free_form_object_as_str
# ---------------------------------------------------------------------------


class TestValidatorRelaxation:
    _schema = {"type": "object", "properties": {"a": {"type": "object"}}}

    def test_strict_rejects_json_string_for_object_field(self):
        c = _FakeValidating(client=object())  # default: strict
        with pytest.raises(OutputValidationError):
            c._validate('{"a": "{\\"k\\": 1}"}', self._schema)

    def test_relaxed_accepts_json_string_for_object_field(self):
        c = _FakeValidating(free_form_object_as_str=True, client=object())
        # The LLM returned {"a": "<string representation of object>"} — still
        # valid with relaxed schema.
        got = c._validate('{"a": "arbitrary JSON-ish"}', self._schema)
        assert got == {"a": "arbitrary JSON-ish"}

    def test_relaxed_still_accepts_normal_object(self):
        c = _FakeValidating(free_form_object_as_str=True, client=object())
        assert c._validate('{"a": {"k": 1}}', self._schema) == {"a": {"k": 1}}


# ---------------------------------------------------------------------------
# generate() — prompt_based_validation + default_generation_kwargs
# ---------------------------------------------------------------------------


class TestPromptBasedValidation:
    def test_prompt_based_injects_schema_into_system_prompt(self, monkeypatch):
        observed: list = []
        _install_scripted_generate(monkeypatch, observed, ['{"a": "ok"}'])
        c = _FakeValidating(prompt_based_validation=True, client=object())
        out = c.generate(
            [{"role": "user", "content": "hi"}],
            schema={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        assert out == {"a": "ok"}
        observed_prompt = observed[-1]["prompt"]
        assert observed_prompt[0]["role"] == "system"
        assert "JSON Schema" in observed_prompt[0]["content"]
        assert "response_format" not in observed[-1]


class TestDefaultGenerationKwargs:
    def test_defaults_applied_when_caller_does_not_set(self, monkeypatch):
        observed: list = []
        _install_scripted_generate(monkeypatch, observed, ['{"a": "ok"}'])
        c = _FakeValidating(
            prompt_based_validation=True,
            default_generation_kwargs={"max_tokens": 123, "temperature": 0.0},
            client=object(),
        )
        c.generate(
            [], schema={"type": "object", "properties": {"a": {"type": "string"}}}
        )
        obs = observed[-1]
        assert obs["max_tokens"] == 123
        assert obs["temperature"] == 0.0

    def test_caller_kwargs_win_over_defaults(self, monkeypatch):
        observed: list = []
        _install_scripted_generate(monkeypatch, observed, ['{"a": "ok"}'])
        c = _FakeValidating(
            prompt_based_validation=True,
            default_generation_kwargs={"max_tokens": 123},
            client=object(),
        )
        c.generate(
            [],
            schema={"type": "object", "properties": {"a": {"type": "string"}}},
            max_tokens=999,
        )
        assert observed[-1]["max_tokens"] == 999


# ---------------------------------------------------------------------------
# Wrapped _parse_llm_response: empty + reasoning-budget exhaustion
# ---------------------------------------------------------------------------


class TestSafeParse:
    def test_value_error_becomes_empty_string(self):
        c = _FakeValidating(client=object())
        # Wrapped parser returns "" (retry-worthy) on ValueError/KeyError
        # instead of propagating.
        assert c._parse_llm_response(ValueError("broken")) == ""

    def test_key_error_becomes_empty_string(self):
        c = _FakeValidating(client=object())
        assert c._parse_llm_response(KeyError("missing")) == ""

    def test_reasoning_budget_warning_logged(self, caplog):
        _FakeValidating(client=object())
        # A litellm-shaped response: reasoning_content set, finish_reason=length,
        # content missing — the classic reasoning-budget exhaustion pattern.
        raw = {
            "choices": [
                {
                    "message": {"reasoning_content": "long thinking..."},
                    "finish_reason": "length",
                }
            ]
        }

        # Wrap raw in a class whose attribute-access fails, forcing ValueError.
        class _FailingParse:
            def __init__(self, payload):
                self._p = payload

            # Intentionally broken access pattern in the *orig* parser.
            def __str__(self):
                return "ok"

        # Force the orig parser to raise ValueError, then check the safe
        # parser emits the targeted warning.
        class _C2(_FakeValidating):
            def _parse_llm_response(self, raw):  # will be wrapped
                raise ValueError("empty")

        with caplog.at_level(logging.WARNING, logger="altk.core.llm.output_parser"):
            c2 = _C2(client=object())
            out = c2._parse_llm_response(raw)
        assert out == ""
        assert any("reasoning" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Schema fidelity: constraints must survive the Pydantic round-trip, otherwise
# provider-native structured output is handed a weaker schema than the one the
# response is later validated against.
# ---------------------------------------------------------------------------


class TestSchemaFidelity:
    _bounded = {
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["score"],
    }

    def test_numeric_bounds_stripped_from_wire_schema(self):
        # Bedrock rejects an ``integer`` carrying minimum/maximum outright
        # ("For 'integer' type, properties maximum, minimum are not
        # supported"), and this model is *only* the wire schema — so the
        # bounds must not survive into it. See issue #118.
        prop = json_schema_to_pydantic_model(self._bounded).model_json_schema()[
            "properties"
        ]["score"]
        assert "minimum" not in prop
        assert "maximum" not in prop
        assert prop["type"] == "integer"

    def test_out_of_range_value_is_still_rejected(self):
        # Strictness lives in ``jsonschema`` against the *original* schema,
        # not in the generated model, so dropping the bounds above costs
        # nothing: an out-of-range score is still invalid and gets retried.
        c = _FakeValidating(client=object())
        with pytest.raises(OutputValidationError):
            c._validate('{"score": 9}', self._bounded)
        assert c._validate('{"score": 3}', self._bounded) == {"score": 3}

    def test_string_and_array_constraints_still_survive(self):
        # Only the numeric bounds are dropped; the rest keep the wire schema
        # faithful to the source.
        props = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 2, "pattern": "^[a-z]+$"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
            }
        ).model_json_schema()["properties"]
        assert props["name"]["minLength"] == 2
        assert props["name"]["pattern"] == "^[a-z]+$"
        assert props["tags"]["minItems"] == 1

    def test_enum_survives_inside_array_items(self):
        # A Field-level constraint cannot reach into ``items``; only a real
        # Literal type does. This is what made reasoning models emit
        # out-of-vocabulary values that failed the original schema.
        m = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["A", "B"]},
                    }
                },
            }
        )
        emitted = json.dumps(m.model_json_schema())
        assert '"A"' in emitted and '"B"' in emitted

    def test_nested_array_object_properties_survive(self):
        m = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"rationale": {"type": "string"}},
                            "required": ["rationale"],
                        },
                    }
                },
            }
        )
        assert "rationale" in json.dumps(m.model_json_schema())

    def test_additional_properties_false_forbids_extras(self):
        m = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            }
        )
        assert m.model_config.get("extra") == "forbid"

    def test_enum_with_null_type_is_optional(self):
        m = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "t": {"type": ["string", "null"], "enum": ["x", "y", None]}
                },
            }
        )
        # Must accept None without raising.
        assert m(t=None).t is None


# ---------------------------------------------------------------------------
# Retry behavior for empty / truncated responses (altk-boost#115).
# ---------------------------------------------------------------------------


class TestEmptyResponseRetries:
    def test_value_error_is_retried_not_propagated(self, monkeypatch):
        """A provider raising ValueError('No content...') must consume retries."""
        observed: list = []
        calls = {"n": 0}
        from altk.core.llm.base import BaseLLMClient

        def fake_generate(self, **kwargs):
            observed.append(kwargs)
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("No content or tool calls found in response")
            return '{"a": "ok"}'

        monkeypatch.setattr(BaseLLMClient, "_generate", fake_generate, raising=True)
        c = _FakeValidating(prompt_based_validation=True, client=object())
        out = c.generate(
            [],
            schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            retries=3,
        )
        assert out == {"a": "ok"}
        assert calls["n"] == 2

    def test_empty_reply_retries_original_prompt_not_a_growing_thread(
        self, monkeypatch
    ):
        """No empty assistant turn is appended, and the retry re-sends the
        original prompt: several backends answer a padded conversation with
        another empty response, which would burn every attempt."""
        observed: list = []
        _install_scripted_generate(monkeypatch, observed, ["", '{"a": "ok"}'])
        c = _FakeValidating(prompt_based_validation=True, client=object())
        out = c.generate(
            [{"role": "user", "content": "hi"}],
            schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            retries=3,
        )
        assert out == {"a": "ok"}
        retry_msgs = observed[-1]["prompt"]
        assert not any(
            m.get("role") == "assistant" and not (m.get("content") or "")
            for m in retry_msgs
        ), "an empty assistant turn must never be sent"
        assert len(retry_msgs) == len(observed[0]["prompt"])

    def test_truncated_reply_escalates_max_tokens(self, monkeypatch):
        """finish_reason='length' means the budget was too small; re-asking with
        the same budget would truncate identically, so it must grow."""
        observed: list = []
        from altk.core.llm.base import BaseLLMClient

        truncated = {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}]
        }
        scripted = [truncated, '{"a": "ok"}']

        def fake_generate(self, **kwargs):
            observed.append(kwargs)
            raw = scripted.pop(0)
            return self._parse_llm_response(raw)

        monkeypatch.setattr(BaseLLMClient, "_generate", fake_generate, raising=True)

        class _RealisticParse(_FakeValidating):
            """A truncated reply carries no text: a real provider parser raises
            on the missing content and the wrapper turns that into ``""``."""

            def _parse_llm_response(self, raw):
                if isinstance(raw, dict):
                    raise ValueError("No content or tool calls found in response")
                return str(raw)

        c = _RealisticParse(prompt_based_validation=True, client=object())
        out = c.generate(
            [{"role": "user", "content": "hi"}],
            schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            retries=3,
            max_tokens=1024,
        )
        assert out == {"a": "ok"}
        assert observed[0]["max_tokens"] == 1024
        assert observed[-1]["max_tokens"] > 1024

    def test_native_schema_skipped_when_model_lacks_support(self, monkeypatch):
        """A model that ignores response_format must be steered by the prompt
        instead — sending the kwarg is a no-op there at best."""
        observed: list = []
        _install_scripted_generate(monkeypatch, observed, ['{"a": "ok"}'])

        class _NoNative(_FakeValidating):
            def supports_native_structured_output(self) -> bool:
                return False

        c = _NoNative(client=object())
        c.generate(
            [{"role": "user", "content": "hi"}],
            schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            schema_field="response_format",
        )
        assert "response_format" not in observed[-1]
        # schema went into a system message instead
        assert observed[-1]["prompt"][0]["role"] == "system"


# ---------------------------------------------------------------------------
# Provider-strictness of the rendered wire schema, pinned against ALTK's own
# SPARC metric schemas — the shapes that issues #118 and #120 were filed on.
# ---------------------------------------------------------------------------


def _sparc_runtime_schemas() -> list[tuple[str, dict]]:
    import glob
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pattern = str(
        root / "altk/pre_tool/sparc/function_calling/metrics/*/*_runtime.json"
    )
    out = []
    for path in sorted(glob.glob(pattern)):
        for metric in json.loads(Path(path).read_text()):
            out.append((metric["name"], metric["jsonschema"]))
    return out


def _strictness_violations(node: Any, path: str = "$") -> list[str]:
    """Report every object schema a strict provider would reject."""
    bad: list[str] = []
    if isinstance(node, dict):
        if node.get("additionalProperties") is True:
            bad.append(f"additionalProperties: true at {path}")
        if node.get("type") == "object" and "additionalProperties" not in node:
            bad.append(f"additionalProperties missing at {path}")
        if "minimum" in node or "maximum" in node:
            bad.append(f"numeric bound at {path}")
        for key, value in node.items():
            bad += _strictness_violations(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            bad += _strictness_violations(value, f"{path}[{i}]")
    return bad


class TestSparcSchemasAreProviderStrict:
    @pytest.mark.parametrize("name, schema", _sparc_runtime_schemas())
    def test_rendered_schema_is_strict_safe(self, name, schema):
        rendered = json_schema_to_pydantic_model(
            schema, free_form_object_as_str=True
        ).model_json_schema()
        assert _strictness_violations(rendered) == [], name

    @pytest.mark.parametrize("name, schema", _sparc_runtime_schemas())
    def test_no_numeric_bounds_or_missing_keyword_by_default(self, name, schema):
        # Without the free-form-as-string workaround, free-form objects still
        # render as ``additionalProperties: true`` (that is the documented
        # trade-off of the knob) — but the #118 bounds and the #120 missing
        # keyword must be gone on the default path too.
        rendered = json_schema_to_pydantic_model(schema).model_json_schema()
        leftover = [v for v in _strictness_violations(rendered) if "true" not in v]
        assert leftover == [], name


# ---------------------------------------------------------------------------
# Native-first: a provider that refuses the schema downgrades the client once,
# instead of spending the retry budget re-sending a schema it will refuse again.
# ---------------------------------------------------------------------------


class _Rejects(_FakeValidating):
    """Fails every native attempt the way a strict provider does."""

    def supports_native_structured_output(self) -> bool:
        if self._native_schema_rejected:
            return False
        return True


class TestNativeSchemaDowngrade:
    _schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }

    def _run(self, monkeypatch, error, scripted_ok='{"a": "ok"}'):
        observed: list = []
        from altk.core.llm.base import BaseLLMClient

        def fake_generate(self, **kwargs):  # noqa: ANN001
            observed.append(kwargs)
            if "response_format" in kwargs:
                raise error
            return self._parse_llm_response(scripted_ok)

        monkeypatch.setattr(BaseLLMClient, "_generate", fake_generate, raising=True)
        c = _Rejects(client=object())
        got = c.generate(
            [{"role": "user", "content": "hi"}],
            schema=self._schema,
            schema_field="response_format",
            retries=3,
        )
        return c, observed, got

    def test_schema_rejection_retries_without_the_kwarg(self, monkeypatch):
        err = Exception(
            "litellm.BadRequestError: BedrockException - output_config.format.schema: "
            "For 'integer' type, properties maximum, minimum are not supported"
        )
        c, observed, got = self._run(monkeypatch, err)
        assert got == {"a": "ok"}
        assert len(observed) == 2, "one native attempt, then one prompt-based"
        assert "response_format" in observed[0]
        assert "response_format" not in observed[1]
        # The schema moved into a system message on the second attempt.
        assert observed[1]["prompt"][0]["role"] == "system"

    def test_rejection_is_remembered_for_later_calls(self, monkeypatch):
        c, observed, _ = self._run(
            monkeypatch,
            Exception("Error code: 400 - Invalid schema for response_format"),
        )
        assert c._native_schema_rejected is True
        assert c.supports_native_structured_output() is False
        before = len(observed)
        c.generate(
            [{"role": "user", "content": "again"}],
            schema=self._schema,
            schema_field="response_format",
            retries=3,
        )
        # The second call never tries native again — one request, no probe.
        assert len(observed) == before + 1
        assert "response_format" not in observed[-1]

    def test_unrelated_provider_errors_still_propagate(self, monkeypatch):
        # A 500 or a rate limit must not be mistaken for a schema rejection,
        # or a transient outage would silently disable native output.
        err = RuntimeError("InternalServerError: upstream connect error")
        with pytest.raises(RuntimeError):
            self._run(monkeypatch, err)

    def test_empty_content_under_native_drops_the_kwarg(self, monkeypatch):
        """A model that ignores ``response_format`` answers with empty content
        rather than an error — the only signal available (issue #119)."""
        observed: list = []
        from altk.core.llm.base import BaseLLMClient

        def fake_generate(self, **kwargs):  # noqa: ANN001
            observed.append(kwargs)
            if "response_format" in kwargs:
                return self._parse_llm_response("")
            return self._parse_llm_response('{"a": "ok"}')

        monkeypatch.setattr(BaseLLMClient, "_generate", fake_generate, raising=True)
        c = _Rejects(client=object())
        got = c.generate(
            [{"role": "user", "content": "hi"}],
            schema=self._schema,
            schema_field="response_format",
            retries=3,
        )
        assert got == {"a": "ok"}
        assert "response_format" in observed[0]
        assert "response_format" not in observed[1]


# ---------------------------------------------------------------------------
# Bounded integers ride the wire as an enum, so the provider keeps enforcing
# the range without the keywords strict providers reject.
# ---------------------------------------------------------------------------


class TestBoundedIntegerAsEnum:
    def test_small_range_becomes_an_enum(self):
        prop = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "output": {"type": "integer", "minimum": 1, "maximum": 5}
                },
            }
        ).model_json_schema()["properties"]["output"]
        assert prop["enum"] == [1, 2, 3, 4, 5]
        assert "minimum" not in prop and "maximum" not in prop

    def test_wide_range_is_left_as_a_plain_integer(self):
        prop = json_schema_to_pydantic_model(
            {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 0, "maximum": 1_000_000}
                },
            }
        ).model_json_schema()["properties"]["count"]
        assert "enum" not in prop
        assert prop["type"] == "integer"

    @pytest.mark.parametrize(
        "prop_schema",
        [
            {"type": "number", "minimum": 0, "maximum": 1},  # confidence: not int
            {"type": "integer", "minimum": 1},  # half-open
            {"type": "integer", "maximum": 5},
            {"type": "boolean"},
        ],
    )
    def test_shapes_that_must_not_be_enumerated(self, prop_schema):
        prop = json_schema_to_pydantic_model(
            {"type": "object", "properties": {"x": prop_schema}}
        ).model_json_schema()["properties"]["x"]
        assert "enum" not in prop

    def test_enum_value_is_accepted_and_out_of_range_rejected(self):
        schema = {
            "type": "object",
            "properties": {"output": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["output"],
        }
        c = _FakeValidating(client=object())
        assert c._validate('{"output": 4}', schema) == {"output": 4}
        with pytest.raises(OutputValidationError):
            c._validate('{"output": 7}', schema)


# ---------------------------------------------------------------------------
# The OpenAI/Azure validating clients must reach for native structured output
# by default — they are the providers with the strongest support for it, and
# leaving it off meant every schema was enforced by retrying instead.
# ---------------------------------------------------------------------------


class TestOpenAIClientsDefaultToNative:
    @pytest.mark.parametrize(
        "registry_name, method",
        [
            ("openai.sync.output_val", "generate"),
            ("openai.async.output_val", "generate_async"),
            ("azure_openai.sync.output_val", "generate"),
            ("azure_openai.async.output_val", "generate_async"),
        ],
    )
    def test_schema_field_defaults_to_response_format(self, registry_name, method):
        import inspect

        from altk.core.llm import get_llm

        sig = inspect.signature(getattr(get_llm(registry_name), method))
        assert sig.parameters["schema_field"].default == "response_format"
