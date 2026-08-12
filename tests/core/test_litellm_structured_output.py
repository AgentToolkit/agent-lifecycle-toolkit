"""Offline tests for the LiteLLM validating client's response handling.

Covers two production failures seen with reasoning ("thinking") models such as
``gpt-oss`` on watsonx:

- the answer arrives in ``reasoning_content`` while ``content`` is empty, and
- the model ignores a native ``response_format`` schema entirely, so the schema
  has to be injected into the system prompt instead.
"""

from __future__ import annotations

import pytest

from altk.core.llm.providers.litellm.litellm import LiteLLMClientOutputVal
from altk.core.llm.providers.litellm.watsonx import WatsonxLiteLLMClientOutputVal


def _response(content=None, reasoning=None, finish="stop"):
    """Build a litellm-shaped response object."""
    from litellm.types.utils import Choices, Message, ModelResponse

    kwargs = {"content": content, "role": "assistant"}
    if reasoning is not None:
        kwargs["reasoning_content"] = reasoning
    return ModelResponse(
        choices=[Choices(message=Message(**kwargs), finish_reason=finish, index=0)]
    )


class TestReasoningContentFallback:
    """``_parse_llm_response`` must not discard a usable reasoning-only reply."""

    def test_reasoning_content_used_when_content_empty(self):
        client = LiteLLMClientOutputVal.__new__(LiteLLMClientOutputVal)
        raw = _response(content="", reasoning='{"a": 1}')
        # Call the unwrapped parser directly: the instance-level wrapper is
        # installed in __init__, which we skip here on purpose.
        assert LiteLLMClientOutputVal._parse_llm_response(client, raw) == '{"a": 1}'

    def test_content_still_preferred_over_reasoning(self):
        client = LiteLLMClientOutputVal.__new__(LiteLLMClientOutputVal)
        raw = _response(content='{"real": true}', reasoning="thinking out loud")
        assert (
            LiteLLMClientOutputVal._parse_llm_response(client, raw) == '{"real": true}'
        )

    def test_still_raises_when_nothing_usable(self):
        client = LiteLLMClientOutputVal.__new__(LiteLLMClientOutputVal)
        with pytest.raises(ValueError, match="No content or tool calls"):
            LiteLLMClientOutputVal._parse_llm_response(client, _response(content=""))


def _bare_client(cls, model_path):
    """A client with only the attributes the capability gate reads.

    ``__new__`` skips ``__init__`` on purpose (no provider connection), so the
    instance state the gate consults is set explicitly here.
    """
    client = cls.__new__(cls)
    client.model_path = model_path
    client.native_structured_output = None
    client._native_schema_rejected = False
    return client


class TestNativeStructuredOutputCapability:
    """Native ``response_format`` is preferred wherever it can work."""

    @pytest.mark.parametrize(
        "cls, model_path",
        [
            # No metadata at all: gateway/proxy strings, and the model from
            # issue #119 that turned out to honor response_format fully. Native
            # is attempted — it costs one request per call instead of several,
            # and a refusal self-corrects.
            (WatsonxLiteLLMClientOutputVal, "watsonx/mistral-large-2512"),
            (LiteLLMClientOutputVal, "some-provider/not-a-real-model-xyz"),
            (LiteLLMClientOutputVal, "openai/aws/claude-haiku-4-5"),
            # Known and reported as supporting response schemas.
            (WatsonxLiteLLMClientOutputVal, "watsonx/mistralai/mistral-large"),
        ],
    )
    def test_native_is_attempted_when_not_known_unsupported(self, cls, model_path):
        assert _bare_client(cls, model_path).supports_native_structured_output() is True

    @pytest.mark.parametrize(
        "model_path",
        [
            # Known-unsupported is trusted: forcing native on the smaller
            # watsonx models made them worse, because constrained decoding
            # drives them to emit whitespace until the token budget is gone.
            "watsonx/openai/gpt-oss-120b",
            "watsonx/mistralai/mistral-small-3-1-24b-instruct-2503",
        ],
    )
    def test_known_unsupported_skips_native(self, model_path):
        client = _bare_client(WatsonxLiteLLMClientOutputVal, model_path)
        assert client.supports_native_structured_output() is False
        # ...but a caller who measured otherwise can still opt in.
        client.native_structured_output = True
        assert client.supports_native_structured_output() is True

    def test_rejection_latches_off_native(self):
        """Once a provider refuses the schema, stop offering it."""
        client = _bare_client(LiteLLMClientOutputVal, "openai/aws/claude-haiku-4-5")
        assert client.supports_native_structured_output() is True
        client._note_native_schema_rejected(
            ValueError("BedrockException: output_config.format.schema not supported")
        )
        assert client.supports_native_structured_output() is False

    def test_explicit_override_beats_a_recorded_rejection(self):
        client = _bare_client(LiteLLMClientOutputVal, "openai/gpt-4o")
        client._native_schema_rejected = True
        client.native_structured_output = True
        assert client.supports_native_structured_output() is True

    @pytest.mark.parametrize(
        "model_path, override, expected",
        [
            # Unknown proxy/gateway model the caller knows does honor it.
            ("openai/some-gateway-model-xyz", True, True),
            # Known-supporting model the caller wants steered by prompt anyway.
            ("openai/gpt-4o", False, False),
        ],
    )
    def test_native_structured_output_override_wins(
        self, model_path, override, expected
    ):
        client = _bare_client(LiteLLMClientOutputVal, model_path)
        client.native_structured_output = override
        assert client.supports_native_structured_output() is expected


class TestValidationKwargsDoNotReachTheProvider:
    """Validation knobs configure ALTK, not the completion request.

    ``_lite_kwargs`` is replayed on every call, so a knob passed to the
    constructor used to travel with it and the provider rejected the request:
    "Unrecognized request arguments supplied: free_form_object_as_str,
    native_structured_output".
    """

    def test_knobs_are_stripped_from_replayed_kwargs(self):
        # No request is made here, so the client needs no credentials.
        client = LiteLLMClientOutputVal(
            model_name="openai/gpt-4o",
            api_base="https://example.invalid/v1",
            native_structured_output=True,
            free_form_object_as_str=True,
            prompt_based_validation=False,
            default_generation_kwargs={"max_tokens": 32},
        )
        assert set(client._lite_kwargs) == {"api_base"}
        # ...while still taking effect on the client itself.
        assert client.native_structured_output is True
        assert client.free_form_object_as_str is True
        assert client.default_generation_kwargs == {"max_tokens": 32}
