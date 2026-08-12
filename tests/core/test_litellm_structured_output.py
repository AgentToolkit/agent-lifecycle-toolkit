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


class TestNativeStructuredOutputCapability:
    """Native ``response_format`` is only used where the model honors it."""

    @pytest.mark.parametrize(
        "model_name, expected",
        [
            # Reasoning model with no response-schema support: must fall back.
            ("openai/gpt-oss-120b", False),
            # Model litellm reports as supporting response schemas.
            ("mistralai/mistral-large", True),
        ],
    )
    def test_watsonx_capability_is_per_model(self, model_name, expected):
        client = WatsonxLiteLLMClientOutputVal.__new__(WatsonxLiteLLMClientOutputVal)
        client.model_path = f"watsonx/{model_name}"
        assert client.supports_native_structured_output() is expected

    def test_unknown_model_falls_back_to_prompt(self):
        """A model litellm has no capability data for cannot be assumed to
        honor ``response_format``; the prompt-based path works everywhere.
        See issue #119."""
        client = LiteLLMClientOutputVal.__new__(LiteLLMClientOutputVal)
        client.native_structured_output = None
        client.model_path = "some-provider/not-a-real-model-xyz"
        assert client.supports_native_structured_output() is False

    def test_bare_watsonx_gpt_oss_is_unknown_and_falls_back(self):
        """``watsonx/gpt-oss-120b`` (no ``openai/`` infix) is absent from
        litellm's cost map, which is how issue #119 was reported."""
        client = WatsonxLiteLLMClientOutputVal.__new__(WatsonxLiteLLMClientOutputVal)
        client.native_structured_output = None
        client.model_path = "watsonx/gpt-oss-120b"
        assert client.supports_native_structured_output() is False

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
        client = LiteLLMClientOutputVal.__new__(LiteLLMClientOutputVal)
        client.model_path = model_path
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
