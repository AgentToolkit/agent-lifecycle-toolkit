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

    def test_unknown_model_assumes_native_support(self):
        """Unknown models keep the previous behavior rather than silently
        switching every call to prompt-based validation."""
        client = LiteLLMClientOutputVal.__new__(LiteLLMClientOutputVal)
        client.model_path = "some-provider/not-a-real-model-xyz"
        assert client.supports_native_structured_output() is True
