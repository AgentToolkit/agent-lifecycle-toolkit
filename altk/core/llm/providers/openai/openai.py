try:
    import openai
except ImportError as e:
    raise ImportError(
        "OpenAI library is required for this module. Please install it with 'pip install toolkit-core[openai]'"
    ) from e

from typing import Any, Optional, Dict, List, Union
from altk.core.llm.base import BaseLLMClient, LLMClient, register_llm
from altk.core.llm.output_parser import ValidatingLLMClient
from altk.core.llm.types import LLMResponse, GenerationMode, ParameterMapper


class BaseOpenAIClient(LLMClient):
    """Base class for OpenAI and Azure OpenAI clients with shared parameter mapping"""

    def _setup_parameter_mapper(self) -> None:
        """Set up parameter mapper for OpenAI-compatible APIs"""
        self._parameter_mapper = ParameterMapper()

        # Direct mappings for text and chat modes
        # Text mode parameters
        self._parameter_mapper.set_text_mapping("max_tokens", "max_tokens")
        self._parameter_mapper.set_text_mapping("temperature", "temperature")
        self._parameter_mapper.set_text_mapping("top_p", "top_p")
        self._parameter_mapper.set_text_mapping("presence_penalty", "presence_penalty")
        self._parameter_mapper.set_text_mapping(
            "frequency_penalty", "frequency_penalty"
        )
        self._parameter_mapper.set_text_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_text_mapping("logprobs", "logprobs")
        self._parameter_mapper.set_text_mapping("top_logprobs", "top_logprobs")
        self._parameter_mapper.set_text_mapping("echo", "echo")
        self._parameter_mapper.set_text_mapping("seed", "seed")
        self._parameter_mapper.set_text_mapping("stream", "stream")
        self._parameter_mapper.set_text_mapping("timeout", "timeout")

        # Chat mode parameters
        self._parameter_mapper.set_chat_mapping("max_tokens", "max_tokens")
        self._parameter_mapper.set_chat_mapping("temperature", "temperature")
        self._parameter_mapper.set_chat_mapping("top_p", "top_p")
        self._parameter_mapper.set_chat_mapping("presence_penalty", "presence_penalty")
        self._parameter_mapper.set_chat_mapping(
            "frequency_penalty", "frequency_penalty"
        )
        self._parameter_mapper.set_chat_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_chat_mapping("logprobs", "logprobs")
        self._parameter_mapper.set_chat_mapping("top_logprobs", "top_logprobs")
        self._parameter_mapper.set_chat_mapping("seed", "seed")
        self._parameter_mapper.set_chat_mapping("stream", "stream")
        self._parameter_mapper.set_chat_mapping("timeout", "timeout")

        # Custom transform for decoding_method
        def transform_decoding_method(value: Any, mode: Any) -> dict[str, Any]:
            # OpenAI doesn't have direct decoding_method, map to temperature for approximation
            if value == "greedy":
                return {"temperature": 0.0}
            elif value == "sample":
                return {}  # Use default temperature
            else:
                return {}  # Unknown method, no transformation

        # Custom transform for min_tokens (not supported by OpenAI)
        def transform_min_tokens(value: Any, mode: Any) -> dict[str, Any]:
            # OpenAI doesn't support min_tokens, so we ignore it and emit a warning
            import warnings

            warnings.warn(
                f"min_tokens parameter ({value}) is not supported by OpenAI. Parameter will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            return {}  # Return empty dict to ignore the parameter

        self._parameter_mapper.set_custom_transform(
            "decoding_method", transform_decoding_method
        )
        self._parameter_mapper.set_custom_transform("min_tokens", transform_min_tokens)


class BaseValidatingOpenAIClient(ValidatingLLMClient):
    """Base class for validating OpenAI and Azure OpenAI clients with shared parameter mapping"""

    def _setup_parameter_mapper(self) -> None:
        """Set up parameter mapper for OpenAI-compatible APIs"""
        self._parameter_mapper = ParameterMapper()

        # Direct mappings for text and chat modes
        # Text mode parameters
        self._parameter_mapper.set_text_mapping("max_tokens", "max_tokens")
        self._parameter_mapper.set_text_mapping("temperature", "temperature")
        self._parameter_mapper.set_text_mapping("top_p", "top_p")
        self._parameter_mapper.set_text_mapping("presence_penalty", "presence_penalty")
        self._parameter_mapper.set_text_mapping(
            "frequency_penalty", "frequency_penalty"
        )
        self._parameter_mapper.set_text_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_text_mapping("logprobs", "logprobs")
        self._parameter_mapper.set_text_mapping("top_logprobs", "top_logprobs")
        self._parameter_mapper.set_text_mapping("echo", "echo")
        self._parameter_mapper.set_text_mapping("seed", "seed")
        self._parameter_mapper.set_text_mapping("stream", "stream")
        self._parameter_mapper.set_text_mapping("timeout", "timeout")

        # Chat mode parameters
        self._parameter_mapper.set_chat_mapping("max_tokens", "max_tokens")
        self._parameter_mapper.set_chat_mapping("temperature", "temperature")
        self._parameter_mapper.set_chat_mapping("top_p", "top_p")
        self._parameter_mapper.set_chat_mapping("presence_penalty", "presence_penalty")
        self._parameter_mapper.set_chat_mapping(
            "frequency_penalty", "frequency_penalty"
        )
        self._parameter_mapper.set_chat_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_chat_mapping("logprobs", "logprobs")
        self._parameter_mapper.set_chat_mapping("top_logprobs", "top_logprobs")
        self._parameter_mapper.set_chat_mapping("seed", "seed")
        self._parameter_mapper.set_chat_mapping("stream", "stream")
        self._parameter_mapper.set_chat_mapping("tools", "tools")
        self._parameter_mapper.set_chat_mapping("tool_choice", "tool_choice")
        self._parameter_mapper.set_chat_mapping("timeout", "timeout")

        # Custom transform for decoding_method
        def transform_decoding_method(value: Any, mode: Any) -> dict[str, Any]:
            # OpenAI doesn't have direct decoding_method, map to temperature for approximation
            if value == "greedy":
                return {"temperature": 0.0}
            elif value == "sample":
                return {}  # Use default temperature
            else:
                return {}  # Unknown method, no transformation

        # Custom transform for min_tokens (not supported by OpenAI)
        def transform_min_tokens(value: Any, mode: Any) -> dict[str, Any]:
            # OpenAI doesn't support min_tokens, so we ignore it and emit a warning
            import warnings

            warnings.warn(
                f"min_tokens parameter ({value}) is not supported by OpenAI. Parameter will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            return {}  # Return empty dict to ignore the parameter

        self._parameter_mapper.set_custom_transform(
            "decoding_method", transform_decoding_method
        )
        self._parameter_mapper.set_custom_transform("min_tokens", transform_min_tokens)


@register_llm("openai.sync")
class SyncOpenAIClient(BaseOpenAIClient, BaseLLMClient):
    """
    Adapter for openai.OpenAI.

    Supports:
      - text: completions.create
      - chat: chat.completions.create
      - text_async: completions.create
      - chat_async: chat.completions.create
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.OpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.TEXT.value, "completions.create", "prompt"
        )
        self.set_method_config(
            GenerationMode.CHAT.value, "chat.completions.create", "messages"
        )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)

    def generate(  # type: ignore
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        mode: Union[str, GenerationMode] = GenerationMode.CHAT,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Generate with proper prompt format validation based on mode.

        Args:
            prompt: Input prompt
            mode: Generation mode (text or chat)
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        mode_str = mode.value if isinstance(mode, GenerationMode) else mode

        # Validate prompt format based on mode
        if mode_str == GenerationMode.TEXT.value:
            # Text mode expects a string prompt
            if isinstance(prompt, list):
                # Convert messages to simple string
                prompt = "\n".join(
                    [msg.get("content", "") for msg in prompt if msg.get("content")]
                )
        elif mode_str == GenerationMode.CHAT.value:
            # Chat mode expects list of messages
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

        return super().generate(prompt=prompt, mode=mode_str, **kwargs)


@register_llm("openai.async")
class AsyncOpenAIClient(BaseOpenAIClient, LLMClient):
    """
    Adapter for openai.AsyncOpenAI.
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.AsyncOpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.TEXT_ASYNC.value, "completions.create", "prompt"
        )
        self.set_method_config(
            GenerationMode.CHAT_ASYNC.value, "chat.completions.create", "messages"
        )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        mode: Union[str, GenerationMode] = GenerationMode.CHAT_ASYNC,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Generate async with proper prompt format validation based on mode.

        Args:
            prompt: Input prompt
            mode: Generation mode (text_async or chat_async)
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        mode_str = mode.value if isinstance(mode, GenerationMode) else mode

        # Validate prompt format based on mode
        if mode_str == GenerationMode.TEXT_ASYNC.value:
            # Text mode expects a string prompt
            if isinstance(prompt, list):
                # Convert messages to simple string
                prompt = "\n".join(
                    [msg.get("content", "") for msg in prompt if msg.get("content")]
                )
        elif mode_str == GenerationMode.CHAT_ASYNC.value:
            # Chat mode expects list of messages
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

        return await super().generate_async(prompt=prompt, mode=mode_str, **kwargs)


@register_llm("openai.sync.output_val")
class SyncOpenAIClientOutputVal(BaseOpenAIClient, ValidatingLLMClient):
    """
    Validating adapter for openai.OpenAI with structured output support.
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.OpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.CHAT.value, "chat.completions.parse", "messages"
        )

    def generate(  # type: ignore
        self,
        prompt: Union[str, List[Dict[str, str]]],
        schema: Optional[Any] = None,
        schema_field: Optional[str] = "response_format",
        retries: int = 3,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """Generate with OpenAI structured output support"""
        # Convert string prompts to message format for chat
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # For OpenAI, we can use their structured output feature
        if schema_field == "response_format" and schema:
            # Let OpenAI handle parsing
            return super().generate(
                prompt=prompt,
                schema=schema,
                schema_field=schema_field,
                retries=retries,
                **kwargs,
            )
        else:
            # Fall back to our validation logic
            return super().generate(
                prompt=prompt,
                schema=schema,
                schema_field=None,  # Don't use OpenAI's structured output
                retries=retries,
                **kwargs,
            )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)


@register_llm("openai.async.output_val")
class AsyncOpenAIClientOutputVal(BaseOpenAIClient, ValidatingLLMClient):
    """
    Validating adapter for openai.AsyncOpenAI with structured output support.
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.AsyncOpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.CHAT_ASYNC.value, "chat.completions.parse", "messages"
        )

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        schema: Optional[Any] = None,
        schema_field: Optional[str] = "response_format",
        retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Generate with OpenAI structured output support"""
        # Convert string prompts to message format for chat
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # For OpenAI, we can use their structured output feature
        if schema_field == "response_format" and schema:
            # Let OpenAI handle parsing
            return await super().generate_async(
                prompt=prompt,
                schema=schema,
                schema_field=schema_field,
                retries=retries,
                **kwargs,
            )
        else:
            # Fall back to our validation logic
            return await super().generate_async(
                prompt=prompt,
                schema=schema,
                schema_field=None,  # Don't use OpenAI's structured output
                retries=retries,
                **kwargs,
            )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)


@register_llm("azure_openai.sync")
class SyncAzureOpenAIClient(BaseOpenAIClient):
    """
    Adapter for openai.AzureOpenAI.

    Supports:
      - text: completions.create
      - chat: chat.completions.create
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.AzureOpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.TEXT.value, "completions.create", "prompt"
        )
        self.set_method_config(
            GenerationMode.CHAT.value, "chat.completions.create", "messages"
        )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)


@register_llm("azure_openai.async")
class AsyncAzureOpenAIClient(BaseOpenAIClient):
    """
    Adapter for openai.AsyncAzureOpenAI.
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.AsyncAzureOpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.TEXT_ASYNC.value, "completions.create", "prompt"
        )
        self.set_method_config(
            GenerationMode.CHAT_ASYNC.value, "chat.completions.create", "messages"
        )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)


@register_llm("azure_openai.sync.output_val")
class SyncAzureOpenAIClientOutputVal(BaseValidatingOpenAIClient):
    """
    Validating adapter for openai.AzureOpenAI with structured output support.
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.AzureOpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.CHAT.value, "chat.completions.parse", "messages"
        )

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        schema: Optional[Any] = None,
        schema_field: Optional[str] = "response_format",
        retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Generate with Azure OpenAI structured output support"""
        # Convert string prompts to message format for chat
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # For Azure OpenAI, we can use their structured output feature
        if schema_field == "response_format" and schema:
            # Let Azure OpenAI handle parsing
            return super().generate(
                prompt=prompt,
                schema=schema,
                schema_field=schema_field,
                retries=retries,
                **kwargs,
            )
        else:
            # Fall back to our validation logic
            return super().generate(
                prompt=prompt,
                schema=schema,  # type: ignore
                schema_field=None,  # Don't use Azure OpenAI's structured output
                retries=retries,
                **kwargs,
            )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)


@register_llm("azure_openai.async.output_val")
class AsyncAzureOpenAIClientOutputVal(BaseValidatingOpenAIClient):
    """
    Validating adapter for openai.AsyncAzureOpenAI with structured output support.
    """

    def __init__(self, *, client: Optional[Any] = None, **provider_kwargs: Any) -> None:
        client_needs_init = client is None
        if client_needs_init:
            super().__init__(client_needs_init=True, **provider_kwargs)
        else:
            super().__init__(client=client, **provider_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return openai.AsyncAzureOpenAI

    def _register_methods(self) -> None:
        self.set_method_config(
            GenerationMode.CHAT_ASYNC.value, "chat.completions.parse", "messages"
        )

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        schema: Optional[Any] = None,
        schema_field: Optional[str] = "response_format",
        retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Generate with Azure OpenAI structured output support"""
        # Convert string prompts to message format for chat
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # For Azure OpenAI, we can use their structured output feature
        if schema_field == "response_format" and schema:
            # Let Azure OpenAI handle parsing
            return await super().generate_async(
                prompt=prompt,
                schema=schema,
                schema_field=schema_field,
                retries=retries,
                **kwargs,
            )
        else:
            # Fall back to our validation logic
            return await super().generate_async(
                prompt=prompt,
                schema=schema,  # type: ignore
                schema_field=None,  # Don't use Azure OpenAI's structured output
                retries=retries,
                **kwargs,
            )

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """Parse response, handling both content and tool calls"""
        return _parse_llm_response(raw)


def _parse_llm_response(raw: Any) -> Union[str, LLMResponse]:
    """
    Helper function to parse OpenAI response and extract content and tool calls.

    Args:
        raw: The raw response from OpenAI API

    Returns:
        str: If no tool calls, returns just the content
        LLMResponse: If tool calls exist, returns object with content and tool_calls
    """
    if (
        not raw
        or not hasattr(raw, "choices")
        or not raw.choices
        or not isinstance(raw.choices, list)
    ):
        raise ValueError("Invalid OpenAI response format")

    first = raw.choices[0]
    content = ""
    tool_calls = []

    # Extract content
    if hasattr(first, "message"):
        content = first.message.content or ""

        # Extract tool calls if present
        if hasattr(first.message, "tool_calls") and first.message.tool_calls:
            for tool_call in first.message.tool_calls:
                tool_call_dict = {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                tool_calls.append(tool_call_dict)

    elif hasattr(first, "text"):
        content = first.text
    else:
        # Fallback to dict access
        content = first.get("message", {}).get("content", first.get("text", ""))

    if not content and not tool_calls:
        raise ValueError("No content or tool calls found in OpenAI response")

    # Return LLMResponse if tool calls exist, otherwise just content
    if tool_calls:
        return LLMResponse(content=content, tool_calls=tool_calls)
    return content
