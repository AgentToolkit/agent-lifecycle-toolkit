try:
    import litellm
except ImportError as e:
    raise ImportError(
        "litellm is not installed. Please install it with `pip install 'toolkit-core[litellm]'`."
    ) from e

from typing import Any, Dict, List, Optional, Union, Type
from altk.core.llm.base import LLMClient, register_llm, Hook
from altk.core.llm.types import GenerationMode, LLMResponse, ParameterMapper
from pydantic import BaseModel
from altk.core.llm.output_parser import ValidatingLLMClient


@register_llm("litellm")
class LiteLLMClient(LLMClient):
    """
    Adapter for litellm.LiteLLM.

    LiteLLM only supports chat-based interactions, so this client
    restricts usage to:
      - chat: completion with messages
      - chat_async: acompletion with messages

    Text-based modes (text, text_async) are not supported.
    """

    def __init__(
        self, model_name: str, hooks: Optional[List[Hook]] = None, **lite_kwargs: Any
    ) -> None:
        self.model_path = model_name
        self._lite_kwargs = lite_kwargs
        super().__init__(client=None, hooks=hooks, **lite_kwargs)

    @classmethod
    def provider_class(cls) -> type:
        return litellm  # type: ignore

    def _register_methods(self) -> None:
        """Register LiteLLM methods - only chat and chat_async are supported"""
        self.set_method_config(GenerationMode.CHAT.value, "completion", "messages")
        self.set_method_config(
            GenerationMode.CHAT_ASYNC.value, "acompletion", "messages"
        )

    def _setup_parameter_mapper(self) -> None:
        """Setup parameter mapping for LiteLLM provider."""
        self._parameter_mapper = ParameterMapper()

        # LiteLLM uses the same parameters for both chat modes
        # Based on litellm.main.py completion function parameters

        # Direct mappings (same parameter names)
        for param in [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "timeout",
            "stream",
            "logprobs",
            "echo",
        ]:
            self._parameter_mapper.set_chat_mapping(param, param)

        # Parameter name mappings
        self._parameter_mapper.set_chat_mapping(
            "max_tokens", "max_tokens"
        )  # or max_completion_tokens
        self._parameter_mapper.set_chat_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_chat_mapping("top_logprobs", "top_logprobs")

        # Custom transforms for complex parameters
        def transform_top_k(value: Any, mode: Any) -> dict[str, Any]:
            # LiteLLM doesn't have top_k directly, but some models support it via model_kwargs
            return {"top_k": value}

        def transform_repetition_penalty(value: Any, mode: Any) -> dict[str, Any]:
            # Map to frequency_penalty if not already set
            return {"repetition_penalty": value}

        def transform_decoding_method(value: Any, mode: Any) -> dict[str, Any]:
            # LiteLLM doesn't have direct decoding_method, map to temperature for approximation
            if value == "greedy":
                return {"temperature": 0.0}
            elif value == "sample":
                return {}  # Use default temperature
            else:
                return {}  # Unknown method, no transformation

        self._parameter_mapper.set_custom_transform("top_k", transform_top_k)
        self._parameter_mapper.set_custom_transform(
            "repetition_penalty", transform_repetition_penalty
        )
        self._parameter_mapper.set_custom_transform(
            "decoding_method", transform_decoding_method
        )

        # Custom transform for min_tokens (not supported by LiteLLM)
        def transform_min_tokens(value: Any, mode: Any) -> dict[str, Any]:
            # LiteLLM doesn't support min_tokens, so we ignore it and emit a warning
            import warnings

            warnings.warn(
                f"min_tokens parameter ({value}) is not supported by LiteLLM. Parameter will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            return {}  # Return empty dict to ignore the parameter

        self._parameter_mapper.set_custom_transform("min_tokens", transform_min_tokens)

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """
        Extract the assistant-generated text and tool calls from a LiteLLM response.

        Returns:
            LLMResponse object containing both content and tool calls if present,
            or just the content string if no tool calls
        """
        choices = getattr(raw, "choices", None) or raw.get("choices", [])
        if not choices:
            raise ValueError("LiteLLM response missing 'choices'")
        first = choices[0]

        content = ""
        tool_calls: list[Any] = []

        # Extract content
        delta = getattr(first, "delta", None)
        if delta and hasattr(delta, "content") and delta.content:
            content = delta.content

        msg = getattr(first, "message", None)
        if msg:
            if hasattr(msg, "content") and msg.content:
                content = msg.content

            # Extract tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = []
                for tool_call in msg.tool_calls:
                    tool_call_dict = {
                        "id": getattr(tool_call, "id", None),
                        "type": getattr(tool_call, "type", "function"),
                        "function": {
                            "name": getattr(tool_call.function, "name", None),
                            "arguments": getattr(tool_call.function, "arguments", None),
                        },
                    }
                    tool_calls.append(tool_call_dict)

        if hasattr(first, "text") and first.text:
            content = first.text

        # Fallback to dict lookup
        if not content:
            content = first.get("delta", {}).get("content", first.get("text", ""))

        if not content and not tool_calls:
            raise ValueError("No content or tool calls found in response")

        # Return LLMResponse if tool calls exist, otherwise just content
        if tool_calls:
            return LLMResponse(content=content, tool_calls=tool_calls)
        return content

    def generate(  # type: ignore
        self, prompt: Union[str, List[Dict[str, Any]]], **kwargs: Any
    ) -> Union[str, LLMResponse]:
        """
        Generate response from LiteLLM model using chat mode only.

        Args:
            prompt: Input prompt (string or list of message dicts)
            **kwargs: Additional parameters for the model

        Returns:
            LLMResponse if tool calls are present, otherwise string content

        Raises:
            ValueError: If unsupported generation mode is requested
        """
        # Check if mode is specified and validate it
        requested_mode = kwargs.get("mode")
        if requested_mode is not None:
            # Convert enum to string if needed
            if hasattr(requested_mode, "value"):
                mode_str = requested_mode.value
            else:
                mode_str = str(requested_mode)

            # Only allow chat and chat_async modes
            if mode_str not in ["chat", "chat_async"]:
                raise ValueError(
                    f"LiteLLM clients only support chat modes (chat, chat_async). "
                    f"Requested mode '{mode_str}' is not supported."
                )

        model_str = self.model_path
        mode = GenerationMode.CHAT.value

        # Normalize prompt to chat format (list of message dicts)
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif not isinstance(prompt, list):
            raise ValueError("Prompt must be a string or list of message dictionaries")

        return super().generate(
            **{
                "prompt": prompt,
                "model": model_str,
                "mode": mode,
                **self._lite_kwargs,
                **kwargs,
            }
        )

    async def generate_async(
        self, prompt: Union[str, List[Dict[str, Any]]], **kwargs: Any
    ) -> Union[str, LLMResponse]:
        """
        Generate response from LiteLLM model asynchronously using chat_async mode only.

        Args:
            prompt: Input prompt (string or list of message dicts)
            **kwargs: Additional parameters for the model

        Returns:
            LLMResponse if tool calls are present, otherwise string content

        Raises:
            ValueError: If unsupported generation mode is requested
        """
        # Check if mode is specified and validate it
        requested_mode = kwargs.get("mode")
        if requested_mode is not None:
            # Convert enum to string if needed
            if hasattr(requested_mode, "value"):
                mode_str = requested_mode.value
            else:
                mode_str = str(requested_mode)

            # Only allow chat and chat_async modes
            if mode_str not in ["chat", "chat_async"]:
                raise ValueError(
                    f"LiteLLM clients only support chat modes (chat, chat_async). "
                    f"Requested mode '{mode_str}' is not supported."
                )

        model_str = self.model_path
        mode = GenerationMode.CHAT_ASYNC.value

        # Normalize prompt to chat format (list of message dicts)
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif not isinstance(prompt, list):
            raise ValueError("Prompt must be a string or list of message dictionaries")

        return await super().generate_async(
            **{
                "prompt": prompt,
                "model": model_str,
                "mode": mode,
                **self._lite_kwargs,
                **kwargs,
            }
        )


@register_llm("litellm.output_val")
class LiteLLMClientOutputVal(ValidatingLLMClient):
    """
    Validating adapter for litellm.LiteLLM.

    Extends ValidatingLLMClient to enforce output structure (via JSON Schema,
    Pydantic models, or simple Python types) on all generate calls,
    with retries and batch support (sync & async).
    """

    def __init__(
        self, model_name: str, hooks: Optional[List[Hook]] = None, **lite_kwargs: Any
    ) -> None:
        """
        Initialize a LiteLLMClient.

        Args:
            model_name: Identifier or path for the LiteLLM model.
            hooks: Optional observability hooks (callable(event, payload)).
            lite_kwargs: Extra arguments passed when initializing the litellm client.
        """
        self.model_path = model_name
        self._lite_kwargs = lite_kwargs
        super().__init__(client=None, hooks=hooks, **lite_kwargs)

    @classmethod
    def provider_class(cls) -> Type[Any]:
        """
        Underlying SDK client for litellm.

        Must be callable with no arguments (per LLMClient __init__ logic).
        """
        return litellm  # type: ignore

    def _register_methods(self) -> None:
        """
        Register how to call litellm methods - only chat modes are supported:

        - 'chat'       → litellm.completion
        - 'chat_async' → litellm.acompletion
        """
        self.set_method_config(GenerationMode.CHAT.value, "completion", "messages")
        self.set_method_config(
            GenerationMode.CHAT_ASYNC.value, "acompletion", "messages"
        )

    def _setup_parameter_mapper(self) -> None:
        """Setup parameter mapping for LiteLLM provider (same as regular LiteLLM)."""
        self._parameter_mapper = ParameterMapper()

        # LiteLLM uses the same parameters for both chat modes
        for param in [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "timeout",
            "stream",
            "logprobs",
            "echo",
        ]:
            self._parameter_mapper.set_chat_mapping(param, param)

        self._parameter_mapper.set_chat_mapping("max_tokens", "max_tokens")
        self._parameter_mapper.set_chat_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_chat_mapping("top_logprobs", "top_logprobs")

        def transform_top_k(value: Any, mode: Any) -> dict[str, Any]:
            return {"top_k": value}

        def transform_repetition_penalty(value: Any, mode: Any) -> dict[str, Any]:
            return {"repetition_penalty": value}

        def transform_decoding_method(value: Any, mode: Any) -> dict[str, Any]:
            # LiteLLM doesn't have direct decoding_method, map to temperature for approximation
            if value == "greedy":
                return {"temperature": 0.0}
            elif value == "sample":
                return {}  # Use default temperature
            else:
                return {}  # Unknown method, no transformation

        self._parameter_mapper.set_custom_transform("top_k", transform_top_k)
        self._parameter_mapper.set_custom_transform(
            "repetition_penalty", transform_repetition_penalty
        )
        self._parameter_mapper.set_custom_transform(
            "decoding_method", transform_decoding_method
        )

        # Custom transform for min_tokens (not supported by LiteLLM)
        def transform_min_tokens(value: Any, mode: Any) -> dict[str, Any]:
            # LiteLLM doesn't support min_tokens, so we ignore it and emit a warning
            import warnings

            warnings.warn(
                f"min_tokens parameter ({value}) is not supported by LiteLLM. Parameter will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            return {}  # Return empty dict to ignore the parameter

        self._parameter_mapper.set_custom_transform("min_tokens", transform_min_tokens)

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """
        Extract the assistant-generated text and tool calls from a LiteLLM response.

        Returns:
            LLMResponse object containing both content and tool calls if present,
            or just the content string if no tool calls
        """
        choices = getattr(raw, "choices", None) or raw.get("choices", [])
        if not choices:
            raise ValueError("LiteLLM response missing 'choices'")
        first = choices[0]

        content = ""
        tool_calls: list[Any] = []

        # Extract content
        delta = getattr(first, "delta", None)
        if delta and hasattr(delta, "content") and delta.content:
            content = delta.content

        msg = getattr(first, "message", None)
        if msg:
            if hasattr(msg, "content") and msg.content:
                content = msg.content

            # Extract tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = []
                for tool_call in msg.tool_calls:
                    tool_call_dict = {
                        "id": getattr(tool_call, "id", None),
                        "type": getattr(tool_call, "type", "function"),
                        "function": {
                            "name": getattr(tool_call.function, "name", None),
                            "arguments": getattr(tool_call.function, "arguments", None),
                        },
                    }
                    tool_calls.append(tool_call_dict)

        if hasattr(first, "text") and first.text:
            content = first.text

        # Fallback to dict lookup
        if not content:
            content = first.get("delta", {}).get("content", first.get("text", ""))

        if not content and not tool_calls:
            raise ValueError("No content or tool calls found in response")

        # Return LLMResponse if tool calls exist, otherwise just content
        if tool_calls:
            return LLMResponse(content=content, tool_calls=tool_calls)
        return content

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        schema: Union[Dict[str, Any], Type[BaseModel], Type[Any]],
        schema_field: Optional[str] = "response_format",
        retries: int = 3,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Synchronous chat generation with validation + retries.

        Args:
            prompt: Either a string or a list of chat messages.
            schema: JSON Schema dict, Pydantic model class, or built-in Python type.
            retries: Maximum attempts (including the first).
            **kwargs: Passed to the underlying litellm call (e.g. temperature).

        Returns:
            The parsed & validated Python object (or Pydantic instance).
        """
        model = self.model_path
        mode = GenerationMode.CHAT.value

        # Normalize prompt to chat-messages
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif not isinstance(prompt, list):
            raise ValueError(
                "LiteLLM only supports chat mode with string or list of message dictionaries"
            )

        # Delegate to ValidatingLLMClient.generate
        return super().generate(
            **{
                "prompt": prompt,
                "schema": schema,
                "schema_field": schema_field,
                "retries": retries,
                "model": model,
                "mode": mode,
                **self._lite_kwargs,
                **kwargs,
            }
        )

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        schema: Union[Dict[str, Any], Type[BaseModel], Type[Any]],
        schema_field: Optional[str] = "response_format",
        retries: int = 3,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Asynchronous chat generation with validation + retries.
        """
        model = self.model_path
        mode = GenerationMode.CHAT_ASYNC.value

        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        elif not isinstance(prompt, list):
            raise ValueError(
                "LiteLLM only supports chat mode with string or list of message dictionaries"
            )

        return await super().generate_async(
            **{
                "prompt": prompt,
                "schema": schema,
                "schema_field": schema_field,
                "retries": retries,
                "model": model,
                "mode": mode,
                **self._lite_kwargs,
                **kwargs,
            }
        )
