from typing import Any, Dict, List, Optional, Type, TypeVar, Union
import os

try:
    from ibm_watsonx_ai import Credentials  # type: ignore
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
except ImportError as e:
    raise ImportError(
        "Please install the ibm-watsonx-ai package: pip install 'toolkit-core[watsonx]'"
    ) from e

from altk.core.llm.base import LLMClient, register_llm, Hook
from altk.core.llm.output_parser import ValidatingLLMClient
from altk.core.llm.types import LLMResponse, GenerationMode, ParameterMapper
from pydantic import BaseModel

from ..consts import WX_URL, WX_API_KEY, WX_PROJECT_ID, WX_SPACE_ID

T = TypeVar("T", bound="WatsonxLLMClient")
SchemaType = Union[Dict[str, Any], Type["BaseModel"], Type[Any]]

# -------------------------------------------------------------------
# 1. Non-validating Watsonx wrapper
# -------------------------------------------------------------------


@register_llm("watsonx")
class WatsonxLLMClient(LLMClient):
    """
    Adapter for IBM watsonx.ai Foundation Model (via ibm_watsonx_ai.foundation_models.ModelInference).

    Supports:
      - text:       sync generation (ModelInference.generate)
      - chat:       sync chat     (ModelInference.chat)
      - text_async: async generation (ModelInference.agenerate)
      - chat_async: async chat       (ModelInference.achat)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        url: Optional[str] = None,
        hooks: Optional[List[Hook]] = None,
        model_id: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        """
        Initialize the Watsonx client.

        Args:
            model_name:   Identifier of the watsonx model (e.g., "meta-llama/llama-3-3-70b-instruct").
            api_key:    (Optional) Your IBM Cloud API Key for watsonx.ai.
            project_id: (Optional) watsonx project ID.
            space_id:   (Optional) watsonx space ID.
            deployment_id: (Optional) watsonx deployment ID.
            url:        (Optional) Base URL for the watsonx endpoint (e.g., "https://us-south.ml.cloud.ibm.com").
            hooks:      Optional observability hooks.
            model_kwargs: Additional keyword args passed to ModelInference constructor.
        """
        self.model_name = model_name
        self._model_kwargs = model_kwargs

        if not url:
            url = os.getenv(WX_URL)
            if not url:
                raise EnvironmentError(
                    f"Missing API URL; please set the '{WX_URL}' environment variable."
                )

        if not api_key:
            api_key = os.getenv(WX_API_KEY)
            if not api_key:
                raise EnvironmentError(
                    f"Missing API key; please set the '{WX_API_KEY}' environment variable."
                )

        if not project_id:
            project_id = os.getenv(WX_PROJECT_ID)
            if not project_id:
                if not space_id:
                    space_id = os.getenv(WX_SPACE_ID)
                raise EnvironmentError(
                    f"Missing project ID; please set the '{WX_PROJECT_ID}' or '{WX_SPACE_ID}' environment variable."
                )

        creds = Credentials(api_key=api_key, url=url)

        if model_id:
            # If model_id is provided, use it as the model_name
            model_name = model_id

        self.model_name = model_name
        self.model_id = model_name

        if not model_name:
            raise ValueError("model_name or model_id must be provided")

        # Assemble provider_kwargs for LLMClient base class
        provider_kwargs: Dict[str, Any] = {
            "model_id": model_name,
            "credentials": creds,
        }
        if project_id:
            provider_kwargs["project_id"] = project_id
        elif space_id:
            provider_kwargs["space_id"] = space_id

        if deployment_id:
            provider_kwargs["deployment_id"] = deployment_id

        # Pass through any additional ModelInference args (params, space_id, verify, validate, etc.)
        provider_kwargs.update(model_kwargs)

        # Initialize underlying ModelInference instance via LLMClient logic
        super().__init__(
            client=None, client_needs_init=True, hooks=hooks, **provider_kwargs
        )

    @classmethod
    def provider_class(cls) -> Type[Any]:
        """
        Underlying SDK client class for watsonx.ai: ModelInference.
        """
        return ModelInference  # type: ignore

    def _register_methods(self) -> None:
        """
        Register how to call watsonx methods:

          - 'text'       -> ModelInference.generate
          - 'text_async' -> ModelInference.agenerate
          - 'chat'       -> ModelInference.chat
          - 'chat_async' -> ModelInference.achat
        """
        self.set_method_config(GenerationMode.TEXT.value, "generate", "prompt")
        self.set_method_config(GenerationMode.TEXT_ASYNC.value, "agenerate", "prompt")
        self.set_method_config(GenerationMode.CHAT.value, "chat", "messages")
        self.set_method_config(GenerationMode.CHAT_ASYNC.value, "achat", "messages")

    def _setup_parameter_mapper(self) -> None:
        """Setup parameter mapping for IBM WatsonX provider."""
        self._parameter_mapper = ParameterMapper()

        # Text generation parameters (based on TextGenParameters)
        self._parameter_mapper.set_text_mapping("temperature", "temperature")
        self._parameter_mapper.set_text_mapping("top_p", "top_p")
        self._parameter_mapper.set_text_mapping("top_k", "top_k")
        self._parameter_mapper.set_text_mapping("max_tokens", "max_new_tokens")
        self._parameter_mapper.set_text_mapping("min_tokens", "min_new_tokens")
        self._parameter_mapper.set_text_mapping(
            "repetition_penalty", "repetition_penalty"
        )
        self._parameter_mapper.set_text_mapping("seed", "random_seed")
        self._parameter_mapper.set_text_mapping("stop_sequences", "stop_sequences")
        self._parameter_mapper.set_text_mapping("timeout", "time_limit")
        self._parameter_mapper.set_text_mapping("decoding_method", "decoding_method")

        # Chat parameters (based on TextChatParameters)
        self._parameter_mapper.set_chat_mapping("temperature", "temperature")
        self._parameter_mapper.set_chat_mapping("top_p", "top_p")
        self._parameter_mapper.set_chat_mapping("max_tokens", "max_tokens")
        self._parameter_mapper.set_chat_mapping(
            "frequency_penalty", "frequency_penalty"
        )
        self._parameter_mapper.set_chat_mapping("presence_penalty", "presence_penalty")
        self._parameter_mapper.set_chat_mapping("seed", "seed")
        self._parameter_mapper.set_chat_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_chat_mapping("timeout", "time_limit")
        self._parameter_mapper.set_chat_mapping("logprobs", "logprobs")
        self._parameter_mapper.set_chat_mapping("top_logprobs", "top_logprobs")

        # Custom transforms for complex parameters
        def transform_echo_text_mode(value: Any, mode: Any) -> dict[str, Any]:
            if mode in ["text", "text_async"]:
                # Text mode can include input text in response
                return (
                    {"include_stop_sequence": value}
                    if "stop" in str(value).lower()
                    else {}
                )
            return {}

        self._parameter_mapper.set_custom_transform("echo", transform_echo_text_mode)

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """
        Extract the generated text and tool calls from a watsonx response.

        - For text generation: raw['results'][0]['generated_text']
        - For chat:           raw['choices'][0]['message']['content']
        """
        content = ""
        tool_calls: list[Any] = []

        # Text‐generation style
        if isinstance(raw, dict) and "results" in raw:
            results = raw["results"]
            if isinstance(results, list) and results:
                first = results[0]
                content = first.get("generated_text", "")

        # Chat style
        elif isinstance(raw, dict) and "choices" in raw:
            choices = raw["choices"]
            if isinstance(choices, list) and choices:
                first = choices[0]
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    # Extract tool calls if present
                    if "tool_calls" in msg and msg["tool_calls"]:
                        tool_calls = []
                        for tool_call in msg["tool_calls"]:
                            tool_call_dict = {
                                "id": tool_call.get("id"),
                                "type": tool_call.get("type", "function"),
                                "function": {
                                    "name": tool_call.get("function", {}).get("name"),
                                    "arguments": tool_call.get("function", {}).get(
                                        "arguments"
                                    ),
                                },
                            }
                            tool_calls.append(tool_call_dict)
                elif "text" in first:
                    content = first["text"]

        if not content and not tool_calls:
            raise ValueError(f"Unexpected watsonx response format: {raw!r}")

        # Return LLMResponse if tool calls exist, otherwise just content
        if tool_calls:
            return LLMResponse(content=content, tool_calls=tool_calls)
        return content

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        mode: Union[str, GenerationMode] = GenerationMode.CHAT,
        generation_args: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Synchronous generation override for WatsonX.

        - If mode is 'chat' and prompt is str, wrap into messages list.
        - If mode is 'text', prompt must be str or list of strings.
        - Handle WatsonX-specific params structure.
        """
        mode_str = mode.value if isinstance(mode, GenerationMode) else mode
        mode_str = mode_str.lower()

        if mode_str not in ("text", "chat"):
            raise KeyError(
                f"Unsupported mode '{mode_str}' for WatsonxLLMClient.generate"
            )

        # Normalize prompt format based on mode
        if mode_str == GenerationMode.CHAT.value:
            # Chat mode expects list of messages
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                prompt = prompt
            else:
                raise TypeError(
                    "For chat mode, prompt must be a string or List[Dict[str,str]]"
                )
        elif mode_str == GenerationMode.TEXT.value:
            # Text mode expects a string prompt
            if isinstance(prompt, list):
                # Convert messages to simple string
                prompt = "\n".join(
                    [msg.get("content", "") for msg in prompt if msg.get("content")]
                )

        # Handle WatsonX params structure
        watsonx_kwargs = {}

        # Extract any existing params from kwargs
        existing_params = kwargs.pop("params", {})

        # Map generation_args to WatsonX parameters if provided
        if generation_args and self._parameter_mapper:
            from altk.core.llm.types import GenerationArgs

            if isinstance(generation_args, GenerationArgs):
                mapped_args = self._parameter_mapper.map_args(generation_args, mode_str)
                # Merge mapped args with existing params
                existing_params.update(mapped_args)

        # Set params if we have any
        if existing_params:
            watsonx_kwargs["params"] = existing_params

        # Add any other kwargs that aren't generation parameters
        watsonx_kwargs.update(kwargs)

        return super().generate(prompt=prompt, mode=mode_str, **watsonx_kwargs)

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        mode: Union[str, GenerationMode] = GenerationMode.CHAT_ASYNC,
        generation_args: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Asynchronous generation override for WatsonX.

        - If mode is 'chat_async', wrap prompt into messages.
        - If mode is 'text_async', prompt must be str or list of strings.
        - Handle WatsonX-specific params structure.
        """
        mode_str = mode.value if isinstance(mode, GenerationMode) else mode
        mode_str = mode_str.lower()

        if mode_str not in ("text_async", "chat_async"):
            raise KeyError(
                f"Unsupported mode '{mode_str}' for WatsonxLLMClient.generate_async"
            )

        if mode_str == GenerationMode.CHAT_ASYNC.value:
            # Chat mode expects list of messages
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                prompt = prompt
            else:
                raise TypeError(
                    "For chat_async mode, prompt must be a string or List[Dict[str,str]]"
                )
        elif mode_str == GenerationMode.TEXT_ASYNC.value:
            # Text mode expects a string prompt
            if isinstance(prompt, list):
                # Convert messages to simple string
                prompt = "\n".join(
                    [msg.get("content", "") for msg in prompt if msg.get("content")]
                )

        # Handle WatsonX params structure
        watsonx_kwargs = {}

        # Extract any existing params from kwargs
        existing_params = kwargs.pop("params", {})

        # Map generation_args to WatsonX parameters if provided
        if generation_args and self._parameter_mapper:
            from altk.core.llm.types import GenerationArgs

            if isinstance(generation_args, GenerationArgs):
                mapped_args = self._parameter_mapper.map_args(generation_args, mode_str)
                # Merge mapped args with existing params
                existing_params.update(mapped_args)

        # Set params if we have any
        if existing_params:
            watsonx_kwargs["params"] = existing_params

        # Add any other kwargs that aren't generation parameters
        watsonx_kwargs.update(kwargs)

        return await super().generate_async(
            prompt=prompt, mode=mode_str, **watsonx_kwargs
        )


# -------------------------------------------------------------------
# 2. Validating Watsonx wrapper
# -------------------------------------------------------------------


@register_llm("watsonx.output_val")
class WatsonxLLMClientOutputVal(ValidatingLLMClient):
    """
    Validating adapter for IBM watsonx.ai Foundation Model.

    Extends ValidatingLLMClient to enforce output structure (via JSON Schema,
    Pydantic models, or simple Python types) on all generate calls,
    with retries and batch support (sync & async).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        url: Optional[str] = None,
        hooks: Optional[List[Hook]] = None,
        model_id: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        """
        Initialize a Watsonx client with output validation.

        Args:
            model_name:   Identifier of the watsonx model.
            api_key:    (Optional) Your IBM Cloud API Key.
            project_id: (Optional) watsonx project ID.
            space_id:   (Optional) watsonx space ID.
            deployment_id: (Optional) watsonx deployment ID.
            url:        (Optional) Base URL for the watsonx endpoint.
            hooks:      Optional observability hooks.
            model_kwargs: Additional arguments passed to the ModelInference constructor.
        """
        self.model_name = model_name
        self._model_kwargs = model_kwargs

        if not url:
            url = os.getenv(WX_URL)
            if not url:
                raise EnvironmentError(
                    f"Missing API URL; please set the '{WX_URL}' environment variable."
                )

        if not api_key:
            api_key = os.getenv(WX_API_KEY)
            if not api_key:
                raise EnvironmentError(
                    f"Missing API key; please set the '{WX_API_KEY}' environment variable."
                )

        if not project_id:
            project_id = os.getenv(WX_PROJECT_ID)
            if not project_id:
                if not space_id:
                    space_id = os.getenv(WX_SPACE_ID)
                raise EnvironmentError(
                    f"Missing project ID; please set the '{WX_PROJECT_ID}' or '{WX_SPACE_ID}' environment variable."
                )

        creds = Credentials(api_key=api_key, url=url)

        if model_id:
            # If model_id is provided, use it as the model_name
            model_name = model_id

        self.model_name = model_name
        self.model_id = model_name

        if not model_name:
            raise ValueError("model_name or model_id must be provided")

        provider_kwargs: Dict[str, Any] = {
            "model_id": model_name,
            "credentials": creds,
        }
        if project_id:
            provider_kwargs["project_id"] = project_id
        elif space_id:
            provider_kwargs["space_id"] = space_id

        if deployment_id:
            provider_kwargs["deployment_id"] = deployment_id
        provider_kwargs.update(model_kwargs)

        super().__init__(
            client=None, client_needs_init=True, hooks=hooks, **provider_kwargs
        )

    @classmethod
    def provider_class(cls) -> Type[Any]:
        """
        Underlying SDK client class: ModelInference.
        """
        return ModelInference  # type: ignore

    def _register_methods(self) -> None:
        """
        Register how to call watsonx methods for validation:

          - 'text'       -> ModelInference.generate
          - 'text_async' -> ModelInference.agenerate
          - 'chat'       -> ModelInference.chat
          - 'chat_async' -> ModelInference.achat
        """
        self.set_method_config(GenerationMode.TEXT.value, "generate", "prompt")
        self.set_method_config(GenerationMode.TEXT_ASYNC.value, "agenerate", "prompt")
        self.set_method_config(GenerationMode.CHAT.value, "chat", "messages")
        self.set_method_config(GenerationMode.CHAT_ASYNC.value, "achat", "messages")

    def _setup_parameter_mapper(self) -> None:
        """Setup parameter mapping for IBM WatsonX provider (same as regular WatsonX)."""
        self._parameter_mapper = ParameterMapper()

        # Text generation parameters (based on TextGenParameters)
        self._parameter_mapper.set_text_mapping("temperature", "temperature")
        self._parameter_mapper.set_text_mapping("top_p", "top_p")
        self._parameter_mapper.set_text_mapping("top_k", "top_k")
        self._parameter_mapper.set_text_mapping("max_tokens", "max_new_tokens")
        self._parameter_mapper.set_text_mapping("min_tokens", "min_new_tokens")
        self._parameter_mapper.set_text_mapping(
            "repetition_penalty", "repetition_penalty"
        )
        self._parameter_mapper.set_text_mapping("seed", "random_seed")
        self._parameter_mapper.set_text_mapping("stop_sequences", "stop_sequences")
        self._parameter_mapper.set_text_mapping("timeout", "time_limit")

        # Chat parameters (based on TextChatParameters)
        self._parameter_mapper.set_chat_mapping("temperature", "temperature")
        self._parameter_mapper.set_chat_mapping("top_p", "top_p")
        self._parameter_mapper.set_chat_mapping("max_tokens", "max_tokens")
        self._parameter_mapper.set_chat_mapping(
            "frequency_penalty", "frequency_penalty"
        )
        self._parameter_mapper.set_chat_mapping("presence_penalty", "presence_penalty")
        self._parameter_mapper.set_chat_mapping("seed", "seed")
        self._parameter_mapper.set_chat_mapping("stop_sequences", "stop")
        self._parameter_mapper.set_chat_mapping("timeout", "time_limit")
        self._parameter_mapper.set_chat_mapping("logprobs", "logprobs")
        self._parameter_mapper.set_chat_mapping("top_logprobs", "top_logprobs")

        def transform_echo_text_mode(value: Any, mode: Any) -> dict[str, Any]:
            if mode in ["text", "text_async"]:
                return (
                    {"include_stop_sequence": value}
                    if "stop" in str(value).lower()
                    else {}
                )
            return {}

        self._parameter_mapper.set_custom_transform("echo", transform_echo_text_mode)

    def _parse_llm_response(self, raw: Any) -> str:
        """
        Extract the assistant-generated text from a watsonx response.

        Same logic as non-validating client.
        """
        if isinstance(raw, dict) and "results" in raw:
            results = raw["results"]
            if isinstance(results, list) and results:
                first = results[0]
                return str(first.get("generated_text", ""))
        if isinstance(raw, dict) and "choices" in raw:
            choices = raw["choices"]
            if isinstance(choices, list) and choices:
                first = choices[0]
                msg = first.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg["content"])
                if "text" in first:
                    return str(first["text"])
        raise ValueError(f"Unexpected watsonx response format: {raw!r}")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        schema: SchemaType,
        retries: int = 3,
        generation_args: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Synchronous chat generation with validation + retries.

        Args:
            prompt: Either a string or a list of chat messages.
            schema: JSON Schema dict, Pydantic model class, or built-in Python type.
            retries: Maximum attempts (including the first).
            generation_args: GenerationArgs to map to provider parameters.
            **kwargs: Passed to the underlying ModelInference call (e.g., temperature).
        """
        mode = "chat"

        # Normalize prompt to chat-messages
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # Handle WatsonX params structure
        watsonx_kwargs = {}

        # Extract any existing params from kwargs
        existing_params = kwargs.pop("params", {})

        # Map generation_args to WatsonX parameters if provided
        if generation_args and self._parameter_mapper:
            from altk.core.llm.types import GenerationArgs

            if isinstance(generation_args, GenerationArgs):
                mapped_args = self._parameter_mapper.map_args(generation_args, mode)
                # Merge mapped args with existing params
                existing_params.update(mapped_args)

        # Set params if we have any
        if existing_params:
            watsonx_kwargs["params"] = existing_params

        # Add any other kwargs that aren't generation parameters
        watsonx_kwargs.update(kwargs)

        return super().generate(
            **{
                "prompt": prompt,
                "schema": schema,
                "retries": retries,
                "mode": mode,
                **self._model_kwargs,
                **watsonx_kwargs,
            }
        )

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        schema: SchemaType,
        retries: int = 3,
        generation_args: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Asynchronous chat generation with validation + retries.

        Args:
            prompt: Either a string or a list of chat messages.
            schema: JSON Schema dict, Pydantic model class, or built-in Python type.
            retries: Maximum attempts.
            generation_args: GenerationArgs to map to provider parameters.
            **kwargs: Passed to the underlying ModelInference call.
        """
        mode = "chat_async"

        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        # Handle WatsonX params structure
        watsonx_kwargs = {}

        # Extract any existing params from kwargs
        existing_params = kwargs.pop("params", {})

        # Map generation_args to WatsonX parameters if provided
        if generation_args and self._parameter_mapper:
            from altk.core.llm.types import GenerationArgs

            if isinstance(generation_args, GenerationArgs):
                mapped_args = self._parameter_mapper.map_args(generation_args, mode)
                # Merge mapped args with existing params
                existing_params.update(mapped_args)

        # Set params if we have any
        if existing_params:
            watsonx_kwargs["params"] = existing_params

        # Add any other kwargs that aren't generation parameters
        watsonx_kwargs.update(kwargs)

        return await super().generate_async(
            **{
                "prompt": prompt,
                "schema": schema,
                "retries": retries,
                "mode": mode,
                **self._model_kwargs,
                **watsonx_kwargs,
            }
        )
