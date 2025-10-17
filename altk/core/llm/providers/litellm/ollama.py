import os
from typing import Any, List, Optional, Union

from altk.core.llm.types import LLMResponse
from .litellm import (
    LiteLLMClient,
    LiteLLMClientOutputVal,
)
from altk.core.llm.base import Hook, register_llm
from ..consts import OLLAMA_API_KEY, OLLAMA_BASE_URL, XGRAMMAR


@register_llm("litellm.ollama")
class OllamaLiteLLMClient(LiteLLMClient):
    """
    Specialized LiteLLMClient for Ollama-hosted models.

    Automatically injects:
      - model_path = "hosted_vllm/{model_name}"
      - api_base URL = "{api_url}/{model_url}/v1"
      - authentication headers
      - guided_decoding_backend
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = "ollama",
        api_url: Optional[str] = "http://localhost:11434",
        guided_decoding_backend: Optional[str] = XGRAMMAR,
        *,
        hooks: Optional[List[Hook]] = None,
        **lite_kwargs: Any,
    ) -> None:
        """
        Initialize the Ollama LiteLLM client.

        Args:
            model_name: Name or identifier of the hosted Ollama model (e.g. "llama2").
            api_key: Ollama API key (defaults to "ollama")
            api_url: Base Ollama API URL (defaults to http://localhost:11434).
            guided_decoding_backend: Backend identifier for guided decoding (defaults to XGRAMMAR).
            hooks: Optional observability hooks to receive events.
            lite_kwargs: Additional parameters passed to the underlying LiteLLM constructor.

        Raises:
            ValueError: If model_url derivation fails.
            EnvironmentError: If API key is missing.
        """
        # Obtain API key from environment if still not provided
        if not api_key:
            api_key = os.getenv(OLLAMA_API_KEY)
            if not api_key:
                raise EnvironmentError(
                    f"Missing API key; please set the '{OLLAMA_API_KEY}' environment variable."
                )

        # Ensure api_url is set, if not provided, it tries to obtain it from the environment variable, otherwise it raises an error.
        if not api_url:
            api_url = os.getenv(OLLAMA_BASE_URL)
            if not api_url:
                raise EnvironmentError(
                    f"Missing API URL; please set the '{OLLAMA_BASE_URL}' environment variable."
                )

        # Construct the full API base endpoint
        api_base = f"{api_url}"

        # Call parent constructor with all required lite parameters
        super().__init__(
            model_name=f"ollama/{model_name}",
            hooks=hooks,
            api_base=api_base,
            api_key=api_key,
            headers={OLLAMA_API_KEY: api_key},
            guided_decoding_backend=guided_decoding_backend,
            **lite_kwargs,
        )


@register_llm("litellm.ollama.output_val")
class OllamaLiteLLMClientOutputVal(LiteLLMClientOutputVal):
    """
    Specialized LiteLLMClientOutputVal for Ollama-hosted models.

    Automatically injects:
      - model_path = "hosted_vllm/{model_name}"
      - api_base URL = "{api_url}/{model_url}/v1"
      - authentication headers with OLLAMA_API_KEY
      - guided_decoding_backend

    Inherits full JSON / Pydantic / type-based output validation,
    retry logic, batch & async support from LiteLLMClientOutputVal.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = "ollama",
        api_url: Optional[str] = "http://localhost:11434",
        guided_decoding_backend: Optional[str] = XGRAMMAR,
        *,
        hooks: Optional[List[Hook]] = None,
        **lite_kwargs: Any,
    ) -> None:
        """
        Initialize the Ollama LiteLLM client with output validation.

        Args:
            model_name: Name of the hosted Ollama model (e.g. "llama2").
            api_key: Ollama API key (defaults to "ollama").
            api_url: Base Ollama API URL (defaults to http://localhost:11434).
            guided_decoding_backend: Backend identifier for guided decoding (defaults to XGRAMMAR).
            hooks: Optional observability hooks to receive events.
            lite_kwargs: Additional parameters passed to the underlying LiteLLM constructor.

        Raises:
            ValueError: If model_url derivation fails.
            EnvironmentError: If API key is missing.
        """
        # Obtain API key from environment if still not provided
        if not api_key:
            api_key = os.getenv(OLLAMA_API_KEY)
            if not api_key:
                raise EnvironmentError(
                    f"Missing `api_key`; please set the '{OLLAMA_API_KEY}' environment variable."
                )

        # Ensure api_url is set, if not provided, it tries to obtain it from the environment variable, otherwise it raises an error.
        if not api_url:
            api_url = os.getenv(OLLAMA_BASE_URL)
            if not api_url:
                raise EnvironmentError(
                    f"Missing `api_url`; please set the '{OLLAMA_BASE_URL}' environment variable."
                )

        # Construct the full API base endpoint
        api_base = f"{api_url}"

        # Call parent constructor with all required lite parameters
        super().__init__(
            model_name=f"ollama/{model_name}",
            hooks=hooks,
            api_base=api_base,
            api_key=api_key,
            headers={OLLAMA_API_KEY: api_key},
            guided_decoding_backend=guided_decoding_backend,
            **lite_kwargs,
        )

    def generate(  # type: ignore
        self,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Synchronous chat generation with validation + retries.
        This method is a wrapper around the generate method of the parent class,
        ensuring that the schema_field is set to None, as Ollama has problems with litellm schema validation.
        Therefore, we disable schema validation for Ollama models, and use the default validation.

        Args:
            **kwargs: Additional keyword arguments passed to the generate method.
        Returns:
            Any: The generated response from the model.
        """

        # Delegate to ValidatingLLMClient.generate
        return super().generate(
            **{
                "schema_field": None,
                **kwargs,
            }
        )

    async def generate_async(  # type: ignore
        self,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Asynchronous chat generation with validation + retries.
        This method is a wrapper around the generate_async method of the parent class,
        ensuring that the schema_field is set to None, as Ollama has problems with litellm schema validation.
        Therefore, we disable schema validation for Ollama models, and use the default validation.

        Args:
            **kwargs: Additional keyword arguments passed to the generate_async method.
        Returns:
            Any: The generated response from the model.
        """

        return await super().generate_async(
            **{
                "schema_field": None,
                **kwargs,
            }
        )
