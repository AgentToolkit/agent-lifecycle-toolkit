"""
LLM Client Library

A flexible, extensible framework for working with any large-language-model (LLM)
provider in a uniform way. Supports OpenAI, IBM Watson, LiteLLM, and more.

Key Features:
- Unified interface for multiple LLM providers
- Output validation with JSON Schema and Pydantic models
- Sync and async support
- Retry logic with validation
- Tool calling support
- Observability hooks
"""

from typing import Dict, Type

# Global registry for LLM clients - initialize once
_REGISTRY: Dict[str, Type["BaseLLMClient"]] = {}

# Core imports
from .base import (
    BaseLLMClient,
    LLMClient,
    get_llm,
    register_llm,
    list_available_llms,
    Hook,
    MethodConfig,
)
from .output_parser import OutputValidationError, ValidatingLLMClient
from .types import GenerationMode, LLMResponse

# Export core components
__all__ = [
    "LLMClient",
    "ValidatingLLMClient",
    "get_llm",
    "register_llm",
    "list_available_llms",
    "Hook",
    "MethodConfig",
    "OutputValidationError",
    "GenerationMode",
    "LLMResponse",
]


# Conditional imports for providers
def _import_providers() -> None:
    """Import providers with optional dependencies"""

    from .providers.auto_from_env.auto_from_env import AutoFromEnvLLMClient

    __all__.extend(["AutoFromEnvLLMClient"])

    # LiteLLM providers
    try:
        import litellm
        from .providers.litellm.litellm import LiteLLMClient, LiteLLMClientOutputVal
        from .providers.litellm.watsonx import (
            WatsonxLiteLLMClient,
            WatsonxLiteLLMClientOutputVal,
        )
        from .providers.litellm.ollama import (
            OllamaLiteLLMClient,
            OllamaLiteLLMClientOutputVal,
        )

        __all__.extend(
            [
                "LiteLLMClient",
                "LiteLLMClientOutputVal",
                "OllamaLiteLLMClient",
                "OllamaLiteLLMClientOutputVal",
                "WatsonxLiteLLMClient",
                "WatsonxLiteLLMClientOutputVal",
            ]
        )

    except ImportError:
        pass

    # OpenAI providers
    try:
        import openai
        from .providers.openai.openai import (
            SyncOpenAIClient,
            AsyncOpenAIClient,
            SyncOpenAIClientOutputVal,
            AsyncOpenAIClientOutputVal,
            SyncAzureOpenAIClient,
            AsyncAzureOpenAIClient,
            SyncAzureOpenAIClientOutputVal,
            AsyncAzureOpenAIClientOutputVal,
        )

        __all__.extend(
            [
                "SyncOpenAIClient",
                "AsyncOpenAIClient",
                "SyncOpenAIClientOutputVal",
                "AsyncOpenAIClientOutputVal",
                "SyncAzureOpenAIClient",
                "AsyncAzureOpenAIClient",
                "SyncAzureOpenAIClientOutputVal",
                "AsyncAzureOpenAIClientOutputVal",
            ]
        )

    except ImportError:
        pass

    # IBM Watson providers
    try:
        import ibm_watsonx_ai  # type: ignore
        from .providers.ibm_watsonx_ai.ibm_watsonx_ai import (
            WatsonxLLMClient,
            WatsonxLLMClientOutputVal,
        )

        __all__.extend(["WatsonxLLMClient", "WatsonxLLMClientOutputVal"])

    except ImportError as e:
        print(f"Optional dependency for IBM Watson not found: {e}")
        pass


# Initialize providers on import
_import_providers()
