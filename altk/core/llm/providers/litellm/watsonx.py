from typing import Any, List, Optional
from .litellm import (
    LiteLLMClient,
    LiteLLMClientOutputVal,
)
from altk.core.llm.base import Hook, register_llm


@register_llm("litellm.watsonx")
class WatsonxLiteLLMClient(LiteLLMClient):
    """
    Specialized LiteLLM client for Watsox models.

    Automatically prefixes the model path with "watsonx/".
    """

    def __init__(
        self, model_name: str, hooks: Optional[List[Hook]] = None, **lite_kwargs: Any
    ) -> None:
        """
        Initialize a Watsonx LiteLLM client.

        Args:
            model_name: Watsonx model identifier (e.g. "gpt-j-6b").
            hooks: Optional observability hooks (callable(event, payload)).
            lite_kwargs: Additional keyword args passed to the underlying LiteLLM constructor.
        """
        # Construct the model_path for Watsonx
        model_path = f"watsonx/{model_name}"

        # Delegate to the validating LiteLLMClient
        super().__init__(model_name=model_path, hooks=hooks, **lite_kwargs)


@register_llm("litellm.watsonx.output_val")
class WatsonxLiteLLMClientOutputVal(LiteLLMClientOutputVal):
    """
    Validating LiteLLM client for IBM Watsonx models.

    Inherits all JSON/Pydantic/type-based output validation, retry logic,
    and batch/async support from LiteLLMClientOutputVal. Automatically
    prefixes the model path with "watsonx/".
    """

    def __init__(
        self, model_name: str, hooks: Optional[List[Hook]] = None, **lite_kwargs: Any
    ) -> None:
        """
        Initialize a Watsonx LiteLLM client with output validation.

        Args:
            model_name: Watsonx model identifier (e.g. "gpt-j-6b").
            hooks: Optional observability hooks (callable(event, payload)).
            lite_kwargs: Additional keyword args passed to the underlying LiteLLM constructor.
        """
        # Construct the model_path for Watsonx
        model_path = f"watsonx/{model_name}"

        # Delegate to the validating LiteLLMClient
        super().__init__(model_name=model_path, hooks=hooks, **lite_kwargs)
