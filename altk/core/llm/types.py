from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass


class GenerationMode(Enum):
    """
    Enum for different generation modes across LLM providers.
    """

    TEXT = "text"
    CHAT = "chat"
    TEXT_ASYNC = "text_async"
    CHAT_ASYNC = "chat_async"


class LLMResponse:
    """Response object that can contain both content and tool calls"""

    def __init__(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def __str__(self) -> str:
        """Return the content of the response as a string."""
        return self.content

    def __repr__(self) -> str:
        """Return a string representation of the LLMResponse object."""
        return f"LLMResponse(content='{self.content}', tool_calls={self.tool_calls})"


@dataclass
class GenerationArgs:
    """
    Provider-agnostic generation arguments.

    These arguments represent common parameters across LLM providers.
    Each provider should implement mapping from these generic arguments
    to their specific parameter names.
    """

    # Core generation parameters
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    # Penalties and biases
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None

    # Stop conditions
    stop_sequences: Optional[List[str]] = None

    # Randomness control
    seed: Optional[int] = None

    # Generation control
    decoding_method: Optional[str] = None  # "greedy" or "sample"

    # Output control
    stream: Optional[bool] = None
    echo: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    # Other common parameters
    timeout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ParameterMapper:
    """
    Abstract base class for mapping generic generation arguments to provider-specific parameters.
    """

    def __init__(self) -> None:
        self._text_mappings: Dict[str, str] = {}
        self._chat_mappings: Dict[str, str] = {}
        self._custom_transforms: Dict[str, Callable[[Any, str], Any]] = {}

    def set_text_mapping(self, generic_param: str, provider_param: str) -> None:
        """Set parameter mapping for text generation mode."""
        self._text_mappings[generic_param] = provider_param

    def set_chat_mapping(self, generic_param: str, provider_param: str) -> None:
        """Set parameter mapping for chat generation mode."""
        self._chat_mappings[generic_param] = provider_param

    def set_custom_transform(
        self, generic_param: str, transform_func: Callable[[Any, str], Any]
    ) -> None:
        """Set a custom transformation function for a parameter."""
        self._custom_transforms[generic_param] = transform_func

    def map_args(self, args: GenerationArgs, mode: str) -> Dict[str, Any]:
        """
        Map generic arguments to provider-specific parameters.

        Args:
            args: Generic generation arguments
            mode: Generation mode ('text', 'chat', 'text_async', 'chat_async')

        Returns:
            Dictionary of provider-specific parameters
        """
        # Determine which mapping to use based on mode
        is_chat_mode = mode in ["chat", "chat_async"]
        mappings = self._chat_mappings if is_chat_mode else self._text_mappings

        provider_args = {}
        args_dict = args.to_dict()

        for generic_param, value in args_dict.items():
            # Check for custom transform first
            if generic_param in self._custom_transforms:
                transform_func = self._custom_transforms[generic_param]
                transformed = transform_func(value, mode)
                if isinstance(transformed, dict):
                    provider_args.update(transformed)
                else:
                    # If transform returns a single value, use the generic param name
                    provider_args[generic_param] = transformed
            # Use direct mapping if available
            elif generic_param in mappings:
                provider_args[mappings[generic_param]] = value
            # Fall back to generic parameter name
            else:
                provider_args[generic_param] = value

        return provider_args
