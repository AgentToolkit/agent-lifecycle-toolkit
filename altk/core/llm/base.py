import inspect
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

# Import from parent module to ensure singleton registry
from altk.core.llm import _REGISTRY
from .types import GenerationMode, GenerationArgs, LLMResponse, ParameterMapper

T = TypeVar("T", bound="BaseLLMClient")
Hook = Callable[[str, Dict[str, Any]], None]


def register_llm(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Register an LLM client class with the global registry.

    Args:
        name: Unique identifier for the client

    Returns:
        Decorator function that registers the class
    """

    def deco(cls: Type["BaseLLMClient"]) -> Type["BaseLLMClient"]:
        _REGISTRY[name] = cls
        return cls

    return deco  # type: ignore


def get_llm(name: str) -> Type["BaseLLMClient"]:
    """
    Retrieve an LLM client class from the global registry.

    Args:
        name: Identifier for the client

    Returns:
        LLMClient class

    Raises:
        ValueError: If no client is registered under the given name
    """
    if not _REGISTRY:
        # Import providers to populate registry
        from altk.core.llm import _import_providers

        _import_providers()

    try:
        return _REGISTRY[name]
    except KeyError:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"No LLMClient registered under '{name}'. Available clients: {available}"
        ) from None


def list_available_llms() -> List[str]:
    """
    List all registered LLM client names.

    Returns:
        List of registered LLM client names
    """
    if not _REGISTRY:
        # Import providers to populate registry
        from altk.core.llm import _import_providers

        _import_providers()

    return list(_REGISTRY.keys())


class MethodConfig:
    """
    Configuration for a provider method.

    Attributes:
        path: Dot-delimited attribute path on the client (e.g. "chat.completions.create").
        prompt_arg: Name of the parameter used for the prompt/messages.
    """

    def __init__(self, path: str, prompt_arg: str) -> None:
        self.path = path
        self.prompt_arg = prompt_arg

    def resolve(self, client: Any) -> Callable[..., Any]:
        """
        Traverse `path` on `client` to retrieve the bound callable.

        Raises:
            AttributeError: if any attribute in the path is missing.
            TypeError: if the resolved attribute is not callable.
        """
        obj: Any = client
        for attr in self.path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                raise AttributeError(
                    f"Could not resolve method path '{self.path}' on {client}"
                )
        if not callable(obj):
            raise TypeError(f"Resolved '{self.path}' is not callable on {client}")
        return obj  # type: ignore


class BaseLLMClient(ABC):
    """Abstract base class for any LLM client."""

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        client_needs_init: bool = False,
        hooks: Optional[List[Hook]] = None,
        **provider_kwargs: Any,
    ) -> None:
        """
        Initialize the wrapper.

        Args:
            client: Pre-initialized provider SDK instance.
            client_needs_init: If True, client is not initialized and will be
                initialized with provider_kwargs.
            hooks: Callables(event_name, payload) for observability.
            provider_kwargs: Passed to provider_class constructor if client is None.

        Raises:
            TypeError: if `client` is provided but is not instance of provider_class.
            RuntimeError: if provider_class instantiation fails.
        """
        self._hooks: List[Hook] = hooks or []
        self._method_configs: Dict[str, MethodConfig] = {}
        self._parameter_mapper: Optional[ParameterMapper] = None

        self._other_kwargs = provider_kwargs
        if client is not None:
            if not isinstance(client, self.provider_class()):
                raise TypeError(
                    f"Expected client of type {self.provider_class().__name__}, "
                    f"got {type(client).__name__}"
                )
            self._client = client
        else:
            if client_needs_init:
                sig = inspect.signature(self.provider_class().__init__)
                init_kwargs = {
                    k: v
                    for k, v in provider_kwargs.items()
                    if k in sig.parameters and k != "self"
                }
                self._other_kwargs = {
                    k: v
                    for k, v in provider_kwargs.items()
                    if k not in sig.parameters and k != "self"
                }
                try:
                    self._client = self.provider_class()(**init_kwargs)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to initialize {self.provider_class().__name__}: {e}"
                    ) from e
            else:
                self._client = self.provider_class()

        self._register_methods()
        self._setup_parameter_mapper()

    @classmethod
    @abstractmethod
    def provider_class(cls) -> Type[Any]:
        """
        Underlying SDK client class, e.g. openai.OpenAI or litellm.LiteLLM.
        """

    @abstractmethod
    def _register_methods(self) -> None:
        """
        Subclasses register MethodConfig entries by calling:
            self.set_method_config(key, path, prompt_arg)
        for keys: 'text', 'chat', 'text_async', 'chat_async'.
        """

    @abstractmethod
    def _setup_parameter_mapper(self) -> None:
        """
        Setup parameter mapping for the provider. Override in subclasses to configure
        mapping from generic GenerationArgs to provider-specific parameters.
        """

    def set_method_config(self, key: str, path: str, prompt_arg: str) -> None:
        """
        Register how to invoke a provider method.

        Args:
            key: Identifier ('text', 'chat', 'text_async', 'chat_async').
            path: Dot-separated path on the SDK client.
            prompt_arg: Name of the argument carrying the prompt/messages.
        """
        self._method_configs[key] = MethodConfig(path, prompt_arg)

    def get_method_config(self, key: str) -> MethodConfig:
        """
        Retrieve a previously registered MethodConfig.

        Raises:
            KeyError: if no config exists for `key`.
        """
        try:
            return self._method_configs[key]
        except KeyError:
            raise KeyError(f"No method config registered under '{key}'") from None

    def get_client(self) -> Any:
        """Return the raw underlying SDK client."""
        return self._client

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        """Invoke all observability hooks, swallowing errors."""
        for hook in self._hooks:
            try:
                hook(event, payload)
            except Exception:
                pass

    def _filter_sensitive_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out sensitive information from call arguments before logging.

        Args:
            args: Dictionary of call arguments that may contain sensitive data

        Returns:
            Filtered dictionary with sensitive keys masked or removed
        """
        # Define sensitive keys that should be filtered out
        sensitive_keys = {
            # IBM WatsonX AI sensitive keys
            "project_id",
            "credentials",
            # General sensitive keys
            "authorization",
            "secret",
            "password",
            "key",
        }

        filtered_args: dict[str, Any] = {}
        for key, value in args.items():
            if key.lower() in sensitive_keys or any(
                sensitive in key.lower()
                for sensitive in ["key", "secret", "auth", "credential"]
            ):
                filtered_args[key] = "SECRET"
            elif isinstance(value, dict) and key == "headers":
                # Special handling for headers dict to filter out sensitive header values
                filtered_headers = {}
                for header_key, header_value in value.items():
                    if any(
                        sensitive in header_key.lower()
                        for sensitive in ["key", "auth", "secret"]
                    ):
                        filtered_headers[header_key] = "SECRET"
                    else:
                        filtered_headers[header_key] = header_value
                filtered_args[key] = filtered_headers
            else:
                filtered_args[key] = value

        return filtered_args

    @abstractmethod
    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        """
        Extract the generated text from a single raw response.

        Raises:
            ValueError: if extraction fails.
        """

    def _generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        mode: Union[str, GenerationMode] = GenerationMode.CHAT,
        generation_args: Optional[GenerationArgs] = None,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Synchronous generation.

        Args:
            prompt: Either a plain string or a list of chat messages dicts.
            mode: Generation mode (text, chat, text_async, chat_async).
            generation_args: Provider-agnostic generation arguments.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text.

        Raises:
            KeyError: if no MethodConfig for `mode`.
            Exception: if the underlying call or parsing fails.
        """
        # Convert enum to string if needed
        mode_str = mode.value if isinstance(mode, GenerationMode) else mode

        # Convert single string prompt to chat format if in chat mode
        if mode is GenerationMode.CHAT and isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        cfg = self.get_method_config(mode_str)
        fn = cfg.resolve(self._client)

        call_args = {cfg.prompt_arg: prompt, **kwargs}

        # Map generic arguments to provider-specific parameters
        if generation_args and self._parameter_mapper:
            mapped_args = self._parameter_mapper.map_args(generation_args, mode_str)
            # Provider-specific kwargs take precedence over mapped args
            for k, v in mapped_args.items():
                if k not in call_args:
                    call_args[k] = v

        sig = inspect.signature(fn)

        for k, v in self._other_kwargs.items():
            if k not in call_args and k in sig.parameters:
                call_args[k] = v

        # Filter sensitive arguments before logging
        filtered_args = self._filter_sensitive_args(call_args)
        self._emit("before_generate", {"mode": mode_str, "args": filtered_args})
        try:
            raw = fn(**call_args)
        except Exception as e:
            self._emit("error", {"phase": "generate", "error": str(e)})
            raise
        text = self._parse_llm_response(raw)
        self._emit("after_generate", {"mode": mode_str, "response": text})
        return text

    async def _generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        mode: Union[str, GenerationMode] = GenerationMode.CHAT_ASYNC,
        generation_args: Optional[GenerationArgs] = None,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Asynchronous generation.

        Uses provider async method if registered, otherwise falls back to thread.

        Args:
            prompt: string or messages list.
            mode: Generation mode (text_async, chat_async).
            generation_args: Provider-agnostic generation arguments.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text.

        Raises:
            Exception: if generation or parsing fails.
        """
        # Convert enum to string if needed
        mode_str = mode.value if isinstance(mode, GenerationMode) else mode

        # Convert single string prompt to chat format if in chat mode
        if mode is GenerationMode.CHAT_ASYNC and isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        if mode_str in self._method_configs:
            cfg = self.get_method_config(mode_str)
            fn = cfg.resolve(self._client)

            call_args = {cfg.prompt_arg: prompt, **kwargs}

            # Map generic arguments to provider-specific parameters
            if generation_args and self._parameter_mapper:
                mapped_args = self._parameter_mapper.map_args(generation_args, mode_str)
                # Provider-specific kwargs take precedence over mapped args
                for k, v in mapped_args.items():
                    if k not in call_args:
                        call_args[k] = v

            sig = inspect.signature(fn)

            for k, v in self._other_kwargs.items():
                if k not in call_args and k in sig.parameters:
                    call_args[k] = v

            # Filter sensitive arguments before logging
            filtered_args = self._filter_sensitive_args(call_args)
            self._emit(
                "before_generate_async", {"mode": mode_str, "args": filtered_args}
            )
            try:
                raw = await fn(**call_args)
            except Exception as e:
                self._emit("error", {"phase": "generate_async", "error": str(e)})
                raise
            text = self._parse_llm_response(raw)
            self._emit("after_generate_async", {"mode": mode_str, "response": text})
            return text

        kwargs["mode"] = mode_str.replace("_async", "")
        kwargs["generation_args"] = generation_args

        # fallback to sync generate in thread
        return await asyncio.to_thread(self._generate, prompt, **kwargs)


class LLMClient(BaseLLMClient):
    """
    Abstract base wrapper for any LLM provider.

    Responsibilities:
      - Accept an existing SDK client or construct one from kwargs.
      - Register provider methods via MethodConfig.
      - Provide sync/async/batch generate calls.
      - Emit observability hooks.
      - Parse raw responses into plain text.
    """

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        mode: Union[str, GenerationMode] = GenerationMode.CHAT,
        generation_args: Optional[GenerationArgs] = None,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Synchronous generation.

        Args:
            prompt: Either a plain string or a list of chat messages dicts.
            mode: Generation mode (text, chat, text_async, chat_async).
            generation_args: Provider-agnostic generation arguments.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text.

        Raises:
            KeyError: if no MethodConfig for `mode`.
            Exception: if the underlying call or parsing fails.
        """
        return self._generate(prompt, mode, generation_args, **kwargs)

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        *,
        mode: Union[str, GenerationMode] = GenerationMode.CHAT_ASYNC,
        generation_args: Optional[GenerationArgs] = None,
        **kwargs: Any,
    ) -> Union[str, LLMResponse]:
        """
        Asynchronous generation.

        Uses provider async method if registered, otherwise falls back to thread.

        Args:
            prompt: string or messages list.
            mode: Generation mode (text_async, chat_async).
            generation_args: Provider-agnostic generation arguments.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text.

        Raises:
            Exception: if generation or parsing fails.
        """
        return await self._generate_async(
            prompt, mode=mode, generation_args=generation_args, **kwargs
        )
