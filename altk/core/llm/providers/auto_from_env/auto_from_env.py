import os
import inspect
from typing import Type, Any, Union

from altk.core.llm.base import LLMClient, register_llm, get_llm
from altk.core.llm.types import LLMResponse


@register_llm("auto_from_env")
class AutoFromEnvLLMClient(LLMClient):
    """
    Default adapter for ALTK, will determine which provider to use based on environment variables.

    Expects the following environment variables to be set:
        - ALTK_MODEL_NAME: optional, model name, assumes litellm if ALTK_PROVIDER_NAME not set
        - ALTK_LLM_PROVIDER: optional, the corresponding name in the LLMClient registry
    If both are not set, client is set to None
    """

    def __init__(self) -> None:
        # We assume LiteLLM if a specific provider is not there
        provider_name = os.getenv("ALTK_LLM_PROVIDER")
        self.model_name = os.getenv("ALTK_MODEL_NAME")
        self.model_name_in_generate = False
        if not self.model_name and not provider_name:
            # If neither is set, does nothing
            self._chosen_provider = None
        else:
            if not provider_name:
                # If only the model name is provided, assume LiteLLM
                provider_name = "litellm"
            provider_type = get_llm(provider_name)
            init_sig = inspect.signature(provider_type)
            if "model_name" in init_sig.parameters:
                # make sure provider needs provider in init
                if not self.model_name:
                    raise EnvironmentError(
                        "Missing model name which is required for this provider; please set the 'ALTK_MODEL_NAME' environment variable or instantiate an appropriate LLMClient."
                    )
                self._chosen_provider = provider_type(model_name=self.model_name)
            else:
                self._chosen_provider = provider_type()
                self.model_name_in_generate = True

    @classmethod
    def provider_class(cls) -> Type[Any]:
        raise NotImplementedError

    def _register_methods(self) -> None:
        if self._chosen_provider:
            self._chosen_provider._register_methods()

    def _parse_llm_response(self, raw: Any) -> Union[str, LLMResponse]:
        if not self._chosen_provider:
            raise Exception(
                "Missing provider name; please set the 'LLM_PROVIDER' environment variable or instantiate an appropriate LLMClient."
            )
        return self._chosen_provider._parse_llm_response(raw)

    def _setup_parameter_mapper(self) -> None:
        """
        Setup parameter mapping for the provider. Override in subclasses to configure
        mapping from generic GenerationArgs to provider-specific parameters.
        """
        pass

    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not self._chosen_provider:
            raise Exception(
                "Missing provider name; please set the 'LLM_PROVIDER' environment variable or instantiate an appropriate LLMClient."
            )

        if self.model_name_in_generate:
            # this is needed for providers like openai
            model_name = kwargs.get("model")
            if not model_name:
                model_name = self.model_name
                return self._chosen_provider._generate(
                    *args, model=model_name, **kwargs
                )
        return self._chosen_provider._generate(*args, **kwargs)

    async def generate_async(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not self._chosen_provider:
            raise Exception(
                "Missing provider name; please set the 'LLM_PROVIDER' environment variable or instantiate an appropriate LLMClient."
            )
        if self.model_name_in_generate:
            # this is needed for providers like openai
            model_name = kwargs.get("model")
            if not model_name:
                model_name = self.model_name
                return await self._chosen_provider._generate_async(
                    *args, model=model_name, **kwargs
                )
        return await self._chosen_provider._generate_async(*args, **kwargs)
