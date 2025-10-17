from altk.core.llm import get_llm
from altk.core.toolkit import ComponentConfig


class TestDirectLLMComponentConfig:
    """Test directly instantiating LiteLLM to ComponentConfig's llm_client
    Note that this test doesn't actually run any LLMs."""

    def test_default_auto_env(self):
        config = ComponentConfig()
        target_provider = get_llm("auto_from_env")
        assert isinstance(config.llm_client, target_provider)

    def test_direct_litellm(self):
        config = ComponentConfig(llm_client="provider/modelname")
        target_provider = get_llm("litellm")
        assert isinstance(config.llm_client, target_provider)
