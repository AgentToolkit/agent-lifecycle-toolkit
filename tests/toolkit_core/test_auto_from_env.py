import pytest

from altk.core.llm import get_llm


class TestAutoFromEnvLLMProvider:
    """Test selection of LLM providers from environment variables.
    Note that this test doesn't actually run any LLMs."""

    def test_missing_both_env_var(self, monkeypatch):
        monkeypatch.delenv("ALTK_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ALTK_MODEL_NAME", raising=False)
        client = get_llm("auto_from_env")()
        assert client._chosen_provider is None

        monkeypatch.setenv("ALTK_LLM_PROVIDER", "thisproviderdoesn'texist")
        with pytest.raises(ValueError):
            client = get_llm("auto_from_env")()

    def test_selecting_default_litellm(self, monkeypatch):
        monkeypatch.delenv("ALTK_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ALTK_MODEL_NAME", "anthropic/claude-sonnet-4-20250514")
        client = get_llm("auto_from_env")()
        target_provider = get_llm("litellm")
        assert isinstance(client._chosen_provider, target_provider)

    def test_selecting_watsonx(self, monkeypatch):
        monkeypatch.setenv("ALTK_LLM_PROVIDER", "watsonx")
        monkeypatch.delenv("ALTK_MODEL_NAME", raising=False)
        with pytest.raises(EnvironmentError):
            client = get_llm("auto_from_env")()

        monkeypatch.setenv("ALTK_MODEL_NAME", "meta-llama/llama-3-3-70b-instruct")
        client = get_llm("auto_from_env")()
        target_provider = get_llm("watsonx")
        assert isinstance(client._chosen_provider, target_provider)

    def test_selecting_litellm_watsonx(self, monkeypatch):
        monkeypatch.setenv("ALTK_LLM_PROVIDER", "litellm.watsonx")
        monkeypatch.delenv("ALTK_MODEL_NAME", raising=False)
        with pytest.raises(EnvironmentError):
            client = get_llm("auto_from_env")()

        monkeypatch.setenv("ALTK_MODEL_NAME", "meta-llama/llama-3-3-70b-instruct")
        client = get_llm("auto_from_env")()
        target_provider = get_llm("litellm.watsonx")
        assert isinstance(client._chosen_provider, target_provider)

    def test_selecting_litellm_ollama(self, monkeypatch):
        monkeypatch.setenv("ALTK_LLM_PROVIDER", "litellm.ollama")
        monkeypatch.delenv("ALTK_MODEL_NAME", raising=False)
        with pytest.raises(EnvironmentError):
            client = get_llm("auto_from_env")()

        monkeypatch.setenv("ALTK_MODEL_NAME", "llama3.2:1b")
        client = get_llm("auto_from_env")()
        target_provider = get_llm("litellm.ollama")
        assert isinstance(client._chosen_provider, target_provider)

    def test_selecting_openai(self, monkeypatch):
        monkeypatch.setenv("ALTK_LLM_PROVIDER", "openai.sync")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            client = get_llm("auto_from_env")()

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        client = get_llm("auto_from_env")()
        target_provider = get_llm("openai.sync")
        assert isinstance(client._chosen_provider, target_provider)

    def test_selecting_azure_openai(self, monkeypatch):
        monkeypatch.setenv("ALTK_LLM_PROVIDER", "azure_openai.sync")
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        with pytest.raises(RuntimeError):
            client = get_llm("auto_from_env")()

        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
        monkeypatch.setenv("OPENAI_API_VERSION", "test")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "test")
        client = get_llm("auto_from_env")()
        target_provider = get_llm("azure_openai.sync")
        assert isinstance(client._chosen_provider, target_provider)
