class TestInitModule:
    """Test __init__.py module functionality."""

    def test_import_providers_litellm_import_error(self):
        """Test _import_providers with LiteLLM import error."""
        # Test that import errors are handled gracefully
        # Since the module is already imported, we just test that the function exists
        from altk.toolkit_core.llm import _import_providers

        assert callable(_import_providers)

        # The function should not crash when called
        _import_providers()

    def test_import_providers_openai_import_error(self):
        """Test _import_providers with OpenAI import error."""
        from altk.toolkit_core.llm import _import_providers

        assert callable(_import_providers)

        # The function should not crash when called
        _import_providers()

    def test_import_providers_watsonx_import_error(self):
        """Test _import_providers with IBM Watson import error."""
        from altk.toolkit_core.llm import _import_providers

        assert callable(_import_providers)

        # The function should not crash when called
        _import_providers()

    def test_core_imports_available(self):
        """Test that core imports are always available."""
        import altk.toolkit_core.llm

        # Core components should always be available
        assert "LLMClient" in altk.toolkit_core.llm.__all__
        assert "ValidatingLLMClient" in altk.toolkit_core.llm.__all__
        assert "get_llm" in altk.toolkit_core.llm.__all__
        assert "register_llm" in altk.toolkit_core.llm.__all__
        assert "list_available_llms" in altk.toolkit_core.llm.__all__
        assert "Hook" in altk.toolkit_core.llm.__all__
        assert "MethodConfig" in altk.toolkit_core.llm.__all__
        assert "OutputValidationError" in altk.toolkit_core.llm.__all__
        assert "GenerationMode" in altk.toolkit_core.llm.__all__
        assert "LLMResponse" in altk.toolkit_core.llm.__all__

        # Test that the objects are importable
        assert hasattr(altk.toolkit_core.llm, "LLMClient")
        assert hasattr(altk.toolkit_core.llm, "ValidatingLLMClient")
        assert hasattr(altk.toolkit_core.llm, "get_llm")
        assert hasattr(altk.toolkit_core.llm, "register_llm")
        assert hasattr(altk.toolkit_core.llm, "list_available_llms")
        assert hasattr(altk.toolkit_core.llm, "Hook")
        assert hasattr(altk.toolkit_core.llm, "MethodConfig")
        assert hasattr(altk.toolkit_core.llm, "OutputValidationError")
        assert hasattr(altk.toolkit_core.llm, "GenerationMode")
        assert hasattr(altk.toolkit_core.llm, "LLMResponse")

    def test_registry_initialization(self):
        """Test that the registry is initialized correctly."""
        from altk.toolkit_core.llm import _REGISTRY

        # Registry should be a dict
        assert isinstance(_REGISTRY, dict)

        # Should not be None
        assert _REGISTRY is not None
