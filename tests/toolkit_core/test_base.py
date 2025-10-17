import asyncio
import pytest
from unittest.mock import Mock, patch
from typing import Any, Type

from altk.core.llm.base import (
    LLMClient,
    MethodConfig,
)
from altk.core.llm.types import GenerationMode


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    @classmethod
    def provider_class(cls) -> Type:
        return Mock

    def _register_methods(self) -> None:
        self.set_method_config("text", "generate_text", "prompt")
        self.set_method_config("chat", "chat.completions.create", "messages")
        self.set_method_config("text_async", "generate_text_async", "prompt")
        self.set_method_config(
            "chat_async", "chat.completions.create_async", "messages"
        )

    def _parse_llm_response(self, raw: Any) -> str:
        if hasattr(raw, "choices"):
            return raw.choices[0].message.content
        return str(raw)

    def _setup_parameter_mapper(self) -> None:
        """
        Setup parameter mapping for the provider. Override in subclasses to configure
        mapping from generic GenerationArgs to provider-specific parameters.
        """
        pass


class TestMethodConfig:
    """Test MethodConfig class."""

    def test_init(self):
        """Test MethodConfig initialization."""
        config = MethodConfig("chat.completions.create", "messages")
        assert config.path == "chat.completions.create"
        assert config.prompt_arg == "messages"

    def test_resolve_simple_path(self):
        """Test resolving a simple method path."""
        mock_client = Mock()
        mock_client.generate = Mock(return_value="test response")

        config = MethodConfig("generate", "prompt")
        method = config.resolve(mock_client)

        assert method == mock_client.generate
        assert callable(method)

    def test_resolve_nested_path(self):
        """Test resolving a nested method path."""
        mock_client = Mock()
        mock_client.chat.completions.create = Mock(return_value="test response")

        config = MethodConfig("chat.completions.create", "messages")
        method = config.resolve(mock_client)

        assert method == mock_client.chat.completions.create
        assert callable(method)

    def test_resolve_missing_attribute(self):
        """Test resolving with missing attribute raises AttributeError."""
        mock_client = Mock()
        mock_client.nonexistent = None

        config = MethodConfig("nonexistent.method", "prompt")

        with pytest.raises(AttributeError, match="Could not resolve method path"):
            config.resolve(mock_client)

    def test_resolve_non_callable(self):
        """Test resolving non-callable attribute raises TypeError."""
        mock_client = Mock()
        mock_client.not_callable = "not a function"

        config = MethodConfig("not_callable", "prompt")

        with pytest.raises(TypeError, match="is not callable"):
            config.resolve(mock_client)


class TestLLMClient:
    """Test LLMClient base class."""

    def test_init_with_client(self):
        """Test initialization with existing client."""
        mock_client = Mock()

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)
            assert llm.get_client() == mock_client

    def test_init_wrong_client_type(self):
        """Test initialization with wrong client type."""
        mock_client = Mock()

        with patch.object(MockLLMClient, "provider_class", return_value=str):
            with pytest.raises(TypeError, match="Expected client of type"):
                MockLLMClient(client=mock_client)

    def test_init_without_client(self):
        """Test initialization without client."""
        with patch.object(MockLLMClient, "provider_class", return_value=Mock):
            llm = MockLLMClient()
            assert llm.get_client() is not None

    def test_init_with_client_needs_init(self):
        """Test initialization with client_needs_init=True."""
        mock_provider = Mock()
        mock_provider.return_value = Mock()

        with patch.object(MockLLMClient, "provider_class", return_value=mock_provider):
            MockLLMClient(client_needs_init=True, test_param="value")

            # Verify client was initialized with correct kwargs
            mock_provider.assert_called_once()

    def test_init_client_needs_init_failure(self):
        """Test initialization failure with client_needs_init=True."""
        mock_provider = Mock()
        mock_provider.side_effect = Exception("Init failed")
        mock_provider.__name__ = "MockProvider"  # Add __name__ attribute

        with patch.object(MockLLMClient, "provider_class", return_value=mock_provider):
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                MockLLMClient(client_needs_init=True)

    def test_init_with_hooks(self):
        """Test initialization with hooks."""
        mock_hook = Mock()

        with patch.object(MockLLMClient, "provider_class", return_value=Mock):
            llm = MockLLMClient(hooks=[mock_hook])
            assert mock_hook in llm._hooks

    def test_set_method_config(self):
        """Test setting method configuration."""
        with patch.object(MockLLMClient, "provider_class", return_value=Mock):
            llm = MockLLMClient()
            llm.set_method_config("test_method", "test.path", "test_arg")

            config = llm.get_method_config("test_method")
            assert config.path == "test.path"
            assert config.prompt_arg == "test_arg"

    def test_get_method_config_not_found(self):
        """Test getting non-existent method config."""
        with patch.object(MockLLMClient, "provider_class", return_value=Mock):
            llm = MockLLMClient()

            with pytest.raises(
                KeyError, match="No method config registered under 'nonexistent'"
            ):
                llm.get_method_config("nonexistent")

    def test_emit_hook_success(self):
        """Test successful hook emission."""
        mock_hook = Mock()

        with patch.object(MockLLMClient, "provider_class", return_value=Mock):
            llm = MockLLMClient(hooks=[mock_hook])
            llm._emit("test_event", {"key": "value"})

            mock_hook.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_hook_failure(self):
        """Test hook emission with failing hook."""
        mock_hook1 = Mock()
        mock_hook2 = Mock(side_effect=Exception("Hook failed"))
        mock_hook3 = Mock()

        with patch.object(MockLLMClient, "provider_class", return_value=Mock):
            llm = MockLLMClient(hooks=[mock_hook1, mock_hook2, mock_hook3])
            llm._emit("test_event", {"key": "value"})

            # All hooks should be called despite the failure
            mock_hook1.assert_called_once()
            mock_hook2.assert_called_once()
            mock_hook3.assert_called_once()

    def test_generate_text_mode(self):
        """Test generate with text mode."""
        mock_client = Mock()
        mock_client.generate_text = Mock(return_value="Generated text")

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)

            result = llm.generate("Test prompt", mode=GenerationMode.TEXT)

            assert result == "Generated text"
            mock_client.generate_text.assert_called_once_with(prompt="Test prompt")

    def test_generate_chat_mode_with_string(self):
        """Test generate with chat mode and string prompt."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Chat response"
        mock_client.chat.completions.create = Mock(return_value=mock_response)

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)

            result = llm.generate("Test prompt", mode=GenerationMode.CHAT)

            assert result == "Chat response"
            mock_client.chat.completions.create.assert_called_once_with(
                messages=[{"role": "user", "content": "Test prompt"}]
            )

    def test_generate_chat_mode_with_messages(self):
        """Test generate with chat mode and message list."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Chat response"
        mock_client.chat.completions.create = Mock(return_value=mock_response)

        messages = [{"role": "user", "content": "Test prompt"}]

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)

            result = llm.generate(messages, mode=GenerationMode.CHAT)

            assert result == "Chat response"
            mock_client.chat.completions.create.assert_called_once_with(
                messages=messages
            )

    def test_generate_with_kwargs(self):
        """Test generate with additional kwargs."""
        mock_client = Mock()
        mock_client.generate_text = Mock(return_value="Generated text")

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)

            result = llm.generate(
                "Test prompt", mode=GenerationMode.TEXT, temperature=0.5
            )

            assert result == "Generated text"
            mock_client.generate_text.assert_called_once_with(
                prompt="Test prompt", temperature=0.5
            )

    def test_generate_with_hook_emission(self):
        """Test generate with hook emission."""
        mock_hook = Mock()
        mock_client = Mock()
        mock_client.generate_text = Mock(return_value="Generated text")

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client, hooks=[mock_hook])

            result = llm.generate("Test prompt", mode=GenerationMode.TEXT)

            assert result == "Generated text"
            assert mock_hook.call_count == 2  # before and after
            mock_hook.assert_any_call(
                "before_generate", {"mode": "text", "args": {"prompt": "Test prompt"}}
            )
            mock_hook.assert_any_call(
                "after_generate", {"mode": "text", "response": "Generated text"}
            )

    def test_generate_with_exception(self):
        """Test generate with exception handling."""
        mock_hook = Mock()
        mock_client = Mock()
        mock_client.generate_text = Mock(side_effect=Exception("Generation failed"))

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client, hooks=[mock_hook])

            with pytest.raises(Exception, match="Generation failed"):
                llm.generate("Test prompt", mode=GenerationMode.TEXT)

            mock_hook.assert_any_call(
                "error", {"phase": "generate", "error": "Generation failed"}
            )

    def test_generate_mode_not_found(self):
        """Test generate with non-existent mode."""
        mock_client = Mock()

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)

            with pytest.raises(
                KeyError, match="No method config registered under 'nonexistent'"
            ):
                llm.generate("Test prompt", mode="nonexistent")

    @pytest.mark.asyncio
    async def test_generate_async_with_async_method(self):
        """Test async generate with async method."""
        mock_client = Mock()
        mock_client.generate_text_async = Mock(return_value=asyncio.Future())
        mock_client.generate_text_async.return_value.set_result("Async generated text")

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)

            result = await llm.generate_async(
                "Test prompt", mode=GenerationMode.TEXT_ASYNC
            )

            assert result == "Async generated text"

    @pytest.mark.asyncio
    async def test_generate_async_chat_mode_with_string(self):
        """Test async generate with chat mode and string prompt."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Async chat response"
        mock_client.chat.completions.create_async = Mock(return_value=asyncio.Future())
        mock_client.chat.completions.create_async.return_value.set_result(mock_response)

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)

            result = await llm.generate_async(
                "Test prompt", mode=GenerationMode.CHAT_ASYNC
            )

            assert result == "Async chat response"

    @pytest.mark.asyncio
    async def test_generate_async_with_hook_emission(self):
        """Test async generate with hook emission."""
        mock_hook = Mock()
        mock_client = Mock()
        mock_client.generate_text_async = Mock(return_value=asyncio.Future())
        mock_client.generate_text_async.return_value.set_result("Async generated text")

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client, hooks=[mock_hook])

            result = await llm.generate_async(
                "Test prompt", mode=GenerationMode.TEXT_ASYNC
            )

            assert result == "Async generated text"
            assert mock_hook.call_count == 2  # before and after
            mock_hook.assert_any_call(
                "before_generate_async",
                {"mode": "text_async", "args": {"prompt": "Test prompt"}},
            )
            mock_hook.assert_any_call(
                "after_generate_async",
                {"mode": "text_async", "response": "Async generated text"},
            )

    @pytest.mark.asyncio
    async def test_generate_async_with_exception(self):
        """Test async generate with exception handling."""
        mock_hook = Mock()
        mock_client = Mock()
        mock_client.generate_text_async = Mock(
            side_effect=Exception("Async generation failed")
        )

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client, hooks=[mock_hook])

            with pytest.raises(Exception, match="Async generation failed"):
                await llm.generate_async("Test prompt", mode=GenerationMode.TEXT_ASYNC)

            mock_hook.assert_any_call(
                "error", {"phase": "generate_async", "error": "Async generation failed"}
            )

    @pytest.mark.asyncio
    async def test_generate_async_fallback_to_sync(self):
        """Test async generate fallback to sync method."""
        mock_client = Mock()
        mock_client.generate_text = Mock(return_value="Sync generated text")

        with patch.object(
            MockLLMClient, "provider_class", return_value=type(mock_client)
        ):
            llm = MockLLMClient(client=mock_client)
            # Remove async method config to force fallback
            llm._method_configs.pop("text_async", None)

            result = await llm.generate_async(
                "Test prompt", mode=GenerationMode.TEXT_ASYNC
            )

            assert result == "Sync generated text"
            mock_client.generate_text.assert_called_once_with(prompt="Test prompt")
