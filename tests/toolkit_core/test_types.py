from altk.core.llm.types import GenerationMode, LLMResponse


class TestGenerationMode:
    """Test GenerationMode enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert GenerationMode.TEXT.value == "text"
        assert GenerationMode.CHAT.value == "chat"
        assert GenerationMode.TEXT_ASYNC.value == "text_async"
        assert GenerationMode.CHAT_ASYNC.value == "chat_async"

    def test_enum_members(self):
        """Test all enum members exist."""
        assert hasattr(GenerationMode, "TEXT")
        assert hasattr(GenerationMode, "CHAT")
        assert hasattr(GenerationMode, "TEXT_ASYNC")
        assert hasattr(GenerationMode, "CHAT_ASYNC")

    def test_enum_comparison(self):
        """Test enum comparison works correctly."""
        assert GenerationMode.TEXT == GenerationMode.TEXT
        assert GenerationMode.TEXT != GenerationMode.CHAT
        assert GenerationMode.CHAT_ASYNC == GenerationMode.CHAT_ASYNC

    def test_enum_string_conversion(self):
        """Test enum string conversion."""
        assert str(GenerationMode.TEXT) == "GenerationMode.TEXT"
        assert str(GenerationMode.CHAT) == "GenerationMode.CHAT"


class TestLLMResponse:
    """Test LLMResponse class."""

    def test_init_with_content_only(self):
        """Test initialization with content only."""
        response = LLMResponse("Test content")
        assert response.content == "Test content"
        assert response.tool_calls == []

    def test_init_with_tool_calls(self):
        """Test initialization with tool calls."""
        tool_calls = [
            {"name": "get_weather", "args": {"location": "NYC"}},
            {"name": "calculate", "args": {"expression": "2+2"}},
        ]
        response = LLMResponse("Test content", tool_calls=tool_calls)
        assert response.content == "Test content"
        assert response.tool_calls == tool_calls

    def test_init_with_none_tool_calls(self):
        """Test initialization with None tool_calls."""
        response = LLMResponse("Test content", tool_calls=None)
        assert response.content == "Test content"
        assert response.tool_calls == []

    def test_str_representation(self):
        """Test string representation returns content."""
        response = LLMResponse("Test content")
        assert str(response) == "Test content"

    def test_str_with_tool_calls(self):
        """Test string representation with tool calls."""
        tool_calls = [{"name": "test", "args": {}}]
        response = LLMResponse("Test content", tool_calls=tool_calls)
        assert str(response) == "Test content"

    def test_repr_representation(self):
        """Test repr representation."""
        response = LLMResponse("Test content")
        expected = "LLMResponse(content='Test content', tool_calls=[])"
        assert repr(response) == expected

    def test_repr_with_tool_calls(self):
        """Test repr representation with tool calls."""
        tool_calls = [{"name": "test", "args": {}}]
        response = LLMResponse("Test content", tool_calls=tool_calls)
        expected = "LLMResponse(content='Test content', tool_calls=[{'name': 'test', 'args': {}}])"
        assert repr(response) == expected

    def test_repr_with_empty_content(self):
        """Test repr with empty content."""
        response = LLMResponse("")
        expected = "LLMResponse(content='', tool_calls=[])"
        assert repr(response) == expected

    def test_response_attributes_are_accessible(self):
        """Test that response attributes are accessible."""
        tool_calls = [{"name": "test", "args": {"param": "value"}}]
        response = LLMResponse("Test content", tool_calls=tool_calls)

        # Test content access
        assert response.content == "Test content"

        # Test tool_calls access
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "test"
        assert response.tool_calls[0]["args"]["param"] == "value"

    def test_response_immutability(self):
        """Test that response content is modifiable (not immutable)."""
        response = LLMResponse("Original content")

        # Content should be modifiable
        response.content = "Modified content"
        assert response.content == "Modified content"

        # Tool calls should be modifiable
        response.tool_calls.append({"name": "new_tool", "args": {}})
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "new_tool"

    def test_response_with_complex_tool_calls(self):
        """Test response with complex tool call structures."""
        complex_tool_calls = [
            {
                "name": "complex_tool",
                "args": {
                    "param1": "value1",
                    "param2": 42,
                    "param3": True,
                    "param4": [1, 2, 3],
                    "param5": {"nested": "value"},
                },
            }
        ]

        response = LLMResponse("Complex content", tool_calls=complex_tool_calls)

        assert response.content == "Complex content"
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]
        assert tool_call["name"] == "complex_tool"
        assert tool_call["args"]["param1"] == "value1"
        assert tool_call["args"]["param2"] == 42
        assert tool_call["args"]["param3"] is True
        assert tool_call["args"]["param4"] == [1, 2, 3]
        assert tool_call["args"]["param5"]["nested"] == "value"
