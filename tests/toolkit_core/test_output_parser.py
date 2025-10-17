import json
import pytest
from unittest.mock import Mock, patch
from typing import Optional
from pydantic import BaseModel

from altk.core.llm.output_parser import (
    json_schema_to_pydantic_model,
    OutputValidationError,
    ValidatingLLMClient,
)
from altk.core.llm.base import BaseLLMClient


class ValidationTestModel(BaseModel):
    """Test Pydantic model for testing."""

    name: str
    age: int
    email: Optional[str] = None


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing."""

    def __init__(self, **kwargs):
        self.mock_client = Mock()
        super().__init__(client=self.mock_client, **kwargs)

    @classmethod
    def provider_class(cls):
        return Mock

    def _register_methods(self):
        self.set_method_config("text", "generate", "prompt")
        self.set_method_config("chat", "chat", "messages")

    def _parse_llm_response(self, raw):
        return str(raw)


class MockValidatingLLMClient(ValidatingLLMClient):
    """Mock validating LLM client for testing."""

    def __init__(self, **kwargs):
        self.mock_client = Mock()
        super().__init__(client=self.mock_client, **kwargs)

    @classmethod
    def provider_class(cls):
        return Mock

    def _register_methods(self):
        self.set_method_config("text", "generate", "prompt")
        self.set_method_config("chat", "chat", "messages")

    def _parse_llm_response(self, raw):
        return str(raw)

    def _setup_parameter_mapper(self) -> None:
        """
        Setup parameter mapping for the provider. Override in subclasses to configure
        mapping from generic GenerationArgs to provider-specific parameters.
        """
        pass


class TestJSONSchemaUtilities:
    """Test JSON schema utility functions."""

    def test_json_schema_to_pydantic_model_basic(self):
        """Test basic JSON schema conversion."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        model = json_schema_to_pydantic_model(schema, "ValidationTestModel")

        assert model.__name__ == "ValidationTestModel"
        assert "name" in model.model_fields
        assert "age" in model.model_fields

        # Test model validation
        instance = model(name="John", age=30)
        assert instance.name == "John"
        assert instance.age == 30

    def test_json_schema_to_pydantic_model_optional_fields(self):
        """Test JSON schema conversion with optional fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name"],
        }

        model = json_schema_to_pydantic_model(schema, "ValidationTestModel")

        # Should work with just required field
        instance = model(name="John")
        assert instance.name == "John"

        # Should work with optional field
        instance = model(name="John", email="john@example.com")
        assert instance.email == "john@example.com"

    def test_json_schema_to_pydantic_model_complex_types(self):
        """Test JSON schema conversion with complex types."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "tags": {"type": "array"},
                "metadata": {"type": "object"},
                "nullable": {"type": "null"},
            },
            "required": ["name"],
        }

        model = json_schema_to_pydantic_model(schema, "ComplexModel")

        instance = model(
            name="Test",
            score=95.5,
            active=True,
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
            nullable=None,
        )

        assert instance.name == "Test"
        assert instance.score == 95.5
        assert instance.active is True
        assert instance.tags == ["tag1", "tag2"]
        assert instance.metadata == {"key": "value"}
        assert instance.nullable is None

    def test_json_schema_to_pydantic_model_union_types(self):
        """Test JSON schema conversion with union types."""
        schema = {
            "type": "object",
            "properties": {"value": {"type": ["string", "integer"]}},
            "required": ["value"],
        }

        model = json_schema_to_pydantic_model(schema, "UnionModel")

        # Should accept string
        instance1 = model(value="test")
        assert instance1.value == "test"

        # Should accept integer
        instance2 = model(value=42)
        assert instance2.value == 42

    def test_json_schema_to_pydantic_model_nullable_union(self):
        """Test JSON schema conversion with nullable union types."""
        schema = {
            "type": "object",
            "properties": {"value": {"type": ["string", "null"]}},
            "required": ["value"],
        }

        model = json_schema_to_pydantic_model(schema, "NullableModel")

        # Should accept string
        instance1 = model(value="test")
        assert instance1.value == "test"

        # Should accept null
        instance2 = model(value=None)
        assert instance2.value is None

    def test_json_schema_to_pydantic_model_unknown_type(self):
        """Test JSON schema conversion with unknown types."""
        schema = {
            "type": "object",
            "properties": {"unknown": {"type": "unknown_type"}},
            "required": ["unknown"],
        }

        model = json_schema_to_pydantic_model(schema, "UnknownModel")

        # Should use Any type for unknown types
        instance = model(unknown="anything")
        assert instance.unknown == "anything"

    def test_json_schema_to_pydantic_model_with_descriptions(self):
        """Test JSON schema conversion with field descriptions."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Person's age"},
            },
            "required": ["name"],
        }

        model = json_schema_to_pydantic_model(schema, "DescribedModel")

        # Should work with descriptions
        instance = model(name="John", age=30)
        assert instance.name == "John"
        assert instance.age == 30


class TestOutputValidationError:
    """Test OutputValidationError class."""

    def test_init_with_message(self):
        """Test initialization with message only."""
        error = OutputValidationError("Validation failed")
        assert str(error) == "Validation failed"

    def test_inheritance(self):
        """Test that OutputValidationError inherits from Exception."""
        error = OutputValidationError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, OutputValidationError)


class TestValidatingLLMClient:
    """Test ValidatingLLMClient class."""

    def test_make_instruction_json_schema(self):
        """Test _make_instruction with JSON schema."""
        client = MockValidatingLLMClient()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        instruction = client._make_instruction(schema)

        assert (
            "JSON object conforming exactly to the following JSON Schema" in instruction
        )
        assert json.dumps(schema, indent=2) in instruction

    def test_make_instruction_pydantic_model(self):
        """Test _make_instruction with Pydantic model."""
        client = MockValidatingLLMClient()

        instruction = client._make_instruction(ValidationTestModel)

        assert (
            "JSON object conforming exactly to this Pydantic model schema"
            in instruction
        )
        assert "name" in instruction  # Should contain model schema

    def test_make_instruction_builtin_types(self):
        """Test _make_instruction with built-in types."""
        client = MockValidatingLLMClient()

        for type_class in [int, float, str, bool, list, dict]:
            instruction = client._make_instruction(type_class)
            assert f"value of type `{type_class.__name__}`" in instruction

    def test_make_instruction_unsupported_type(self):
        """Test _make_instruction with unsupported type."""
        client = MockValidatingLLMClient()

        with pytest.raises(TypeError, match="Unsupported schema type"):
            client._make_instruction(set)

    def test_extract_json_code_fence(self):
        """Test _extract_json with code fence."""
        raw = """
        Here is the JSON:
        ```json
        {"name": "John"}
        ```
        """

        result = MockValidatingLLMClient._extract_json(raw)
        assert result.strip() == '{"name": "John"}'

    def test_extract_json_code_fence_no_language(self):
        """Test _extract_json with code fence without language."""
        raw = """
        Here is the JSON:
        ```
        {"name": "John"}
        ```
        """

        result = MockValidatingLLMClient._extract_json(raw)
        assert result.strip() == '{"name": "John"}'

    def test_extract_json_inline_braces(self):
        """Test _extract_json with inline braces."""
        raw = 'The result is {"name": "John"} and that\'s it.'

        result = MockValidatingLLMClient._extract_json(raw)
        assert result == '{"name": "John"}'

    def test_extract_json_no_json(self):
        """Test _extract_json with no JSON."""
        raw = "No JSON here"

        result = MockValidatingLLMClient._extract_json(raw)
        assert result == raw

    def test_clean_raw(self):
        """Test _clean_raw method."""
        client = MockValidatingLLMClient()

        raw = """

        ```json
        {"name": "John"}
        ```

        """

        result = client._clean_raw(raw)
        assert result == '{"name": "John"}'

    def test_validate_json_schema_success(self):
        """Test _validate with JSON schema success."""
        client = MockValidatingLLMClient()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with patch("altk.core.llm.output_parser.jsonschema") as mock_jsonschema:
            mock_jsonschema.validate.return_value = None

            result = client._validate('{"name": "John"}', schema)

            assert result == {"name": "John"}
            mock_jsonschema.validate.assert_called_once()

    def test_validate_json_schema_missing_library(self):
        """Test _validate with missing jsonschema library."""
        client = MockValidatingLLMClient()
        schema = {"type": "object"}

        with patch("altk.core.llm.output_parser.jsonschema", None):
            with pytest.raises(ImportError, match="jsonschema is required"):
                client._validate('{"name": "John"}', schema)

    def test_validate_json_schema_validation_error(self):
        """Test _validate with JSON schema validation error."""
        client = MockValidatingLLMClient()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with patch("altk.core.llm.output_parser.jsonschema") as mock_jsonschema:
            # Create a proper ValidationError mock
            class MockValidationError(Exception):
                def __init__(self, message):
                    self.message = message
                    super().__init__(message)

            mock_jsonschema.ValidationError = MockValidationError
            mock_jsonschema.validate.side_effect = MockValidationError("Invalid type")

            with pytest.raises(
                OutputValidationError, match="JSON Schema validation error"
            ):
                client._validate('{"name": 123}', schema)

    def test_validate_pydantic_model_success(self):
        """Test _validate with Pydantic model success."""
        client = MockValidatingLLMClient()

        result = client._validate('{"name": "John", "age": 30}', ValidationTestModel)

        assert isinstance(result, ValidationTestModel)
        assert result.name == "John"
        assert result.age == 30

    def test_validate_pydantic_model_error(self):
        """Test _validate with Pydantic model validation error."""
        client = MockValidatingLLMClient()

        with pytest.raises(OutputValidationError, match="Pydantic validation error"):
            client._validate('{"name": "John"}', ValidationTestModel)  # Missing age

    def test_validate_builtin_types_success(self):
        """Test _validate with built-in types success."""
        client = MockValidatingLLMClient()

        # Test int
        result = client._validate("42", int)
        assert result == 42

        # Test str
        result = client._validate('"hello"', str)
        assert result == "hello"

        # Test bool
        result = client._validate("true", bool)
        assert result is True

        # Test list
        result = client._validate("[1, 2, 3]", list)
        assert result == [1, 2, 3]

        # Test dict
        result = client._validate('{"key": "value"}', dict)
        assert result == {"key": "value"}

    def test_validate_builtin_types_error(self):
        """Test _validate with built-in types error."""
        client = MockValidatingLLMClient()

        with pytest.raises(OutputValidationError, match="Type mismatch"):
            client._validate('"hello"', int)

    def test_validate_unsupported_schema(self):
        """Test _validate with unsupported schema type."""
        client = MockValidatingLLMClient()

        with pytest.raises(TypeError, match="Unsupported schema type"):
            client._validate('{"key": "value"}', set)

    def test_validate_json_decode_error_fallback(self):
        """Test _validate with JSON decode error fallback."""
        client = MockValidatingLLMClient()

        # Test with unicode escape fallback
        with patch("json.loads") as mock_loads:
            mock_loads.side_effect = [
                json.JSONDecodeError("Invalid JSON", "", 0),
                {"name": "John"},
            ]

            client._validate("invalid json", {"type": "object"})

            assert mock_loads.call_count == 2

    def test_validate_string_schema(self):
        """Test _validate with string schema."""
        client = MockValidatingLLMClient()

        result = client._validate("raw text", str)
        assert result == "raw text"

    def test_inject_system_string_prompt(self):
        """Test _inject_system with string prompt."""
        client = MockValidatingLLMClient()

        result = client._inject_system("User prompt", "System instruction")

        assert result == "System instruction\n\nUser prompt"

    def test_inject_system_chat_with_system(self):
        """Test _inject_system with chat messages with existing system."""
        client = MockValidatingLLMClient()

        messages = [
            {"role": "system", "content": "Existing system"},
            {"role": "user", "content": "User message"},
        ]

        result = client._inject_system(messages, "New instruction")

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Existing system\n\nNew instruction"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "User message"

    def test_inject_system_chat_without_system(self):
        """Test _inject_system with chat messages without system."""
        client = MockValidatingLLMClient()

        messages = [{"role": "user", "content": "User message"}]

        result = client._inject_system(messages, "System instruction")

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System instruction"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "User message"

    def test_generate_success_first_try(self):
        """Test generate with success on first try."""
        client = MockValidatingLLMClient()

        with patch.object(
            BaseLLMClient, "_generate", return_value='{"name": "John", "age": 30}'
        ):
            result = client.generate("Generate person", schema=ValidationTestModel)

            assert isinstance(result, ValidationTestModel)
            assert result.name == "John"
            assert result.age == 30

    def test_generate_with_retries(self):
        """Test generate with retries."""
        client = MockValidatingLLMClient()

        with patch.object(BaseLLMClient, "_generate") as mock_generate:
            mock_generate.side_effect = ["Invalid JSON", '{"name": "John", "age": 30}']

            result = client.generate(
                "Generate person", schema=ValidationTestModel, retries=2
            )

            assert isinstance(result, ValidationTestModel)
            assert result.name == "John"
            assert result.age == 30
            assert mock_generate.call_count == 2

    def test_generate_max_retries_exceeded(self):
        """Test generate with max retries exceeded."""
        client = MockValidatingLLMClient()

        with patch.object(BaseLLMClient, "_generate", return_value="Invalid JSON"):
            with pytest.raises(OutputValidationError, match="Failed after 2 attempts"):
                client.generate(
                    "Generate person", schema=ValidationTestModel, retries=2
                )

    def test_generate_with_schema_in_system_prompt(self):
        """Test generate with schema in system prompt."""
        client = MockValidatingLLMClient()

        with patch.object(
            BaseLLMClient, "_generate", return_value='{"name": "John", "age": 30}'
        ):
            result = client.generate(
                "Generate person",
                schema=ValidationTestModel,
                include_schema_in_system_prompt=True,
            )

            assert isinstance(result, ValidationTestModel)

    def test_generate_with_schema_field(self):
        """Test generate with schema_field parameter."""
        client = MockValidatingLLMClient()

        with patch.object(
            BaseLLMClient, "_generate", return_value='{"name": "John", "age": 30}'
        ):
            result = client.generate(
                "Generate person",
                schema=ValidationTestModel,
                schema_field="response_format",
            )

            assert isinstance(result, ValidationTestModel)

    def test_generate_non_string_response(self):
        """Test generate with non-string response."""
        client = MockValidatingLLMClient()

        mock_response = ValidationTestModel(name="John", age=30)

        with patch.object(BaseLLMClient, "_generate", return_value=mock_response):
            result = client.generate("Generate person", schema=ValidationTestModel)

            assert result == mock_response

    @pytest.mark.asyncio
    async def test_generate_async_success(self):
        """Test generate_async with success."""
        client = MockValidatingLLMClient()

        with patch.object(
            BaseLLMClient, "_generate_async", return_value='{"name": "John", "age": 30}'
        ):
            result = await client.generate_async(
                "Generate person", schema=ValidationTestModel
            )

            assert isinstance(result, ValidationTestModel)
            assert result.name == "John"
            assert result.age == 30

    @pytest.mark.asyncio
    async def test_generate_async_with_retries(self):
        """Test generate_async with retries."""
        client = MockValidatingLLMClient()

        with patch.object(BaseLLMClient, "_generate_async") as mock_generate:
            mock_generate.side_effect = ["Invalid JSON", '{"name": "John", "age": 30}']

            result = await client.generate_async(
                "Generate person", schema=ValidationTestModel, retries=2
            )

            assert isinstance(result, ValidationTestModel)
            assert result.name == "John"
            assert result.age == 30
            assert mock_generate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_async_max_retries_exceeded(self):
        """Test generate_async with max retries exceeded."""
        client = MockValidatingLLMClient()

        with patch.object(
            BaseLLMClient, "_generate_async", return_value="Invalid JSON"
        ):
            with pytest.raises(OutputValidationError, match="Failed after 2 attempts"):
                await client.generate_async(
                    "Generate person", schema=ValidationTestModel, retries=2
                )

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_generate_async_non_string_response(self):
        """Test async generate with non-string response."""
        client = MockValidatingLLMClient()

        # Mock the parent generate method to return a non-string response
        with patch.object(BaseLLMClient, "_generate_async") as mock_generate:
            mock_generate.return_value = {"response": "test"}

            result = await client.generate_async(
                "test prompt",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                retries=1,
            )

            assert result == {"response": "test"}


class TestEdgeCases:
    """Test edge cases and missing coverage."""

    def test_jsonschema_import_error(self):
        """Test handling when jsonschema is not available."""
        # Mock jsonschema as None to simulate import error
        with patch("altk.core.llm.output_parser.jsonschema", None):
            client = MockValidatingLLMClient()
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}

            with pytest.raises(ImportError, match="jsonschema"):
                client._validate('{"name": "test"}', schema)

    def test_unicode_escape_fallback(self):
        """Test unicode escape fallback in JSON parsing."""
        client = MockValidatingLLMClient()

        # Create a string that will fail regular JSON parsing but work with unicode escape
        bad_json = '{"name": "test\\u0000"}'  # null character that may cause issues

        with patch("json.loads") as mock_loads:
            # First call fails, second call (unicode escape) succeeds
            mock_loads.side_effect = [
                json.JSONDecodeError("msg", "doc", 0),
                {"name": "test"},
            ]

            result = client._validate(bad_json, {"type": "object"})
            assert result == {"name": "test"}

    def test_unicode_escape_fallback_fails(self):
        """Test when unicode escape fallback also fails."""
        client = MockValidatingLLMClient()

        with patch("json.loads") as mock_loads:
            # Both calls fail
            mock_loads.side_effect = [
                json.JSONDecodeError("msg", "doc", 0),
                Exception("Unicode escape failed"),
            ]

            with pytest.raises(OutputValidationError):
                client._validate("invalid json", {"type": "object"})

    def test_json_schema_to_pydantic_model_unknown_type(self):
        """Test json_schema_to_pydantic_model with unknown type."""
        schema = {
            "type": "object",
            "properties": {"unknown_field": {"type": "unknown_type"}},
            "required": ["unknown_field"],
        }

        # Should handle unknown type by using Any
        model = json_schema_to_pydantic_model(schema, "UnknownTypeModel")

        # The model should be created successfully
        assert model.__name__ == "UnknownTypeModel"
        assert "unknown_field" in model.model_fields

        # Should accept any value for unknown type
        instance = model(unknown_field="any_value")
        assert instance.unknown_field == "any_value"

    def test_generate_with_schema_field_dict_conversion(self):
        """Test generate with schema_field containing dict that gets converted."""
        client = MockValidatingLLMClient()

        with patch.object(BaseLLMClient, "_generate") as mock_generate:
            mock_generate.return_value = '{"name": "test"}'

            result = client.generate(
                "test prompt",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                schema_field="response_format",
                retries=1,
            )

            assert result == {"name": "test"}

    @pytest.mark.asyncio
    async def test_generate_async_with_schema_field_dict_conversion(self):
        """Test async generate with schema_field containing dict that gets converted."""
        client = MockValidatingLLMClient()

        with patch.object(BaseLLMClient, "_generate_async") as mock_generate:
            mock_generate.return_value = '{"name": "test"}'

            result = await client.generate_async(
                "test prompt",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                schema_field="response_format",
                retries=1,
            )

            assert result == {"name": "test"}

    def test_generate_retry_with_instruction(self):
        """Test retry logic with instruction in sync generate."""
        client = MockValidatingLLMClient()

        call_count = 0

        def mock_generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "invalid json"
            return '{"name": "test"}'

        with patch.object(
            BaseLLMClient, "_generate", side_effect=mock_generate_side_effect
        ):
            result = client.generate(
                "test prompt",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                include_schema_in_system_prompt=True,
                retries=2,
            )

            assert result == {"name": "test"}
            assert call_count == 2

    def test_generate_retry_without_instruction(self):
        """Test retry logic without instruction in sync generate."""
        client = MockValidatingLLMClient()

        call_count = 0

        def mock_generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "invalid json"
            return '{"name": "test"}'

        with patch.object(
            BaseLLMClient, "_generate", side_effect=mock_generate_side_effect
        ):
            result = client.generate(
                "test prompt",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                include_schema_in_system_prompt=False,
                retries=2,
            )

            assert result == {"name": "test"}
            assert call_count == 2

    def test_generate_retry_chat_format(self):
        """Test retry logic with chat format in sync generate."""
        client = MockValidatingLLMClient()

        call_count = 0

        def mock_generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "invalid json"
            return '{"name": "test"}'

        with patch.object(
            BaseLLMClient, "_generate", side_effect=mock_generate_side_effect
        ):
            result = client.generate(
                [{"role": "user", "content": "test prompt"}],
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                retries=2,
            )

            assert result == {"name": "test"}
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_async_retry_with_instruction(self):
        """Test async retry logic with instruction."""
        client = MockValidatingLLMClient()

        call_count = 0

        async def mock_generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "invalid json"
            return '{"name": "test"}'

        with patch.object(
            BaseLLMClient, "_generate_async", side_effect=mock_generate_side_effect
        ):
            result = await client.generate_async(
                "test prompt",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                include_schema_in_system_prompt=True,
                retries=2,
            )

            assert result == {"name": "test"}
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_async_retry_without_instruction(self):
        """Test async retry logic without instruction."""
        client = MockValidatingLLMClient()

        call_count = 0

        async def mock_generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "invalid json"
            return '{"name": "test"}'

        with patch.object(
            BaseLLMClient, "_generate_async", side_effect=mock_generate_side_effect
        ):
            result = await client.generate_async(
                "test prompt",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                include_schema_in_system_prompt=False,
                retries=2,
            )

            assert result == {"name": "test"}
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_generate_async_retry_chat_format(self):
        """Test async retry logic with chat format."""
        client = MockValidatingLLMClient()

        call_count = 0

        async def mock_generate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "invalid json"
            return '{"name": "test"}'

        with patch.object(
            BaseLLMClient, "_generate_async", side_effect=mock_generate_side_effect
        ):
            result = await client.generate_async(
                [{"role": "user", "content": "test prompt"}],
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
                retries=2,
            )

            assert result == {"name": "test"}
            assert call_count == 2
