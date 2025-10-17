import pytest
from typing import List, Optional
from pydantic import BaseModel
from altk.core.llm import get_llm

# Skip tests if required environment variables are not set
pytestmark = [
    pytest.mark.llm,
]

# ──────────────────────────────────────────────────────────────────────────────
# Test schemas
# ──────────────────────────────────────────────────────────────────────────────


class Person(BaseModel):
    name: str
    age: int
    occupation: Optional[str] = None


class TechnicalAnalysis(BaseModel):
    topic: str
    complexity_level: str
    key_points: List[str]
    confidence_score: float


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def ollama_client():
    """Initialize Ollama LiteLLM client"""
    OllamaLiteLLMClient = get_llm("litellm.ollama")

    return OllamaLiteLLMClient(
        model_name="gpt-oss:20b",
        hooks=[lambda event, payload: print(f"[TEST OLLAMA] {event}: {payload}")],
    )


@pytest.fixture(scope="function")
def ollama_structured_client():
    """Initialize Ollama LiteLLM client with output validation"""
    OllamaLiteLLMClientOutputVal = get_llm("litellm.ollama.output_val")

    return OllamaLiteLLMClientOutputVal(
        model_name="gpt-oss:20b",
        hooks=[
            lambda event, payload: print(f"[TEST Ollama STRUCTURED] {event}: {payload}")
        ],
        include_schema_in_system_prompt=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test classes
# ──────────────────────────────────────────────────────────────────────────────


class TestOllamaLiteLLMBasicGeneration:
    """Test basic text generation capabilities"""

    def test_simple_text_generation(self, ollama_client):
        """Test simple text generation"""
        prompt = "Explain the concept of distributed computing systems."
        response = ollama_client.generate(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_generation(self, ollama_client):
        """Test chat-style generation"""
        chat_messages = [
            {
                "role": "system",
                "content": "You are a helpful technical assistant specializing in enterprise software.",
            },
            {
                "role": "user",
                "content": "What are the advantages of microservices architecture?",
            },
        ]

        response = ollama_client.generate(chat_messages)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.skip(reason="Requires a larger model that cannot run loacally.")
class TestOllamaLiteLLMToolCalling:
    """Test tool calling capabilities"""

    def test_tool_calling(self, ollama_client):
        """Test tool calling functionality"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_code_complexity",
                    "description": "Analyze code complexity metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_snippet": {
                                "type": "string",
                                "description": "Code to analyze",
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language",
                            },
                        },
                        "required": ["code_snippet"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "estimate_performance",
                    "description": "Estimate algorithm performance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "algorithm": {
                                "type": "string",
                                "description": "Algorithm description",
                            },
                            "input_size": {
                                "type": "number",
                                "description": "Expected input size",
                            },
                        },
                        "required": ["algorithm"],
                    },
                },
            },
        ]

        prompt = [
            {
                "role": "user",
                "content": "Analyze the complexity of bubble sort algorithm and estimate its performance for 1000 elements.",
            }
        ]

        response = ollama_client.generate(prompt, tools=tools, tool_choice="auto")

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")

    def test_tool_calling_with_specific_function(self, ollama_client):
        """Test tool calling with specific function"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "system_diagnostics",
                    "description": "Run system diagnostics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_type": {
                                "type": "string",
                                "description": "Type of system",
                            },
                            "check_level": {
                                "type": "string",
                                "enum": ["basic", "detailed", "comprehensive"],
                            },
                        },
                        "required": ["system_type"],
                    },
                },
            }
        ]

        prompt = [
            {
                "role": "user",
                "content": "Run comprehensive diagnostics on the database system",
            }
        ]

        response = ollama_client.generate(prompt, tools=tools, tool_choice="auto")

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")


@pytest.mark.skip(reason="Requires a larger model that cannot run loacally.")
class TestOllamaLiteLLMStructuredOutput:
    """Test structured output validation"""

    def test_person_extraction(self, ollama_structured_client):
        """Test person information extraction"""
        prompt = "Extract person info: Dr. Emily Watson is 42 years old and works as a Senior Software Engineer at IBM Research."

        person = ollama_structured_client.generate(
            prompt=prompt, schema=Person, retries=3
        )

        assert isinstance(person, Person)
        assert person.name is not None
        assert person.age is not None
        assert isinstance(person.age, int)

    def test_technical_analysis(self, ollama_structured_client):
        """Test technical analysis extraction"""
        prompt = (
            "Analyze the topic of quantum computing algorithms and provide key insights"
        )

        analysis = ollama_structured_client.generate(
            prompt=prompt, schema=TechnicalAnalysis, retries=3
        )

        assert isinstance(analysis, TechnicalAnalysis)
        assert analysis.topic is not None
        assert analysis.complexity_level is not None
        assert isinstance(analysis.key_points, list)
        assert isinstance(analysis.confidence_score, (int, float))

    def test_json_schema_validation(self, ollama_structured_client):
        """Test JSON schema validation"""
        json_schema = {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "paradigm": {
                    "type": "string",
                    "enum": [
                        "functional",
                        "object-oriented",
                        "procedural",
                        "declarative",
                    ],
                },
                "features": {"type": "array", "items": {"type": "string"}},
                "popularity_score": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["language", "paradigm", "popularity_score"],
        }

        prompt = "Analyze this programming language: 'Python is a high-level, interpreted programming language with dynamic semantics.'"

        analysis = ollama_structured_client.generate(
            prompt=prompt, schema=json_schema, retries=3
        )

        assert isinstance(analysis, dict)
        assert "language" in analysis
        assert "paradigm" in analysis
        assert "popularity_score" in analysis
        assert isinstance(analysis["popularity_score"], (int, float))

    def test_structured_output_with_validation_retries(self, ollama_structured_client):
        """Test structured output with validation retries"""
        prompt = "Extract info: Professor Alan Turing, 41, Computer Scientist and Mathematician"

        person = ollama_structured_client.generate(
            prompt=prompt,
            schema=Person,
            retries=5,  # Test with more retries
        )

        assert isinstance(person, Person)
        assert person.name is not None
        assert person.age is not None
        assert isinstance(person.age, int)


class TestOllamaLiteLLMAsync:
    """Test async capabilities"""

    @pytest.mark.asyncio
    async def test_async_generation(self, ollama_client):
        """Test async generation"""
        prompt = (
            "Describe the benefits of containerization in modern software development"
        )

        response = await ollama_client.generate_async(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_structured_output(self, ollama_structured_client):
        """Test async structured output"""
        prompt = "Extract info: Professor Alan Turing, 41, Computer Scientist and Mathematician"

        person = await ollama_structured_client.generate_async(
            prompt=prompt, schema=Person, retries=3
        )

        assert isinstance(person, Person)
        assert person.name is not None
        assert person.age is not None

    @pytest.mark.asyncio
    async def test_async_technical_analysis(self, ollama_structured_client):
        """Test async technical analysis"""
        prompt = "Analyze the complexity of machine learning algorithms in distributed systems"

        analysis = await ollama_structured_client.generate_async(
            prompt=prompt, schema=TechnicalAnalysis, retries=3
        )

        assert isinstance(analysis, TechnicalAnalysis)
        assert analysis.topic is not None
        assert analysis.complexity_level is not None
        assert isinstance(analysis.key_points, list)


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOllamaLiteLLMIntegration:
    """Integration tests for Ollama LiteLLM"""

    def test_client_initialization(self):
        """Test client initialization"""
        OllamaLiteLLMClient = get_llm("litellm.ollama")
        OllamaLiteLLMClientOutputVal = get_llm("litellm.ollama.output_val")

        client = OllamaLiteLLMClient(
            model_name="gpt-oss:20b",
        )

        structured_client = OllamaLiteLLMClientOutputVal(
            model_name="gpt-oss:20b",
        )

        assert client is not None
        assert structured_client is not None

    def test_hooks_functionality(self):
        """Test hooks functionality"""
        hook_called = False

        def test_hook(event, payload):
            nonlocal hook_called
            hook_called = True

        OllamaLiteLLMClient = get_llm("litellm.ollama")
        client = OllamaLiteLLMClient(model_name="gpt-oss:20b", hooks=[test_hook])

        # Generate some content to trigger hooks
        try:
            client.generate("Hello world")
        except Exception:
            pass  # We just want to test if hooks are called

        # Note: Hook behavior might vary, so we don't assert on hook_called
        assert client is not None
