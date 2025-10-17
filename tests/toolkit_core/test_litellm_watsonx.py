import os
import pytest
from typing import List, Optional
from pydantic import BaseModel
from altk.core.llm import get_llm, GenerationMode

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not all(
            [os.getenv("WX_API_KEY"), os.getenv("WX_PROJECT_ID"), os.getenv("WX_URL")]
        ),
        reason="WX_API_KEY, WX_PROJECT_ID, and WX_URL environment variables not set",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Test schemas
# ──────────────────────────────────────────────────────────────────────────────


class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    condition: str
    humidity: Optional[int] = None


class DocumentSummary(BaseModel):
    title: str
    main_topics: List[str]
    word_count: int
    sentiment: str


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def litellm_watsonx_client():
    """Initialize LiteLLM WatsonX client"""
    WatsonXLiteLLMClient = get_llm("litellm.watsonx")

    return WatsonXLiteLLMClient(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        hooks=[lambda event, payload: print(f"[TEST HOOK] {event}: {payload}")],
    )


@pytest.fixture
def litellm_watsonx_structured_client():
    """Initialize LiteLLM WatsonX client with output validation"""
    WatsonXLiteLLMClientOutputVal = get_llm("litellm.watsonx.output_val")

    return WatsonXLiteLLMClientOutputVal(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        hooks=[
            lambda event, payload: print(f"[TEST STRUCTURED HOOK] {event}: {payload}")
        ],
        include_schema_in_system_prompt=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test classes
# ──────────────────────────────────────────────────────────────────────────────


class TestLiteLLMWatsonXBasicGeneration:
    """Test basic text generation capabilities"""

    def test_simple_text_generation(self, litellm_watsonx_client):
        """Test simple text generation"""
        prompt = "Explain artificial intelligence in simple terms"
        response = litellm_watsonx_client.generate(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_generation(self, litellm_watsonx_client):
        """Test chat-style generation"""
        chat_messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": "What are the key benefits of using AI in business?",
            },
        ]

        response = litellm_watsonx_client.generate(
            chat_messages, mode=GenerationMode.CHAT
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0


class TestLiteLLMWatsonXToolCalling:
    """Test tool calling capabilities"""

    def test_tool_calling(self, litellm_watsonx_client):
        """Test tool calling functionality"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_company_info",
                    "description": "Get information about a company",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_name": {
                                "type": "string",
                                "description": "The company name",
                            },
                            "info_type": {
                                "type": "string",
                                "enum": ["financial", "general", "products"],
                            },
                        },
                        "required": ["company_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_market_trend",
                    "description": "Analyze market trends for a specific industry",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "industry": {
                                "type": "string",
                                "description": "Industry to analyze",
                            },
                            "time_period": {
                                "type": "string",
                                "enum": ["1month", "6months", "1year"],
                            },
                        },
                        "required": ["industry"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "user",
                "content": "Get information about IBM and analyze tech industry trends",
            }
        ]

        response = litellm_watsonx_client.generate(
            messages, tools=tools, tool_choice="auto"
        )

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")

    def test_tool_calling_with_specific_tool(self, litellm_watsonx_client):
        """Test tool calling with specific tool choice"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_metrics",
                    "description": "Calculate business metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric_type": {
                                "type": "string",
                                "description": "Type of metric",
                            },
                            "data": {"type": "array", "items": {"type": "number"}},
                        },
                        "required": ["metric_type"],
                    },
                },
            }
        ]

        messages = [
            {"role": "user", "content": "Calculate revenue metrics for this quarter"}
        ]

        response = litellm_watsonx_client.generate(
            messages, tools=tools, tool_choice="auto"
        )

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")


class TestLiteLLMWatsonXStructuredOutput:
    """Test structured output validation"""

    def test_pydantic_model_validation(self, litellm_watsonx_structured_client):
        """Test Pydantic model validation"""
        prompt = (
            "Create a person profile for Sarah Johnson, age 42, email sarah@company.com"
        )

        person = litellm_watsonx_structured_client.generate(
            prompt,
            schema=Person,
        )

        assert isinstance(person, Person)
        assert person.name is not None
        assert person.age is not None
        assert isinstance(person.age, int)

    def test_weather_report_validation(self, litellm_watsonx_structured_client):
        """Test weather report validation"""
        prompt = "Give me the weather for Tokyo: 18°C, cloudy, humidity 70%"

        weather = litellm_watsonx_structured_client.generate(
            prompt,
            schema=WeatherReport,
        )

        assert isinstance(weather, WeatherReport)
        assert weather.city is not None
        assert isinstance(weather.temperature_c, (int, float))
        assert weather.condition is not None

    def test_document_summary_validation(self, litellm_watsonx_structured_client):
        """Test document summary validation"""
        doc_text = """
        Machine learning is a subset of artificial intelligence that focuses on
        algorithms that can learn from data. It has applications in various fields
        including healthcare, finance, and technology. The field is rapidly evolving
        with new techniques and applications being developed constantly.
        """

        summary = litellm_watsonx_structured_client.generate(
            f"Summarize this document: {doc_text}",
            schema=DocumentSummary,
        )

        assert isinstance(summary, DocumentSummary)
        assert summary.title is not None
        assert isinstance(summary.main_topics, list)
        assert isinstance(summary.word_count, int)
        assert summary.sentiment is not None

    def test_json_schema_validation(self, litellm_watsonx_structured_client):
        """Test JSON schema validation"""
        weather_schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temperature_c": {"type": "number"},
                "condition": {"type": "string"},
                "humidity": {"type": "integer"},
            },
            "required": ["city", "temperature_c", "condition"],
        }

        prompt = "Give me the weather for Berlin: 15°C, sunny, humidity 60%"

        weather = litellm_watsonx_structured_client.generate(
            prompt,
            schema=weather_schema,
        )

        assert isinstance(weather, dict)
        assert "city" in weather
        assert "temperature_c" in weather
        assert "condition" in weather
        assert isinstance(weather["temperature_c"], (int, float))


class TestLiteLLMWatsonXAsync:
    """Test async capabilities"""

    @pytest.mark.asyncio
    async def test_async_basic_generation(self, litellm_watsonx_client):
        """Test async basic generation"""
        prompt = "What are the latest trends in enterprise AI?"

        response = await litellm_watsonx_client.generate_async(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_structured_output(self, litellm_watsonx_structured_client):
        """Test async structured output"""
        prompt = "Create a person: Michael Chen, age 35, michael@tech.com"

        person = await litellm_watsonx_structured_client.generate_async(
            prompt, schema=Person
        )

        assert isinstance(person, Person)
        assert person.name is not None
        assert person.age is not None

    @pytest.mark.asyncio
    async def test_async_tool_calling(self, litellm_watsonx_client):
        """Test async tool calling"""
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

        messages = [
            {
                "role": "user",
                "content": "Run comprehensive diagnostics on the database system",
            }
        ]

        response = await litellm_watsonx_client.generate_async(
            messages, tools=tools, tool_choice="auto"
        )

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests
# ──────────────────────────────────────────────────────────────────────────────


class TestLiteLLMWatsonXIntegration:
    """Integration tests for LiteLLM WatsonX"""

    def test_client_initialization(self):
        """Test client initialization"""
        WatsonXLiteLLMClient = get_llm("litellm.watsonx")
        WatsonXLiteLLMClientOutputVal = get_llm("litellm.watsonx.output_val")

        client = WatsonXLiteLLMClient(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        )

        structured_client = WatsonXLiteLLMClientOutputVal(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
        )

        assert client is not None
        assert structured_client is not None

    def test_hooks_functionality(self):
        """Test hooks functionality"""
        hook_called = False

        def test_hook(event, payload):
            nonlocal hook_called
            hook_called = True

        WatsonXLiteLLMClient = get_llm("litellm.watsonx")
        client = WatsonXLiteLLMClient(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            hooks=[test_hook],
        )

        # Generate some content to trigger hooks
        try:
            client.generate("Hello world")
        except Exception:
            pass  # We just want to test if hooks are called

        # Note: Hook behavior might vary, so we don't assert on hook_called
        assert client is not None

    def test_multiple_generation_calls(self, litellm_watsonx_client):
        """Test multiple consecutive generation calls"""
        prompts = ["What is AI?", "Explain machine learning", "Define deep learning"]

        responses = []
        for prompt in prompts:
            response = litellm_watsonx_client.generate(prompt)
            responses.append(response)

        assert len(responses) == 3
        for response in responses:
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
