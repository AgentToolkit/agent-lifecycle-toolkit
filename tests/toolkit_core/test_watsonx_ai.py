import os
import pytest
from typing import List, Optional
from pydantic import BaseModel
from altk.core.llm import get_llm, GenerationMode

# Skip tests if required environment variables are not set
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not all([os.getenv("WX_API_KEY"), os.getenv("WX_PROJECT_ID")]),
        reason="WX_API_KEY and WX_PROJECT_ID environment variables not set",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# Test schemas
# ──────────────────────────────────────────────────────────────────────────────


class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class BusinessInsight(BaseModel):
    insight: str
    confidence: float
    supporting_data: List[str]
    recommendation: str


class TechnicalAnalysis(BaseModel):
    topic: str
    complexity_level: str
    key_points: List[str]
    estimated_time: str


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def watsonx_client():
    """Initialize WatsonX AI client"""
    WatsonXAIClient = get_llm("watsonx")

    return WatsonXAIClient(
        model_name="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
    )


@pytest.fixture(scope="function")
def watsonx_structured_client():
    """Initialize WatsonX AI client with output validation"""
    WatsonXAIClientOutputVal = get_llm("watsonx.output_val")

    return WatsonXAIClientOutputVal(
        model_name="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
        include_schema_in_system_prompt=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test classes
# ──────────────────────────────────────────────────────────────────────────────


class TestWatsonXAIBasicGeneration:
    """Test basic text generation capabilities"""

    def test_simple_text_generation(self, watsonx_client):
        """Test simple text generation"""
        prompt = (
            "Explain the benefits of using IBM Watson AI in enterprise applications"
        )
        response = watsonx_client.generate(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_generation(self, watsonx_client):
        """Test chat-style generation"""
        chat_messages = [
            {
                "role": "system",
                "content": "You are an AI assistant specialized in enterprise technology.",
            },
            {
                "role": "user",
                "content": "What are the key considerations for implementing AI in a large organization?",
            },
        ]

        response = watsonx_client.generate(chat_messages, mode=GenerationMode.CHAT)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_text_vs_chat_modes(self, watsonx_client):
        """Test different generation modes"""
        # Test text mode
        text_prompt = "The key advantages of IBM Watson for enterprise AI are"
        text_response = watsonx_client.generate(text_prompt, mode=GenerationMode.TEXT)

        # Test chat mode
        chat_messages = [
            {
                "role": "system",
                "content": "You are an AI assistant specialized in enterprise technology.",
            },
            {
                "role": "user",
                "content": "What are the key advantages of IBM Watson for enterprise AI?",
            },
        ]
        chat_response = watsonx_client.generate(chat_messages, mode=GenerationMode.CHAT)

        assert text_response is not None
        assert chat_response is not None
        assert isinstance(text_response, str)
        assert isinstance(chat_response, str)


class TestWatsonXAIToolCalling:
    """Test tool calling capabilities"""

    def test_tool_calling(self, watsonx_client):
        """Test tool calling functionality"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_business_data",
                    "description": "Analyze business data for insights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_type": {
                                "type": "string",
                                "description": "Type of business data",
                            },
                            "metrics": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["data_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "Generate business report",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "report_type": {"type": "string"},
                            "period": {"type": "string"},
                        },
                        "required": ["report_type"],
                    },
                },
            },
        ]

        prompt = [
            {
                "role": "user",
                "content": "Analyze our sales data for Q4 and generate a quarterly report.",
            }
        ]

        response = watsonx_client.generate(prompt, tools=tools, tool_choice="auto")

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")


class TestWatsonXAIStructuredOutput:
    """Test structured output validation"""

    def test_pydantic_model_validation(self, watsonx_structured_client):
        """Test Pydantic model validation"""
        prompt = "Generate information about a person named David who is 40 years old and works as a data scientist."

        response = watsonx_structured_client.generate(prompt, schema=Person, retries=3)

        assert isinstance(response, Person)
        assert response.name is not None
        assert response.age is not None
        assert isinstance(response.age, int)

    def test_business_insight_validation(self, watsonx_structured_client):
        """Test business insight validation"""
        prompt = "Analyze the trend that 70% of companies are adopting AI technologies and provide insights."

        response = watsonx_structured_client.generate(
            prompt, schema=BusinessInsight, retries=3
        )

        assert isinstance(response, BusinessInsight)
        assert response.insight is not None
        assert isinstance(response.confidence, float)
        assert isinstance(response.supporting_data, list)
        assert response.recommendation is not None

    def test_json_schema_validation(self, watsonx_structured_client):
        """Test JSON schema validation"""
        json_schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "complexity": {"type": "string"},
                "points": {"type": "array", "items": {"type": "string"}},
                "rating": {"type": "number", "minimum": 0, "maximum": 10},
            },
            "required": ["topic", "complexity", "rating"],
        }

        prompt = "Analyze the complexity of machine learning algorithms."

        response = watsonx_structured_client.generate(
            prompt, schema=json_schema, retries=3
        )

        assert isinstance(response, dict)
        assert "topic" in response
        assert "complexity" in response
        assert "rating" in response
        assert isinstance(response["rating"], (int, float))


class TestWatsonXAIAsync:
    """Test async capabilities"""

    @pytest.mark.asyncio
    async def test_async_text_generation(self, watsonx_client):
        """Test async text generation"""
        prompt = "What are the future trends in artificial intelligence?"

        response = await watsonx_client.generate_async(
            prompt, mode=GenerationMode.TEXT_ASYNC
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_chat_generation(self, watsonx_client):
        """Test async chat generation"""
        chat_messages = [
            {"role": "system", "content": "You are an AI expert."},
            {
                "role": "user",
                "content": "What are the ethical considerations in AI development?",
            },
        ]

        response = await watsonx_client.generate_async(
            chat_messages, mode=GenerationMode.CHAT_ASYNC
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_structured_output(self, watsonx_structured_client):
        """Test async structured output"""
        prompt = "Generate information about a person named Emma who is 29 years old."

        response = await watsonx_structured_client.generate_async(
            prompt, schema=Person, include_schema_in_system_prompt=True, retries=3
        )

        assert isinstance(response, Person)
        assert response.name is not None
        assert response.age is not None
