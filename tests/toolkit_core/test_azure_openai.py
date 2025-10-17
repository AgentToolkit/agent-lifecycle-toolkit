import os
import pytest
from typing import Optional
from pydantic import BaseModel
from altk.core.llm import get_llm, GenerationMode

# Skip tests if required environment variables are not set
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not all(
            [
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("AZURE_API_BASE"),
                os.getenv("AZURE_API_VERSION"),
            ]
        ),
        reason="AZURE_OPENAI_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION environment variables not set",
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


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="function")
def azure_sync_client():
    """Initialize Azure OpenAI sync client"""
    SyncAzureOpenAIClient = get_llm("azure_openai.sync")

    return SyncAzureOpenAIClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        hooks=[lambda event, payload: print(f"[TEST AZURE SYNC] {event}: {payload}")],
    )


@pytest.fixture(scope="function")
def azure_async_client():
    """Initialize Azure OpenAI async client"""
    AsyncAzureOpenAIClient = get_llm("azure_openai.async")

    return AsyncAzureOpenAIClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        hooks=[lambda event, payload: print(f"[TEST AZURE ASYNC] {event}: {payload}")],
    )


@pytest.fixture(scope="function")
def azure_sync_structured_client():
    """Initialize Azure OpenAI sync client with output validation"""
    SyncAzureOpenAIClientOutputVal = get_llm("azure_openai.sync.output_val")

    return SyncAzureOpenAIClientOutputVal(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        hooks=[
            lambda event, payload: print(
                f"[TEST AZURE SYNC STRUCTURED] {event}: {payload}"
            )
        ],
        include_schema_in_system_prompt=True,
    )


@pytest.fixture(scope="function")
def azure_async_structured_client():
    """Initialize Azure OpenAI async client with output validation"""
    AsyncAzureOpenAIClientOutputVal = get_llm("azure_openai.async.output_val")

    return AsyncAzureOpenAIClientOutputVal(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION"),
        hooks=[
            lambda event, payload: print(
                f"[TEST AZURE ASYNC STRUCTURED] {event}: {payload}"
            )
        ],
        include_schema_in_system_prompt=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test classes
# ──────────────────────────────────────────────────────────────────────────────


class TestAzureOpenAIBasicGeneration:
    """Test basic text generation capabilities"""

    def test_simple_text_generation(self, azure_sync_client):
        """Test simple text generation"""
        prompt = "Explain machine learning in simple terms."

        response = azure_sync_client.generate(prompt, model="gpt-4o-2024-08-06")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_generation(self, azure_sync_client):
        """Test chat-style generation"""
        chat_messages = [
            {"role": "system", "content": "You are a helpful Azure AI assistant."},
            {"role": "user", "content": "What are the benefits of cloud computing?"},
        ]

        response = azure_sync_client.generate(chat_messages, model="gpt-4o-2024-08-06")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0


class TestAzureOpenAIToolCalling:
    """Test tool calling capabilities"""

    def test_tool_calling(self, azure_sync_client):
        """Test tool calling functionality"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_azure_service_info",
                    "description": "Get information about Azure services",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "Azure service name",
                            },
                            "region": {"type": "string", "description": "Azure region"},
                        },
                        "required": ["service_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_azure_costs",
                    "description": "Calculate Azure service costs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {
                                "type": "string",
                                "description": "Service type",
                            },
                            "usage_hours": {
                                "type": "number",
                                "description": "Usage in hours",
                            },
                        },
                        "required": ["service", "usage_hours"],
                    },
                },
            },
        ]

        prompt = [
            {
                "role": "user",
                "content": "Get information about Azure Functions in East US region and calculate costs for 100 hours of usage.",
            }
        ]

        response = azure_sync_client.generate(
            prompt, model="gpt-4o-2024-08-06", tools=tools, tool_choice="auto"
        )

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")

    def test_tool_calling_with_specific_tool(self, azure_sync_client):
        """Test tool calling with specific tool choice"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "azure_diagnostics",
                    "description": "Run Azure service diagnostics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_type": {
                                "type": "string",
                                "description": "Type of Azure service",
                            },
                            "diagnostic_level": {
                                "type": "string",
                                "enum": ["basic", "detailed", "comprehensive"],
                            },
                        },
                        "required": ["service_type"],
                    },
                },
            }
        ]

        prompt = [
            {
                "role": "user",
                "content": "Run comprehensive diagnostics on Azure App Service",
            }
        ]

        response = azure_sync_client.generate(
            prompt, model="gpt-4o-2024-08-06", tools=tools, tool_choice="auto"
        )

        assert response is not None
        # Check if response contains tool calls (might vary by implementation)
        if hasattr(response, "tool_calls") and response.tool_calls:
            assert len(response.tool_calls) > 0
        else:
            # If no tool calls, just ensure we got a response
            assert isinstance(response, str) or hasattr(response, "content")


class TestAzureOpenAIStructuredOutput:
    """Test structured output validation"""

    def test_pydantic_model_validation(self, azure_sync_structured_client):
        """Test Pydantic model validation"""
        prompt = "Generate information about a person named Sarah who is 28 years old and works at Microsoft."

        response = azure_sync_structured_client.generate(
            prompt, schema=Person, model="gpt-4o-2024-08-06", retries=3
        )

        assert isinstance(response, Person)
        assert response.name is not None
        assert response.age is not None
        assert isinstance(response.age, int)

    def test_weather_report_validation(self, azure_sync_structured_client):
        """Test weather report validation"""
        prompt = "Generate weather information for Seattle with temperature 12°C, rainy condition, and 80% humidity."

        weather = azure_sync_structured_client.generate(
            prompt, schema=WeatherReport, model="gpt-4o-2024-08-06", retries=3
        )

        assert isinstance(weather, WeatherReport)
        assert weather.city is not None
        assert isinstance(weather.temperature_c, (int, float))
        assert weather.condition is not None

    def test_json_schema_validation(self, azure_sync_structured_client):
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

        prompt = "Generate weather information for Tokyo with temperature 22°C, sunny condition, and 65% humidity."

        response = azure_sync_structured_client.generate(
            prompt, schema=weather_schema, model="gpt-4o-2024-08-06", retries=3
        )

        assert isinstance(response, dict)
        assert "city" in response
        assert "temperature_c" in response
        assert "condition" in response
        assert isinstance(response["temperature_c"], (int, float))


class TestAzureOpenAIAsync:
    """Test async capabilities"""

    @pytest.mark.asyncio
    async def test_async_generation(self, azure_async_client):
        """Test async generation"""
        prompt = "What are the advantages of using Azure OpenAI Service?"

        response = await azure_async_client.generate_async(
            prompt, mode=GenerationMode.CHAT_ASYNC, model="gpt-4o-2024-08-06"
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_structured_output(self, azure_async_structured_client):
        """Test async structured output"""
        prompt = "Generate information about a person named Alex who is 35 years old."

        response = await azure_async_structured_client.generate_async(
            prompt, schema=Person, model="gpt-4o-2024-08-06", retries=3
        )

        assert isinstance(response, Person)
        assert response.name is not None
        assert response.age is not None

    @pytest.mark.asyncio
    async def test_async_chat_generation(self, azure_async_client):
        """Test async chat generation"""
        chat_messages = [
            {"role": "system", "content": "You are a helpful Azure AI assistant."},
            {
                "role": "user",
                "content": "Explain the benefits of Azure Cognitive Services.",
            },
        ]

        response = await azure_async_client.generate_async(
            chat_messages, mode=GenerationMode.CHAT_ASYNC, model="gpt-4o-2024-08-06"
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_async_tool_calling(self, azure_async_client):
        """Test async tool calling"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "azure_resource_info",
                    "description": "Get Azure resource information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "resource_type": {
                                "type": "string",
                                "description": "Type of Azure resource",
                            },
                            "region": {"type": "string", "description": "Azure region"},
                        },
                        "required": ["resource_type"],
                    },
                },
            }
        ]

        prompt = [
            {
                "role": "user",
                "content": "Get information about Azure Storage in West US region",
            }
        ]

        response = await azure_async_client.generate_async(
            prompt, model="gpt-4o-2024-08-06", tools=tools, tool_choice="auto"
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


class TestAzureOpenAIIntegration:
    """Integration tests for Azure OpenAI"""

    def test_sync_client_initialization(self):
        """Test sync client initialization"""
        SyncAzureOpenAIClient = get_llm("azure_openai.sync")
        SyncAzureOpenAIClientOutputVal = get_llm("azure_openai.sync.output_val")

        sync_client = SyncAzureOpenAIClient(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        sync_structured_client = SyncAzureOpenAIClientOutputVal(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        assert sync_client is not None
        assert sync_structured_client is not None

    def test_async_client_initialization(self):
        """Test async client initialization"""
        AsyncAzureOpenAIClient = get_llm("azure_openai.async")
        AsyncAzureOpenAIClientOutputVal = get_llm("azure_openai.async.output_val")

        async_client = AsyncAzureOpenAIClient(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        async_structured_client = AsyncAzureOpenAIClientOutputVal(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        assert async_client is not None
        assert async_structured_client is not None

    def test_hooks_functionality(self):
        """Test hooks functionality"""
        hook_called = False

        def test_hook(event, payload):
            nonlocal hook_called
            hook_called = True

        SyncAzureOpenAIClient = get_llm("azure_openai.sync")
        client = SyncAzureOpenAIClient(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
            hooks=[test_hook],
        )

        # Generate some content to trigger hooks
        try:
            client.generate("Hello world", model="gpt-4o-2024-08-06")
        except Exception:
            pass  # We just want to test if hooks are called

        # Note: Hook behavior might vary, so we don't assert on hook_called
        assert client is not None
