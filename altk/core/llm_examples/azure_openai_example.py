import os
import asyncio
from typing import Optional
from pydantic import BaseModel

from altk.core.llm import get_llm, GenerationMode
from altk.core.llm.types import GenerationArgs


# ──────────────────────────────────────────────────────────────────────────────
# 1. Define schemas for structured output
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
# 2. Initialize Azure OpenAI clients
# ──────────────────────────────────────────────────────────────────────────────

try:
    # Basic Azure OpenAI clients
    SyncAzureOpenAIClient = get_llm("azure_openai.sync")
    AsyncAzureOpenAIClient = get_llm("azure_openai.async")

    # Structured output clients
    SyncAzureOpenAIClientOutputVal = get_llm("azure_openai.sync.output_val")
    AsyncAzureOpenAIClientOutputVal = get_llm("azure_openai.async.output_val")

    # Initialize clients
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-api-key-here")
    AZURE_API_BASE = os.getenv("AZURE_API_BASE", "https://eteopenai.azure-api.net")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")

    sync_client = SyncAzureOpenAIClient(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_API_BASE,
        api_version=AZURE_API_VERSION,
        hooks=[lambda event, payload: print(f"[AZURE SYNC] {event}: {payload}")],
    )

    async_client = AsyncAzureOpenAIClient(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_API_BASE,
        api_version=AZURE_API_VERSION,
        hooks=[lambda event, payload: print(f"[AZURE ASYNC] {event}: {payload}")],
    )

    sync_structured_client = SyncAzureOpenAIClientOutputVal(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_API_BASE,
        api_version=AZURE_API_VERSION,
        hooks=[
            lambda event, payload: print(f"[AZURE SYNC STRUCTURED] {event}: {payload}")
        ],
        include_schema_in_system_prompt=True,
    )

    async_structured_client = AsyncAzureOpenAIClientOutputVal(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_API_BASE,
        api_version=AZURE_API_VERSION,
        hooks=[
            lambda event, payload: print(f"[AZURE ASYNC STRUCTURED] {event}: {payload}")
        ],
        include_schema_in_system_prompt=True,
    )

    print("✅ Azure OpenAI clients initialized successfully")

except Exception as e:
    print(f"❌ Failed to initialize Azure OpenAI clients: {e}")
    print("Make sure to install: pip install toolkit-core[openai]")
    print(
        "And set your AZURE_OPENAI_API_KEY, AZURE_API_BASE and AZURE_API_VERSION environment variables"
    )
    exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Basic generation examples (without output validation)
# ──────────────────────────────────────────────────────────────────────────────


def demo_basic_generation():
    """Demonstrate basic text generation without output validation"""
    print("\n" + "=" * 60)
    print("BASIC GENERATION EXAMPLES (No Output Validation)")
    print("=" * 60)

    # Simple text generation
    print("\n--- Simple Text Generation ---")
    text_prompt = "Explain machine learning in simple terms."

    # Azure OpenAI generation parameters
    azure_gen_args = GenerationArgs(
        max_tokens=100,
        temperature=0.5,
        top_p=0.8,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        decoding_method="sample",
    )

    try:
        response = sync_client.generate(
            text_prompt, model="gpt-4o-2024-08-06", generation_args=azure_gen_args
        )
        print(f"Response: {response}")
        print(f"Generation Args: {azure_gen_args}")
    except Exception as e:
        print(f"Error: {e}")

    # Chat messages
    print("\n--- Chat Messages ---")
    chat_prompt = [
        {"role": "system", "content": "You are a helpful Azure AI assistant."},
        {"role": "user", "content": "What are the benefits of cloud computing?"},
    ]
    try:
        response = sync_client.generate(chat_prompt, model="gpt-4o-2024-08-06")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Tool calling examples (without output validation)
# ──────────────────────────────────────────────────────────────────────────────


def demo_tool_calling():
    """Demonstrate tool calling without output validation"""
    print("\n" + "=" * 60)
    print("TOOL CALLING EXAMPLES (No Output Validation)")
    print("=" * 60)

    # Define tools
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
                        "service": {"type": "string", "description": "Service type"},
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

    # Tool calling example
    print("\n--- Tool Calling ---")
    prompt = [
        {
            "role": "user",
            "content": "Get information about Azure Functions in East US region and calculate costs for 100 hours of usage.",
        }
    ]

    try:
        response = sync_client.generate(
            prompt, model="gpt-4o-2024-08-06", tools=tools, tool_choice="auto"
        )
        print(f"Response: {response}")

        # Check if response contains tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("Tool calls detected:")
            for tool_call in response.tool_calls:
                print(
                    f"  - {tool_call['function']['name']}: {tool_call['function']['arguments']}"
                )
    except Exception as e:
        print(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Structured output examples (with output validation)
# ──────────────────────────────────────────────────────────────────────────────


def demo_structured_output():
    """Demonstrate structured output validation"""
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT EXAMPLES (With Output Validation)")
    print("=" * 60)

    # Pydantic model validation
    print("\n--- Pydantic Model Validation ---")
    prompt = "Generate information about a person named Sarah who is 28 years old and works at Microsoft."

    try:
        response = sync_structured_client.generate(
            prompt, schema=Person, model="gpt-4o-2024-08-06", retries=3
        )
        print(f"Validated Person: {response}")
        print(f"Name: {response.name}, Age: {response.age}, Email: {response.email}")
    except Exception as e:
        print(f"Error: {e}")

    # JSON Schema validation
    print("\n--- JSON Schema Validation ---")
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

    try:
        response = sync_structured_client.generate(
            "Generate weather information for Seattle with temperature 12°C, rainy condition, and 80% humidity.",
            schema=weather_schema,
            model="gpt-4o-2024-08-06",
            retries=3,
        )
        print(f"Validated Weather: {response}")
    except Exception as e:
        print(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Async examples
# ──────────────────────────────────────────────────────────────────────────────


async def demo_async_examples():
    """Demonstrate async generation capabilities"""
    print("\n" + "=" * 60)
    print("ASYNC GENERATION EXAMPLES")
    print("=" * 60)

    # Async text generation
    print("\n--- Async Text Generation ---")
    try:
        response = await async_client.generate_async(
            "What are the advantages of using Azure OpenAI Service?",
            mode=GenerationMode.CHAT_ASYNC,
            model="gpt-4o-2024-08-06",
        )
        print(f"Async response: {response}")
    except Exception as e:
        print(f"Async error: {e}")

    # Async structured output
    print("\n--- Async Structured Output ---")
    try:
        response = await async_structured_client.generate_async(
            "Generate information about a person named Alex who is 35 years old.",
            schema=Person,
            model="gpt-4o-2024-08-06",
            retries=3,
        )
        print(f"Async structured response: {response}")
    except Exception as e:
        print(f"Async structured error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main execution
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """Run all examples"""
    print("🚀 Azure OpenAI Provider Examples")
    print("=" * 60)

    demo_basic_generation()
    demo_tool_calling()
    demo_structured_output()

    # Run async examples
    print("\n--- Running Async Examples ---")
    try:
        asyncio.run(demo_async_examples())
    except Exception as e:
        print(f"Async examples error: {e}")

    print("\n" + "=" * 60)
    print("✅ All Azure OpenAI examples completed!")


if __name__ == "__main__":
    main()
