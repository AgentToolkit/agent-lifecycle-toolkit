import asyncio
from typing import List, Optional
from pydantic import BaseModel

from altk.core.llm import get_llm
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


class DocumentSummary(BaseModel):
    title: str
    main_topics: List[str]
    word_count: int
    sentiment: str


# ──────────────────────────────────────────────────────────────────────────────
# 2. Initialize LiteLLM WatsonX clients
# ──────────────────────────────────────────────────────────────────────────────

# Basic LiteLLM WatsonX clients
try:
    WatsonXLiteLLMClient = get_llm("litellm.watsonx")
    WatsonXLiteLLMClientOutputVal = get_llm("litellm.watsonx.output_val")

    client = WatsonXLiteLLMClient(
        model_name="meta-llama/llama-3-3-70b-instruct",
        hooks=[lambda event, payload: print(f"[SYNC HOOK] {event}: {payload}")],
    )

    client_output_val = WatsonXLiteLLMClientOutputVal(
        model_name="meta-llama/llama-3-3-70b-instruct",
        hooks=[lambda event, payload: print(f"[SYNC VAL HOOK] {event}: {payload}")],
        include_schema_in_system_prompt=True,
    )

except Exception as e:
    print(f"Error initializing LiteLLM WatsonX clients: {e}")
    print(
        "Make sure to set WX_API_KEY, WX_PROJECT_ID, and WX_URL environment variables."
    )
    exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Basic generation examples
# ──────────────────────────────────────────────────────────────────────────────


def demo_basic_generation():
    """Demonstrate basic text generation without validation"""
    print("\n=== Basic LiteLLM WatsonX Generation ===")

    # Simple text generation
    prompt = "Explain artificial intelligence in simple terms"

    # LiteLLM WatsonX generation arguments
    litellm_gen_args = GenerationArgs(
        max_tokens=120,
        temperature=0.5,
        top_p=0.9,
        seed=456,
        decoding_method="sample",
    )

    response = client.generate(prompt, generation_args=litellm_gen_args)
    print(f"Response: {response}")
    print(f"Generation Args: {litellm_gen_args}")

    # Chat-style generation
    chat_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "What are the key benefits of using AI in business?",
        },
    ]
    chat_response = client.generate(chat_messages)
    print(f"Chat Response: {chat_response}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Tool calling examples
# ──────────────────────────────────────────────────────────────────────────────


def demo_tool_calling():
    """Demonstrate tool calling without validation"""
    print("\n=== LiteLLM WatsonX Tool Calling ===")

    # Define tools
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

    response = client.generate(messages, tools=tools, tool_choice="auto")

    print(f"Tool Response: {response}")

    # Check if response contains tool calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Tool calls detected: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Structured output examples
# ──────────────────────────────────────────────────────────────────────────────


def demo_structured_output():
    """Demonstrate structured output with validation"""
    print("\n=== LiteLLM WatsonX Structured Output ===")

    # Pydantic model validation
    prompt = (
        "Create a person profile for Sarah Johnson, age 42, email sarah@company.com"
    )
    person = client_output_val.generate(
        prompt,
        schema=Person,
    )

    print(f"Person object: {person}")
    print(f"  - Name: {person.name}")
    print(f"  - Age: {person.age}")
    print(f"  - Email: {person.email}")

    # JSON Schema validation
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

    weather_prompt = "Give me the weather for Tokyo: 18°C, cloudy, humidity 70%"
    weather = client_output_val.generate(
        weather_prompt,
        schema=weather_schema,
    )

    print(f"Weather data: {weather}")

    # Document summary with validation
    doc_text = """
    Machine learning is a subset of artificial intelligence that focuses on
    algorithms that can learn from data. It has applications in various fields
    including healthcare, finance, and technology. The field is rapidly evolving
    with new techniques and applications being developed constantly.
    """

    summary = client_output_val.generate(
        f"Summarize this document: {doc_text}",
        schema=DocumentSummary,
    )

    print(f"Document Summary: {summary}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Async examples
# ──────────────────────────────────────────────────────────────────────────────


async def demo_async():
    """Demonstrate async operations"""
    print("\n=== LiteLLM WatsonX Async Examples ===")

    # Basic async generation
    prompt = "What are the latest trends in enterprise AI?"
    response = await client.generate_async(prompt)
    print(f"Async Response: {response}")

    # Async with validation
    person_prompt = "Create a person: Michael Chen, 35, michael@tech.com"
    person = await client_output_val.generate_async(
        person_prompt,
        schema=Person,
        include_schema_in_system_prompt=True,
    )
    print(f"Async Person: {person}")

    # Async tool calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "database_query",
                "description": "Query a database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query"},
                        "database": {"type": "string", "description": "Database name"},
                    },
                    "required": ["query", "database"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "Query the sales database for last month's revenue"}
    ]
    tool_response = await client.generate_async(
        messages, tools=tools, tool_choice="auto"
    )
    print(f"Async Tool Response: {tool_response}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main execution
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """Run all examples"""
    print("LiteLLM WatsonX Client Examples")
    print("=" * 50)

    # Run synchronous examples
    demo_basic_generation()
    demo_tool_calling()
    demo_structured_output()

    # Run async examples
    print("\nRunning async examples...")
    asyncio.run(demo_async())

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
