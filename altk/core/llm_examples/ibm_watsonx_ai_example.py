import asyncio
from typing import List, Optional
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
# 2. Initialize IBM WatsonX AI clients
# ──────────────────────────────────────────────────────────────────────────────

# Basic IBM WatsonX AI clients
try:
    WatsonXAIClient = get_llm("watsonx")
    WatsonXAIClientOutputVal = get_llm("watsonx.output_val")

    client = WatsonXAIClient(
        model_name="meta-llama/llama-3-3-70b-instruct",
        hooks=[lambda event, payload: print(f"[SYNC HOOK] {event}: {payload}")],
    )

    client_output_val = WatsonXAIClientOutputVal(
        model_name="meta-llama/llama-3-3-70b-instruct",
        hooks=[lambda event, payload: print(f"[SYNC VAL HOOK] {event}: {payload}")],
        include_schema_in_system_prompt=True,
    )

    print("✅ IBM WatsonX AI clients initialized successfully")

except Exception as e:
    print(f"❌ Error initializing IBM WatsonX AI clients: {e}")
    print("Make sure to set WX_API_KEY and WX_PROJECT_ID environment variables")
    exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Text vs. Chat generation examples
# ──────────────────────────────────────────────────────────────────────────────


def demo_text_vs_chat_generation():
    """Demonstrate the difference between text and chat generation modes"""
    print("\n" + "=" * 60)
    print("TEXT vs CHAT GENERATION EXAMPLES")
    print("=" * 60)

    # Text generation (using generate endpoint)
    print("\n--- Text Generation Mode ---")
    text_prompt = "The key advantages of IBM Watson for enterprise AI are"

    # WatsonX text generation parameters (uses TextGenParameters internally)
    watsonx_text_args = GenerationArgs(
        max_tokens=100,
        temperature=0.4,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        seed=42,
        stop_sequences=[".", "\n\n"],
        decoding_method="sample",  # "greedy" or "sample"
    )

    try:
        response = client.generate(
            text_prompt, mode=GenerationMode.TEXT, generation_args=watsonx_text_args
        )
        print(f"Text completion: {response}")
        print(f"Generation Args: {watsonx_text_args}")
    except Exception as e:
        print(f"Text generation error: {e}")

    # Chat generation (using chat endpoint)
    print("\n--- Chat Generation Mode ---")
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
    try:
        response = client.generate(chat_messages, mode=GenerationMode.CHAT)
        print(f"Chat completion: {response}")
    except Exception as e:
        print(f"Chat generation error: {e}")

    # Demonstrate automatic mode selection
    print("\n--- Auto Mode Selection ---")
    try:
        # String input - will use default chat mode with auto-conversion
        response = client.generate("Explain the benefits of foundation models in AI.")
        print(f"Auto-mode response: {response}")
    except Exception as e:
        print(f"Auto-mode error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Basic generation examples
# ──────────────────────────────────────────────────────────────────────────────


def demo_basic_generation():
    """Demonstrate basic text generation without validation"""
    print("\n" + "=" * 60)
    print("BASIC GENERATION EXAMPLES")
    print("=" * 60)

    # Simple text generation
    print("\n--- Simple Text Generation ---")
    prompt = "Explain the benefits of using IBM Watson AI in enterprise applications"
    try:
        response = client.generate(prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Chat-style generation
    print("\n--- Chat-style Generation ---")
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

    # WatsonX chat generation parameters (uses TextChatParameters internally)
    watsonx_chat_args = GenerationArgs(
        max_tokens=150,
        temperature=0.6,
        top_p=0.85,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        seed=123,
        stop_sequences=["Human:", "AI:"],
    )

    try:
        chat_response = client.generate(
            chat_messages, mode=GenerationMode.CHAT, generation_args=watsonx_chat_args
        )
        print(f"Chat Response: {chat_response}")
        print(f"Generation Args: {watsonx_chat_args}")
    except Exception as e:
        print(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Tool calling examples
# ──────────────────────────────────────────────────────────────────────────────


def demo_tool_calling():
    """Demonstrate tool calling without validation"""
    print("\n" + "=" * 60)
    print("TOOL CALLING EXAMPLES")
    print("=" * 60)

    # Define tools
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

    # Tool calling example
    print("\n--- Tool Calling ---")
    prompt = [
        {
            "role": "user",
            "content": "Analyze our sales data for Q4 and generate a quarterly report.",
        }
    ]

    try:
        response = client.generate(prompt, tools=tools, tool_choice="auto")
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
# 5. Structured output examples
# ──────────────────────────────────────────────────────────────────────────────


def demo_structured_output():
    """Demonstrate structured output validation"""
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT EXAMPLES")
    print("=" * 60)

    # Pydantic model validation
    print("\n--- Pydantic Model Validation ---")
    prompt = "Generate information about a person named David who is 40 years old and works as a data scientist."

    try:
        response = client_output_val.generate(prompt, schema=Person, retries=3)
        print(f"Validated Person: {response}")
        print(f"Name: {response.name}, Age: {response.age}, Email: {response.email}")
    except Exception as e:
        print(f"Error: {e}")

    # Business insight validation
    print("\n--- Business Insight Validation ---")
    business_prompt = "Analyze the trend that 70% of companies are adopting AI technologies and provide insights."

    try:
        response = client_output_val.generate(
            business_prompt, schema=BusinessInsight, retries=3
        )
        print(f"Validated Business Insight: {response}")
        print(f"Confidence: {response.confidence}")
        print(f"Recommendation: {response.recommendation}")
    except Exception as e:
        print(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Async examples
# ──────────────────────────────────────────────────────────────────────────────


async def demo_async():
    """Demonstrate async generation"""
    print("\n" + "=" * 60)
    print("ASYNC GENERATION EXAMPLES")
    print("=" * 60)

    # Async text generation
    print("\n--- Async Text Generation ---")
    try:
        response = await client.generate_async(
            "What are the future trends in artificial intelligence?",
            mode=GenerationMode.TEXT_ASYNC,
        )
        print(f"Async text response: {response}")
    except Exception as e:
        print(f"Async text error: {e}")

    # Async chat generation
    print("\n--- Async Chat Generation ---")
    chat_messages = [
        {"role": "system", "content": "You are an AI expert."},
        {
            "role": "user",
            "content": "What are the ethical considerations in AI development?",
        },
    ]
    try:
        response = await client.generate_async(
            chat_messages, mode=GenerationMode.CHAT_ASYNC
        )
        print(f"Async chat response: {response}")
    except Exception as e:
        print(f"Async chat error: {e}")

    # Async structured output
    print("\n--- Async Structured Output ---")
    try:
        response = await client_output_val.generate_async(
            "Generate information about a person named Emma who is 29 years old.",
            schema=Person,
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
    print("🚀 IBM WatsonX AI Client Examples")
    print("=" * 60)

    demo_text_vs_chat_generation()
    demo_basic_generation()
    demo_tool_calling()
    demo_structured_output()

    # Run async examples
    print("\n--- Running Async Examples ---")
    try:
        asyncio.run(demo_async())
    except Exception as e:
        print(f"Async examples error: {e}")

    print("\n" + "=" * 60)
    print("✅ All IBM WatsonX AI examples completed!")


if __name__ == "__main__":
    main()
