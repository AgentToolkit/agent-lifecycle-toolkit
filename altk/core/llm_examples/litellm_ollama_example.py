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
    occupation: Optional[str] = None


class TechnicalAnalysis(BaseModel):
    topic: str
    complexity_level: str
    key_points: List[str]
    confidence_score: float


# ──────────────────────────────────────────────────────────────────────────────
# 2. Initialize Ollama LiteLLM clients
# ──────────────────────────────────────────────────────────────────────────────

try:
    # Basic Ollama LiteLLM clients
    OllamaLiteLLMClient = get_llm("litellm.ollama")
    OllamaLiteLLMClientOutputVal = get_llm("litellm.ollama.output_val")

    ollama_client = OllamaLiteLLMClient(
        model_name="llama3.2:1b",
        hooks=[lambda event, payload: print(f"[Ollama] {event}: {payload}")],
    )

    ollama_structured_client = OllamaLiteLLMClientOutputVal(
        model_name="llama3.2:1b",
        hooks=[lambda event, payload: print(f"[Ollama STRUCTURED] {event}: {payload}")],
        include_schema_in_system_prompt=True,
    )

    print("✅ Ollama LiteLLM clients initialized successfully")

except Exception as e:
    print(f"❌ Failed to initialize Ollama clients: {e}")
    print("Make sure to install: pip install toolkit-core[litellm]")
    print("And set your OLLAMA_API_KEY and OLLAMA_API_URL environment variables.")
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
    text_prompt = "Explain the concept of distributed computing systems."

    # Create generation arguments for controlling the output
    gen_args = GenerationArgs(
        max_tokens=150, temperature=0.7, top_p=0.9, stop_sequences=["<END>", "---"]
    )

    try:
        response = ollama_client.generate(text_prompt, generation_args=gen_args)
        print(f"Prompt: {text_prompt}")
        print(f"Response: {response}")
        print(f"Generation Args: {gen_args}")
    except Exception as e:
        print(f"Error: {e}")

    # Chat messages
    print("\n--- Chat Messages ---")
    chat_prompt = [
        {
            "role": "system",
            "content": "You are a helpful technical assistant specializing in enterprise software.",
        },
        {
            "role": "user",
            "content": "What are the advantages of microservices architecture?",
        },
    ]

    # Create different generation arguments for chat
    chat_gen_args = GenerationArgs(
        max_tokens=200,
        temperature=0.3,
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )

    try:
        response = ollama_client.generate(chat_prompt, generation_args=chat_gen_args)
        print(f"Prompt: {chat_prompt}")
        print(f"Response: {response}")
        print(f"Generation Args: {chat_gen_args}")
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

    # Tool calling example
    print("\n--- Tool Calling ---")
    prompt = [
        {
            "role": "user",
            "content": "Analyze the complexity of bubble sort algorithm and estimate its performance for 1000 elements.",
        }
    ]

    try:
        response = ollama_client.generate(prompt, tools=tools, tool_choice="auto")

        print(f"Prompt: {prompt[0]['content']}")

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Content: {response.content}")
            print("Tool calls:")
            for tool_call in response.tool_calls:
                print(
                    f"  - {tool_call['function']['name']}: {tool_call['function']['arguments']}"
                )
        else:
            print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Structured output examples (with output validation)
# ──────────────────────────────────────────────────────────────────────────────


def demo_structured_output():
    """Demonstrate structured output with validation"""
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT EXAMPLES (With Output Validation)")
    print("=" * 60)

    # Person extraction
    print("\n--- Person Extraction ---")
    prompt = "Extract person info: Dr. Emily Watson is 42 years old and works as a Senior Software Engineer at IBM Research."

    try:
        person = ollama_structured_client.generate(
            prompt=prompt, schema=Person, retries=3
        )
        print(f"Prompt: {prompt}")
        print(f"Extracted Person: {person}")
        print(
            f"Name: {person.name}, Age: {person.age}, Occupation: {person.occupation}"
        )
        print(f"Type: {type(person)}")
    except Exception as e:
        print(f"Person extraction failed: {e}")

    # Technical analysis
    print("\n--- Technical Analysis ---")
    prompt = (
        "Analyze the topic of quantum computing algorithms and provide key insights"
    )

    try:
        analysis = ollama_structured_client.generate(
            prompt=prompt, schema=TechnicalAnalysis, retries=3
        )
        print(f"Prompt: {prompt}")
        print(f"Technical Analysis: {analysis}")
        print(f"Topic: {analysis.topic}, Complexity: {analysis.complexity_level}")
        print(f"Key Points: {analysis.key_points}")
        print(f"Type: {type(analysis)}")
    except Exception as e:
        print(f"Technical analysis failed: {e}")

    # JSON Schema example
    print("\n--- JSON Schema Analysis ---")
    json_schema = {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "paradigm": {
                "type": "string",
                "enum": ["functional", "object-oriented", "procedural", "declarative"],
            },
            "features": {"type": "array", "items": {"type": "string"}},
            "popularity_score": {"type": "number", "minimum": 0, "maximum": 100},
        },
        "required": ["language", "paradigm", "popularity_score"],
    }

    prompt = "Analyze this programming language: 'Python is a high-level, interpreted programming language with dynamic semantics.'"

    try:
        analysis = ollama_structured_client.generate(
            prompt=prompt, schema=json_schema, retries=3
        )
        print(f"Prompt: {prompt}")
        print(f"Analysis: {analysis}")
        print(f"Type: {type(analysis)}")
    except Exception as e:
        print(f"JSON schema analysis failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Async examples
# ──────────────────────────────────────────────────────────────────────────────


async def demo_async_examples():
    """Demonstrate async functionality"""
    print("\n" + "=" * 60)
    print("ASYNC EXAMPLES")
    print("=" * 60)

    # Basic async generation
    print("\n--- Async Generation ---")
    prompt = "Describe the benefits of containerization in modern software development"
    try:
        response = await ollama_client.generate_async(prompt)
        print(f"Async Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Async structured output
    print("\n--- Async Structured Output ---")
    prompt = (
        "Extract info: Professor Alan Turing, 41, Computer Scientist and Mathematician"
    )

    try:
        person = await ollama_structured_client.generate_async(
            prompt=prompt, schema=Person, retries=3
        )
        print(f"Async Structured Response: {person}")
        print(f"Type: {type(person)}")
    except Exception as e:
        print(f"Async structured output failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main execution
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """Run all Ollama LiteLLM examples"""
    print("Ollama LiteLLM Provider Examples")
    print("=" * 60)

    # Run sync examples
    demo_basic_generation()
    demo_tool_calling()
    demo_structured_output()

    # Run async examples
    print("\nRunning async examples...")
    asyncio.run(demo_async_examples())

    print("\n" + "=" * 60)
    print("✅ All Ollama LiteLLM examples completed!")


if __name__ == "__main__":
    main()
