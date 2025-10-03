import json
import os
from langchain_core.messages import HumanMessage, AIMessage
import asyncio

from altk.pre_tool_reflection_toolkit.core import (
    SPARCReflectionRunInput,
    SPARCExecutionMode,
    Track,
)
from altk.pre_tool_reflection_toolkit.sparc import (
    SPARCReflectionComponent,
)
from altk.toolkit_core.core.toolkit import AgentPhase, ComponentConfig
from altk.toolkit_core.llm import get_llm


# Example tool specifications for mixed-domain scenarios
MULTI_DOMAIN_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, state/country (e.g., 'New York, NY')",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit", "kelvin"],
                        "description": "Temperature units",
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight from origin to destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Departure airport code (e.g., 'JFK')",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Arrival airport code (e.g., 'LAX')",
                    },
                    "departure_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Departure date in YYYY-MM-DD format",
                    },
                    "passengers": {
                        "type": "integer",
                        "description": "Number of passengers",
                        "minimum": 1,
                        "maximum": 9,
                    },
                    "class": {
                        "type": "string",
                        "enum": ["economy", "business", "first"],
                        "description": "Travel class",
                        "default": "economy",
                    },
                },
                "required": ["origin", "destination", "departure_date", "passengers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send an SMS message to a phone number",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "Recipient phone number in international format (+1234567890)",
                    },
                    "message": {
                        "type": "string",
                        "description": "SMS message content",
                        "maxLength": 160,
                    },
                },
                "required": ["phone_number", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tip",
            "description": "Calculate tip amount for a bill",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_amount": {
                        "type": "number",
                        "description": "Total bill amount before tip",
                        "minimum": 0.01,
                    },
                    "tip_percentage": {
                        "type": "number",
                        "description": "Tip percentage (e.g., 0.15 for 15%)",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "party_size": {
                        "type": "integer",
                        "description": "Number of people in the party",
                        "minimum": 1,
                    },
                },
                "required": ["bill_amount", "tip_percentage"],
            },
        },
    },
]

FINANCIAL_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "check_account_balance",
            "description": "Check the current balance of a bank account (requires authentication)",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_number": {
                        "type": "string",
                        "description": "Bank account number",
                    },
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings", "credit"],
                        "description": "Type of account",
                    },
                },
                "required": ["account_number", "account_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_money",
            "description": "Transfer money between accounts (requires authentication and sufficient funds)",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_account": {
                        "type": "string",
                        "description": "Source account number",
                    },
                    "to_account": {
                        "type": "string",
                        "description": "Destination account number",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount to transfer",
                        "minimum": 0.01,
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency code (e.g., 'USD')",
                        "default": "USD",
                    },
                },
                "required": ["from_account", "to_account", "amount"],
            },
        },
    },
]


def build_config():
    """Build ComponentConfig with WatsonX ValidatingLLMClient."""
    WATSONX_CLIENT = get_llm("watsonx.output_val")
    return ComponentConfig(
        llm_client=WATSONX_CLIENT(
            model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            api_key=os.getenv("WX_API_KEY"),
            project_id=os.getenv("WX_PROJECT_ID"),
            url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
        )
    )


async def run_semantic_issues_examples():
    """Run examples showing semantic validation issues."""

    # Initialize middleware using the new ComponentConfig pattern
    config = build_config()
    middleware = SPARCReflectionComponent(
        config=config,
        track=Track.FAST_TRACK,
        execution_mode=SPARCExecutionMode.ASYNC,
    )

    if middleware._initialization_error:
        print(f"Failed to initialize middleware: {middleware._initialization_error}")
        return

    print("=== Semantic Validation Issues Examples ===\n")

    # Example 1: Function selection misalignment
    await run_function_misalignment_example(middleware)

    # Example 2: Parameter value grounding issues
    await run_value_grounding_example(middleware)

    # Example 3: Parameter hallucination
    await run_parameter_hallucination_example(middleware)

    # Example 4: Valid semantic call (for comparison)
    await run_valid_semantic_example(middleware)


async def run_function_misalignment_example(middleware: SPARCReflectionComponent):
    """Example with function selection that doesn't align with user intent."""

    print("### Example 1: Function Selection Misalignment ###")
    print("User asks about weather, but tool call tries to book a flight\n")

    conversation_context = [
        HumanMessage(content="What's the weather like in New York today?"),
        AIMessage(content="I'll check the weather for you in New York."),
    ]

    # Wrong function selected - booking flight instead of getting weather
    misaligned_function_call = {
        "id": "1",
        "type": "function",
        "function": {
            "name": "book_flight",  # Should be "get_weather"
            "arguments": json.dumps(
                {
                    "origin": "JFK",
                    "destination": "LAX",
                    "departure_date": "2024-06-21",
                    "passengers": 1,
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=MULTI_DOMAIN_TOOL_SPECS,
        tool_calls=[misaligned_function_call],
    )

    result = await middleware.aprocess(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Function Selection Misalignment", result)


async def run_value_grounding_example(middleware: SPARCReflectionComponent):
    """Example with parameter values not grounded in conversation context."""

    print("\n### Example 2: Parameter Value Grounding Issues ###")
    print(
        "User mentions specific location and phone number, but tool call uses different values\n"
    )

    conversation_context = [
        HumanMessage(
            content="Send an SMS to my mom at +1234567890 saying 'Happy Birthday'"
        ),
    ]

    # Parameter values don't match what user said
    ungrounded_values_call = {
        "id": "2",
        "type": "function",
        "function": {
            "name": "send_sms",
            "arguments": json.dumps(
                {
                    "phone_number": "+9876543210",  # Different from +1234567890 mentioned
                    "message": "Hello there!",  # Different from "Happy Birthday" mentioned
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=MULTI_DOMAIN_TOOL_SPECS,
        tool_calls=[ungrounded_values_call],
    )

    result = await middleware.aprocess(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Parameter Value Grounding Issues", result)


async def run_parameter_hallucination_example(middleware: SPARCReflectionComponent):
    """Example with hallucinated parameter values not mentioned in conversation."""

    print("\n### Example 5: Parameter Hallucination ###")
    print("User asks simple weather question, but tool call adds invented details\n")

    conversation_context = [
        HumanMessage(content="Is it sunny in Miami?"),
        AIMessage(content="I'll check the current weather in Miami for you."),
    ]

    # Tool call adds details not mentioned by user (hallucinated parameters)
    hallucinated_params_call = {
        "id": "5",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps(
                {
                    "location": "Miami Beach, FL, USA, Specific Neighborhood",  # Too specific
                    "units": "kelvin",  # Unusual choice not requested by user
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=MULTI_DOMAIN_TOOL_SPECS,
        tool_calls=[hallucinated_params_call],
    )

    result = await middleware.aprocess(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Parameter Hallucination", result)


async def run_valid_semantic_example(middleware: SPARCReflectionComponent):
    """Example with semantically valid and well-grounded function call."""

    print("\n### Example 6: Valid Semantic Alignment (Reference) ###")
    print(
        "User asks for weather, function and parameters align perfectly with intent\n"
    )

    conversation_context = [
        HumanMessage(
            content="What's the weather like in Boston today? I prefer Fahrenheit."
        ),
        AIMessage(
            content="I'll check the current weather in Boston with Fahrenheit temperature."
        ),
    ]

    # Well-aligned function call that matches user intent and preferences
    valid_semantic_call = {
        "id": "6",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps(
                {
                    "location": "Boston, MA",
                    "units": "fahrenheit",  # Matches user preference
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=MULTI_DOMAIN_TOOL_SPECS,
        tool_calls=[valid_semantic_call],
    )

    result = await middleware.aprocess(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Valid Semantic Alignment", result)


def print_reflection_result(example_name: str, result):
    """Print the reflection result in a readable format."""
    print(f"**{example_name} Result:**")
    print(f"Decision: {result.output.reflection_result.decision}")
    print(f"Execution time: {result.output.execution_time_ms:.2f}ms")

    if result.output.reflection_result.issues:
        print("Issues found:")
        for i, issue in enumerate(result.output.reflection_result.issues, 1):
            print(f"  {i}. [{issue.issue_type}] {issue.metric_name}")
            print(f"     Explanation: {issue.explanation}")
            if issue.correction:
                print(f"     Suggested correction: {issue.correction}")
    else:
        print("No issues found.")

    print("-" * 60)


if __name__ == "__main__":
    # Run actual middleware examples
    asyncio.run(run_semantic_issues_examples())
