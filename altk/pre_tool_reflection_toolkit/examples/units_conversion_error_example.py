import json
import os
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

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


# Example tool specifications
WEATHER_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. 'New York, NY'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units to use",
                    },
                },
                "required": ["location", "units"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "description": "Set the thermostat temperature",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "Target temperature in Celsius",
                    },
                    "location": {"type": "string", "description": "Room or zone name"},
                },
                "required": ["temperature", "location"],
            },
        },
    },
]

DISTANCE_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_travel_time",
            "description": "Calculate travel time between two locations",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_km": {
                        "type": "number",
                        "description": "Distance in kilometers",
                    },
                    "speed_kmh": {
                        "type": "number",
                        "description": "Average speed in kilometers per hour",
                    },
                },
                "required": ["distance_km", "speed_kmh"],
            },
        },
    }
]


def build_config():
    """Build ComponentConfig with WatsonX ValidatingLLMClient."""
    WATSONX_CLIENT = get_llm("watsonx.output_val")
    return ComponentConfig(
        llm_client=WATSONX_CLIENT(
            model_id="meta-llama/llama-3-3-70b-instruct",
            api_key=os.getenv("WX_API_KEY"),
            project_id=os.getenv("WX_PROJECT_ID"),
            url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
        )
    )


async def run_units_conversion_examples():
    """Run examples showing units conversion mistakes."""

    # Initialize middleware using the new ComponentConfig pattern
    # TRANSFORMATIONS_ONLY track is perfect for unit conversion issues
    config = build_config()
    middleware = SPARCReflectionComponent(
        config=config,
        track=Track.TRANSFORMATIONS_ONLY,
        execution_mode=SPARCExecutionMode.ASYNC,
    )

    if middleware._initialization_error:
        print(f"Failed to initialize middleware: {middleware._initialization_error}")
        return

    print("=== Units Conversion Error Examples ===\n")

    # Example 1: Temperature conversion error (Fahrenheit to Celsius)
    await run_temperature_conversion_example(middleware)

    # Example 2: Distance conversion error (Miles to Kilometers)
    await run_distance_conversion_example(middleware)

    # Example 3: Correct conversion (for comparison)
    await run_correct_conversion_example(middleware)


async def run_temperature_conversion_example(middleware: SPARCReflectionComponent):
    """Example with incorrect temperature unit conversion."""

    print("### Example 1: Temperature Conversion Error ###")
    print("User says: 'Set the thermostat to 75 Fahrenheit in the living room'")
    print("Tool expects: Temperature in Celsius")
    print("Wrong conversion: Using 75 directly instead of converting 75°F to ~24°C\n")

    conversation_context = [
        HumanMessage(content="Set the thermostat to 75 Fahrenheit in the living room"),
        AIMessage(
            content="I'll set the thermostat to 75 Fahrenheit in the living room."
        ),
    ]

    # Incorrect tool call - using 75 directly instead of converting F to C
    incorrect_tool_call = {
        "id": "1",
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "arguments": json.dumps(
                {
                    "temperature": 75.0,  # Should be ~24 (75°F = 23.9°C)
                    "location": "living room",
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=WEATHER_TOOL_SPECS,
        tool_calls=[incorrect_tool_call],
    )

    result = await middleware.aprocess(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Temperature Conversion Error", result)


async def run_distance_conversion_example(middleware: SPARCReflectionComponent):
    """Example with incorrect distance unit conversion."""

    print("\n### Example 2: Distance Conversion Error ###")
    print("User says: 'Calculate travel time for 50 miles at 60 mph'")
    print("Tool expects: Distance in km, speed in km/h")
    print("Wrong conversion: Using miles/mph values directly\n")

    conversation_context = [
        HumanMessage(content="Calculate travel time for 50 miles at 60 mph"),
        AIMessage(content="I'll calculate the travel time for 50 miles at 60 mph."),
    ]

    # Incorrect tool call - using miles/mph instead of km/kmh
    incorrect_tool_call = {
        "id": "2",
        "type": "function",
        "function": {
            "name": "calculate_travel_time",
            "arguments": json.dumps(
                {
                    "distance_km": 50.0,  # Should be ~80.5 km (50 miles = 80.47 km)
                    "speed_kmh": 60.0,  # Should be ~96.6 km/h (60 mph = 96.56 km/h)
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=DISTANCE_TOOL_SPECS,
        tool_calls=[incorrect_tool_call],
    )

    result = await middleware.aprocess(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Distance Conversion Error", result)


async def run_correct_conversion_example(middleware: SPARCReflectionComponent):
    """Example with correct unit conversion for comparison."""

    print("\n### Example 3: Correct Conversion (Reference) ###")
    print("User says: 'Set the thermostat to 75 Fahrenheit in the bedroom'")
    print("Correct conversion: 75°F converted to 24°C\n")

    conversation_context = [
        HumanMessage(content="Set the thermostat to 75 Fahrenheit in the bedroom"),
        AIMessage(
            content="I'll set the thermostat to 24 degrees Celsius (75°F) in the bedroom."
        ),
    ]

    # Correct tool call - properly converted F to C
    correct_tool_call = {
        "id": "3",
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "arguments": json.dumps(
                {
                    "temperature": 23.888,  # Correct conversion: 75°F = ~24°C
                    "location": "bedroom",
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=WEATHER_TOOL_SPECS,
        tool_calls=[correct_tool_call],
    )

    result = await middleware.aprocess(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Correct Conversion", result)


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
    asyncio.run(run_units_conversion_examples())
