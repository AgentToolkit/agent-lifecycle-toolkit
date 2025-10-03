import os
import json
from dotenv import load_dotenv
from typing import cast
from langchain_core.messages import HumanMessage, AIMessage
from altk.pre_tool_reflection_toolkit.core.types import SPARCReflectionRunOutput

# Import middleware components
from altk.pre_tool_reflection_toolkit.core import (
    SPARCReflectionConfig,
    SPARCExecutionMode,
    SPARCReflectionRunInput,
)
from altk.pre_tool_reflection_toolkit.sparc import (
    SPARCReflectionComponent,
)
from altk.toolkit_core.core.toolkit import ComponentConfig, AgentPhase
from altk.toolkit_core.llm import get_llm

# Import available metrics
from altk.pre_tool_reflection_toolkit.core.consts import (
    METRIC_GENERAL_HALLUCINATION_CHECK,
    METRIC_FUNCTION_SELECTION_APPROPRIATENESS,
    METRIC_AGENTIC_CONSTRAINTS_SATISFACTION,
    METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT,
    METRIC_PARAMETER_HALLUCINATION_CHECK,
)

# Load environment variables
load_dotenv()


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


def run_custom_config_examples():
    """Run examples with different custom metric configurations."""

    print("=== Custom Metric Configuration Examples ===\n")

    # Build the middleware config
    config = build_config()

    # Example 1: Only function selection validation
    print("### Example 1: Function Selection Only ###")
    print("Custom config with only function selection appropriateness validation")

    custom_config_1 = SPARCReflectionConfig(
        general_metrics=None,
        function_metrics=[METRIC_FUNCTION_SELECTION_APPROPRIATENESS],
        parameter_metrics=None,
    )

    middleware_1 = SPARCReflectionComponent(
        config=config,
        custom_config=custom_config_1,
        execution_mode=SPARCExecutionMode.ASYNC,
    )

    # Test with function selection misalignment
    conversation_context = [
        HumanMessage(content="What's the weather like in New York today?"),
        AIMessage(content="I'll check the weather for you."),
    ]

    tool_specs = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get weather for",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Book a flight ticket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["origin", "destination"],
                },
            },
        },
    ]

    # Wrong function selected
    wrong_function_call = {
        "id": "1",
        "type": "function",
        "function": {
            "name": "book_flight",  # Should be get_weather
            "arguments": json.dumps({"origin": "NYC", "destination": "LAX"}),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=tool_specs,
        tool_calls=[wrong_function_call],
    )

    result = cast(
        SPARCReflectionRunOutput,
        middleware_1.process(run_input, phase=AgentPhase.RUNTIME),
    )
    print("**Function Selection Only Result:**")
    print(f"Decision: {result.output.reflection_result.decision}")
    print(f"Execution time: {result.output.execution_time_ms:.2f}ms")
    if result.output.reflection_result.issues:
        print("Issues found:")
        for issue in result.output.reflection_result.issues:
            print(f"  - [{issue.issue_type}] {issue.metric_name}")
            print(f"    Explanation: {issue.explanation}")
    else:
        print("No issues found.")
    print("-" * 60)

    # Example 2: Comprehensive parameter validation
    print("\n### Example 2: Comprehensive Parameter Validation ###")
    print("Custom config with all parameter-level metrics")

    custom_config_2 = SPARCReflectionConfig(
        general_metrics=[METRIC_GENERAL_HALLUCINATION_CHECK],
        function_metrics=None,
        parameter_metrics=[
            METRIC_PARAMETER_HALLUCINATION_CHECK,
            METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT,
        ],
    )

    middleware_2 = SPARCReflectionComponent(
        config=config,
        custom_config=custom_config_2,
        execution_mode=SPARCExecutionMode.ASYNC,
    )

    # Test with parameter hallucination
    hallucinated_call = {
        "id": "2",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps(
                {
                    "location": "New York, NY, USA, Manhattan, Upper East Side, 5th Avenue"  # Too specific
                }
            ),
        },
    }

    run_input_2 = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=tool_specs,
        tool_calls=[hallucinated_call],
    )

    result_2 = cast(
        SPARCReflectionRunOutput,
        middleware_2.process(run_input_2, phase=AgentPhase.RUNTIME),
    )
    print("**Parameter Validation Result:**")
    print(f"Decision: {result_2.output.reflection_result.decision}")
    print(f"Execution time: {result_2.output.execution_time_ms:.2f}ms")
    if result_2.output.reflection_result.issues:
        print("Issues found:")
        for issue in result_2.output.reflection_result.issues:
            print(f"  - [{issue.issue_type}] {issue.metric_name}")
            print(f"    Explanation: {issue.explanation}")
    else:
        print("No issues found.")
    print("-" * 60)

    # Example 3: Minimal agentic validation
    print("\n### Example 3: Minimal Agentic Validation ###")
    print("Custom config with only agentic constraints satisfaction")

    custom_config_3 = SPARCReflectionConfig(
        general_metrics=None,
        function_metrics=[METRIC_AGENTIC_CONSTRAINTS_SATISFACTION],
        parameter_metrics=None,
    )

    middleware_3 = SPARCReflectionComponent(
        config=config,
        custom_config=custom_config_3,
        execution_mode=SPARCExecutionMode.ASYNC,
    )

    # Test with valid call
    valid_call = {
        "id": "3",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps({"location": "New York"}),
        },
    }

    run_input_3 = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=tool_specs,
        tool_calls=[valid_call],
    )

    result_3 = cast(
        SPARCReflectionRunOutput,
        middleware_3.process(run_input_3, phase=AgentPhase.RUNTIME),
    )
    print("**Agentic Validation Result:**")
    print(f"Decision: {result_3.output.reflection_result.decision}")
    print(f"Execution time: {result_3.output.execution_time_ms:.2f}ms")
    if result_3.output.reflection_result.issues:
        print("Issues found:")
        for issue in result_3.output.reflection_result.issues:
            print(f"  - [{issue.issue_type}] {issue.metric_name}")
            print(f"    Explanation: {issue.explanation}")
    else:
        print("No issues found.")
    print("-" * 60)


if __name__ == "__main__":
    run_custom_config_examples()
