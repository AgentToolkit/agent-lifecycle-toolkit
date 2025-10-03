import json
import os
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
EMAIL_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to one or more recipients",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "email",
                            "pattern": "^\\S+@\\S+\\.\\S+$",
                        },
                        "description": "List of recipient email addresses",
                        "minItems": 1,
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content",
                        "minLength": 1,
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "description": "Email priority level",
                        "default": "normal",
                    },
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to attach",
                        "default": [],
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with participants",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Meeting title",
                        "minLength": 1,
                        "maxLength": 100,
                    },
                    "participants": {
                        "type": "array",
                        "items": {"type": "string", "format": "email"},
                        "description": "List of participant email addresses",
                        "minItems": 1,
                    },
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Meeting start time in ISO 8601 format",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Meeting duration in minutes",
                        "minimum": 15,
                        "maximum": 480,
                    },
                    "location": {
                        "type": "string",
                        "description": "Meeting location or video conference link",
                    },
                },
                "required": ["title", "participants", "start_time", "duration_minutes"],
            },
        },
    },
]

CALCULATION_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_compound_interest",
            "description": "Calculate compound interest for an investment",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "Initial investment amount",
                        "minimum": 0.01,
                    },
                    "rate": {
                        "type": "number",
                        "description": "Annual interest rate as a decimal (e.g., 0.05 for 5%)",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "time_years": {
                        "type": "integer",
                        "description": "Investment period in years",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "compound_frequency": {
                        "type": "integer",
                        "description": "Number of times interest is compounded per year",
                        "enum": [1, 2, 4, 12, 365],
                    },
                },
                "required": ["principal", "rate", "time_years", "compound_frequency"],
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


def run_static_issues_examples():
    """Run examples showing static validation issues."""

    # Initialize middleware using the new ComponentConfig pattern
    # SYNTAX track is perfect for static validation as it doesn't need a model
    config = build_config()
    middleware = SPARCReflectionComponent(
        config=config,
        track=Track.SYNTAX,
        execution_mode=SPARCExecutionMode.ASYNC,
        verbose_logging=True,
        continue_on_static=False,  # Stop on static failures
    )

    if middleware._initialization_error:
        print(f"Failed to initialize middleware: {middleware._initialization_error}")
        return

    print("=== Static Validation Issues Examples ===\n")

    # Example 1: Missing required parameters
    run_missing_parameters_example(middleware)

    # Example 2: Invalid parameter types
    run_invalid_types_example(middleware)

    # Example 4: Schema constraint violations
    run_schema_violations_example(middleware)

    # Example 5: Valid call (for comparison)
    run_valid_static_example(middleware)


def run_missing_parameters_example(middleware: SPARCReflectionComponent):
    """Example with missing required parameters."""

    print("### Example 1: Missing Required Parameters ###")
    print("Tool call missing required 'subject' and 'body' parameters\n")

    conversation_context = [
        HumanMessage(content="Send an email to john@example.com"),
        AIMessage(content="I'll send an email to john@example.com"),
    ]

    # Missing required parameters: subject and body
    missing_params_call = {
        "id": "1",
        "type": "function",
        "function": {
            "name": "send_email",
            "arguments": json.dumps(
                {
                    "to": ["john@example.com"],
                    # Missing required 'subject' and 'body'
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=EMAIL_TOOL_SPECS,
        tool_calls=[missing_params_call],
    )

    result = middleware.process(run_input, AgentPhase.RUNTIME)
    print_reflection_result("Missing Required Parameters", result)


def run_invalid_types_example(middleware: SPARCReflectionComponent):
    """Example with invalid parameter types."""

    print("\n### Example 2: Invalid Parameter Types ###")
    print(
        "Tool call with incorrect parameter types (string instead of array, number instead of string)\n"
    )

    conversation_context = [
        HumanMessage(
            content="Schedule a 2-hour meeting with the team tomorrow at 2 PM"
        ),
        AIMessage(content="I'll schedule a 2-hour meeting with the team"),
    ]

    # Invalid types: participants should be array, duration_minutes should be integer
    invalid_types_call = {
        "id": "2",
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "arguments": json.dumps(
                {
                    "title": "Team Meeting",
                    "participants": "team@example.com",  # Should be array
                    "start_time": "2024-06-21T14:00:00Z",
                    "duration_minutes": "120",  # Should be integer
                    "location": "Conference Room A",
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=EMAIL_TOOL_SPECS,
        tool_calls=[invalid_types_call],
    )

    result = middleware.process(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Invalid Parameter Types", result)


def run_schema_violations_example(middleware: SPARCReflectionComponent):
    """Example with schema constraint violations."""

    print("\n### Example 3: Schema Constraint Violations ###")
    print(
        "Tool call with values that violate schema constraints (min/max, enum, format)\n"
    )

    conversation_context = [
        HumanMessage(content="Send a high priority email to invalid-email"),
        AIMessage(content="I'll send a high priority email"),
    ]

    # Schema violations: invalid email format, invalid priority enum, empty subject
    schema_violations_call = {
        "id": "4",
        "type": "function",
        "function": {
            "name": "send_email",
            "arguments": json.dumps(
                {
                    "to": ["not-an-email"],  # Invalid email format
                    "subject": "",  # Violates minLength: 1
                    "body": "This is the email body",
                    "priority": "urgent",  # Not in enum: ["low", "normal", "high"]
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=EMAIL_TOOL_SPECS,
        tool_calls=[schema_violations_call],
    )

    result = middleware.process(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Schema Constraint Violations", result)


def run_valid_static_example(middleware: SPARCReflectionComponent):
    """Example with valid static structure for comparison."""

    print("\n### Example 4: Valid Static Structure (Reference) ###")
    print("Tool call with all required parameters and correct types\n")

    conversation_context = [
        HumanMessage(
            content="Send an email to team@example.com with subject 'Weekly Update'"
        ),
        AIMessage(content="I'll send the weekly update email to the team"),
    ]

    # Valid tool call
    valid_call = {
        "id": "5",
        "type": "function",
        "function": {
            "name": "send_email",
            "arguments": json.dumps(
                {
                    "to": ["team@example.com"],
                    "subject": "Weekly Update",
                    "body": "Here's this week's progress update for the team.",
                    "priority": "normal",
                }
            ),
        },
    }

    run_input = SPARCReflectionRunInput(
        messages=conversation_context,
        tool_specs=EMAIL_TOOL_SPECS,
        tool_calls=[valid_call],
    )

    result = middleware.process(run_input, phase=AgentPhase.RUNTIME)
    print_reflection_result("Valid Static Structure", result)


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
    run_static_issues_examples()
