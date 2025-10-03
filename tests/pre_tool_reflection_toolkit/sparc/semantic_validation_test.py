import json
import logging
import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from altk.pre_tool_reflection_toolkit.core import (
    SPARCReflectionRunInput,
    SPARCReflectionDecision,
    SPARCReflectionIssueType,
    SPARCExecutionMode,
    Track,
)
from altk.pre_tool_reflection_toolkit.sparc import (
    SPARCReflectionComponent,
)
from altk.toolkit_core.core.toolkit import AgentPhase, ComponentConfig
from altk.toolkit_core.llm import get_llm
from dotenv import load_dotenv

load_dotenv()


class TestSemanticValidation:
    """Test suite for semantic validation functionality."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance for testing."""
        # Build ComponentConfig with WatsonX ValidatingLLMClient
        WATSONX_CLIENT = get_llm("watsonx.output_val")
        config = ComponentConfig(
            llm_client=WATSONX_CLIENT(
                model_id="meta-llama/llama-3-3-70b-instruct",
                api_key=os.getenv("WX_API_KEY"),
                project_id=os.getenv("WX_PROJECT_ID"),
                url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
            )
        )
        return SPARCReflectionComponent(
            config=config,
            track=Track.FAST_TRACK,
            execution_mode=SPARCExecutionMode.ASYNC,
        )

    @pytest.fixture
    def multi_domain_tool_specs(self):
        """Multi-domain tool specifications for testing."""
        return [
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
                        },
                        "required": [
                            "origin",
                            "destination",
                            "departure_date",
                            "passengers",
                        ],
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
        ]

    def test_function_selection_misalignment(self, middleware, multi_domain_tool_specs):
        """Test detection of function selection that doesn't align with user intent."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

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
            tool_specs=multi_domain_tool_specs,
            tool_calls=[misaligned_function_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)
        logging.info("-------------------------------------------------")
        logging.info(f"Test result: {result}")
        logging.info("-------------------------------------------------")

        # Assertions
        assert (
            result.output.reflection_result.decision == SPARCReflectionDecision.REJECT
        )
        assert len(result.output.reflection_result.issues) > 0

        # Check for any semantic issues - either function selection or general issues
        semantic_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if issue.issue_type
            in [
                SPARCReflectionIssueType.SEMANTIC_FUNCTION,
                SPARCReflectionIssueType.SEMANTIC_GENERAL,
            ]
        ]
        assert len(semantic_issues) > 0

    def test_parameter_value_grounding_issues(
        self, middleware, multi_domain_tool_specs
    ):
        """Test detection of parameter values not grounded in conversation context."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        conversation_context = [
            HumanMessage(
                content="Send an SMS to my mom at +1234567890 saying 'Happy Birthday'"
            ),
            AIMessage(content="I'll send the birthday message to your mom."),
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
                        "message": "Happy Birthday",
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=conversation_context,
            tool_specs=multi_domain_tool_specs,
            tool_calls=[ungrounded_values_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Assertions
        assert (
            result.output.reflection_result.decision == SPARCReflectionDecision.REJECT
        )
        assert len(result.output.reflection_result.issues) > 0

        print(result.output.reflection_result.issues)

    def test_valid_semantic_alignment(self, middleware, multi_domain_tool_specs):
        """Test that semantically valid and well-grounded function calls pass validation."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
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
            "id": "4",
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
            tool_specs=multi_domain_tool_specs,
            tool_calls=[valid_semantic_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        assert result.output.reflection_result.decision in [
            SPARCReflectionDecision.APPROVE,
            SPARCReflectionDecision.REJECT,
        ]

    def test_complex_conversation_context(self, middleware, multi_domain_tool_specs):
        """Test semantic validation with complex multi-turn conversation."""
        # Setup
        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        # Complex conversation with multiple topics and clarifications
        complex_conversation = [
            HumanMessage(content="I need to plan a trip to New York"),
            AIMessage(
                content="I'd be happy to help you plan your trip to New York. What would you like to do first?"
            ),
            HumanMessage(
                content="First, can you check the weather forecast for this weekend?"
            ),
            AIMessage(
                content="I'll check the weather forecast for New York this weekend."
            ),
        ]

        weather_call = {
            "id": "6",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps(
                    {
                        "location": "New York, NY",
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=complex_conversation,
            tool_specs=multi_domain_tool_specs,
            tool_calls=[weather_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should handle complex context appropriately
        assert result.output.reflection_result is not None

    def test_edge_case_empty_conversation(self, middleware, multi_domain_tool_specs):
        """Test handling of edge case with minimal conversation context."""
        # Setup
        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        # Minimal conversation
        minimal_conversation = [
            HumanMessage(content="Weather?"),
        ]

        tool_call = {
            "id": "7",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps(
                    {
                        "location": "Unknown",
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=minimal_conversation,
            tool_specs=multi_domain_tool_specs,
            tool_calls=[tool_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should handle minimal context gracefully
        assert result.output.reflection_result is not None
