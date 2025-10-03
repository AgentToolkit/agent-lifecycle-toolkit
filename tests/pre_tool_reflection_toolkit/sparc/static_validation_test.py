import json
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


class TestStaticValidation:
    """Test suite for static validation functionality."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance for testing."""
        # Build ComponentConfig with WatsonX ValidatingLLMClient
        WATSONX_CLIENT = get_llm("watsonx.output_val")
        config = ComponentConfig(
            llm_client=WATSONX_CLIENT(
                model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
                api_key=os.getenv("WX_API_KEY"),
                project_id=os.getenv("WX_PROJECT_ID"),
                url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
            )
        )
        return SPARCReflectionComponent(
            config=config,
            track=Track.SYNTAX,
            execution_mode=SPARCExecutionMode.ASYNC,
            continue_on_static=False,
        )

    @pytest.fixture
    def email_tool_specs(self):
        """Email tool specifications for testing."""
        return [
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
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
            }
        ]

    @pytest.fixture
    def meeting_tool_specs(self):
        """Meeting tool specifications for testing."""
        return [
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
                        },
                        "required": [
                            "title",
                            "participants",
                            "start_time",
                            "duration_minutes",
                        ],
                    },
                },
            }
        ]

    @pytest.fixture
    def basic_conversation(self):
        """Basic conversation context for testing."""
        return [
            HumanMessage(content="Send an email to john@example.com"),
            AIMessage(content="I'll send an email to john@example.com"),
        ]

    def test_missing_required_parameters(
        self, middleware, email_tool_specs, basic_conversation
    ):
        """Test detection of missing required parameters."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        # Test missing required parameters
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
            messages=basic_conversation,
            tool_specs=email_tool_specs,
            tool_calls=[missing_params_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Assertions
        assert (
            result.output.reflection_result.decision == SPARCReflectionDecision.REJECT
        )
        assert len(result.output.reflection_result.issues) > 0

        # Check for static validation issues
        static_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if issue.issue_type == SPARCReflectionIssueType.STATIC
        ]
        assert len(static_issues) > 0

        # Verify the specific missing parameters are detected
        issue_explanations = " ".join([issue.explanation for issue in static_issues])
        assert "subject" in issue_explanations.lower()
        assert "body" in issue_explanations.lower()

    def test_invalid_parameter_types(self, middleware, meeting_tool_specs):
        """Test detection of invalid parameter types."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        conversation = [
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
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=conversation,
            tool_specs=meeting_tool_specs,
            tool_calls=[invalid_types_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Assertions
        assert (
            result.output.reflection_result.decision == SPARCReflectionDecision.REJECT
        )
        assert len(result.output.reflection_result.issues) > 0

        static_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if issue.issue_type == SPARCReflectionIssueType.STATIC
        ]
        assert len(static_issues) > 0

    def test_schema_constraint_violations(self, middleware, email_tool_specs):
        """Test detection of schema constraint violations."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        conversation = [
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
            messages=conversation,
            tool_specs=email_tool_specs,
            tool_calls=[schema_violations_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Assertions
        assert (
            result.output.reflection_result.decision == SPARCReflectionDecision.REJECT
        )
        assert len(result.output.reflection_result.issues) > 0

        static_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if issue.issue_type == SPARCReflectionIssueType.STATIC
        ]
        assert len(static_issues) > 0

    def test_valid_static_structure(self, middleware, email_tool_specs):
        """Test that valid structure passes static validation."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        conversation = [
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
            messages=conversation,
            tool_specs=email_tool_specs,
            tool_calls=[valid_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Assertions - should approve or only have non-static issues
        static_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if issue.issue_type == SPARCReflectionIssueType.STATIC
        ]
        assert len(static_issues) == 0  # No static validation issues

    def test_malformed_json_arguments(
        self, middleware, email_tool_specs, basic_conversation
    ):
        """Test handling of malformed JSON in tool call arguments."""

        assert not middleware._initialization_error, (
            f"Initialization failed: {middleware._initialization_error}"
        )

        # Malformed JSON in arguments
        malformed_call = {
            "id": "6",
            "type": "function",
            "function": {
                "name": "send_email",
                "arguments": '{"to": ["test@example.com"], "subject": "Test", "body": "Test", invalid_json}',
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=basic_conversation,
            tool_specs=email_tool_specs,
            tool_calls=[malformed_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should detect JSON parsing error
        assert result.output.reflection_result.decision == SPARCReflectionDecision.ERROR
        assert len(result.output.reflection_result.issues) > 0
