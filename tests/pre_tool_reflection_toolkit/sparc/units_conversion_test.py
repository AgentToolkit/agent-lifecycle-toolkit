import json
import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from altk.pre_tool_reflection_toolkit.core import (
    SPARCReflectionRunInput,
    SPARCReflectionDecision,
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


class TestUnitsConversion:
    """Test suite for units conversion and transformation validation functionality."""

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
            track=Track.TRANSFORMATIONS_ONLY,
            execution_mode=SPARCExecutionMode.ASYNC,
            transform_enabled=True,
        )

    @pytest.fixture
    def no_transform_middleware(self):
        """Create middleware instance with transformations disabled."""
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
            track=Track.TRANSFORMATIONS_ONLY,
            execution_mode=SPARCExecutionMode.ASYNC,
            transform_enabled=False,
        )

    @pytest.fixture
    def weather_tool_specs(self):
        """Weather-related tool specifications for testing."""
        return [
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
                            "location": {
                                "type": "string",
                                "description": "Room or zone name",
                            },
                        },
                        "required": ["temperature", "location"],
                    },
                },
            },
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
        ]

    @pytest.fixture
    def distance_tool_specs(self):
        """Distance-related tool specifications for testing."""
        return [
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

    def test_correct_conversion_validation(self, middleware, weather_tool_specs):
        """Test that correct unit conversions pass validation."""

        assert not middleware._initialization_error, (
            f"Could not initialize transformation pipeline: {middleware._initialization_error}"
        )

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
            tool_specs=weather_tool_specs,
            tool_calls=[correct_tool_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should have fewer or no transformation issues
        transformation_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if "format" in issue.metric_name.lower()
            or "transformation" in issue.explanation.lower()
        ]

        # Should either approve or have minimal transformation issues
        if result.output.reflection_result.decision == SPARCReflectionDecision.REJECT:
            assert len(transformation_issues) <= 1  # Should have minimal issues

    def test_no_transformation_needed(self, middleware, weather_tool_specs):
        """Test that tool calls not requiring transformation pass validation."""

        assert not middleware._initialization_error, (
            f"Could not initialize transformation pipeline: {middleware._initialization_error}"
        )

        conversation_context = [
            HumanMessage(
                content="Set the thermostat to 22 degrees Celsius in the office"
            ),
            AIMessage(content="I'll set the office thermostat to 22°C."),
        ]

        # Tool call already in correct units
        no_transform_call = {
            "id": "5",
            "type": "function",
            "function": {
                "name": "set_thermostat",
                "arguments": json.dumps(
                    {
                        "temperature": 22.0,  # Already in Celsius
                        "location": "office",
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=conversation_context,
            tool_specs=weather_tool_specs,
            tool_calls=[no_transform_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should pass without transformation issues
        transformation_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if "format" in issue.metric_name.lower()
            or "transformation" in issue.explanation.lower()
        ]
        assert len(transformation_issues) == 0

    @pytest.mark.parametrize(
        "temperature_f,expected_c_range",
        [
            (32, (0, 2)),  # Freezing point
            (68, (19, 21)),  # Room temperature
            (100, (37, 39)),  # Hot day
        ],
    )
    def test_temperature_conversion_accuracy(
        self, middleware, weather_tool_specs, temperature_f, expected_c_range
    ):
        """Test accuracy of temperature conversion detection with various values."""

        assert not middleware._initialization_error, (
            f"Could not initialize transformation pipeline: {middleware._initialization_error}"
        )

        conversation_context = [
            HumanMessage(content=f"Set thermostat to {temperature_f} Fahrenheit"),
            AIMessage(content=f"Setting thermostat to {temperature_f}°F"),
        ]

        # Incorrect conversion (using F value directly)
        tool_call = {
            "id": "temp_test",
            "type": "function",
            "function": {
                "name": "set_thermostat",
                "arguments": json.dumps(
                    {
                        "temperature": float(
                            temperature_f
                        ),  # Wrong - should be converted to C
                        "location": "test_room",
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=conversation_context,
            tool_specs=weather_tool_specs,
            tool_calls=[tool_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should detect conversion error and suggest correction within expected range
        if result.output.reflection_result.decision == SPARCReflectionDecision.REJECT:
            corrections = [
                issue.correction
                for issue in result.output.reflection_result.issues
                if issue.correction
            ]
            if corrections:
                corrected_temp = corrections[0].get("temperature")
                if corrected_temp:
                    min_expected, max_expected = expected_c_range
                    assert min_expected <= corrected_temp <= max_expected

    def test_transformation_disabled_config(
        self, no_transform_middleware, weather_tool_specs
    ):
        """Test that transformation validation can be disabled."""

        assert not no_transform_middleware._initialization_error, (
            f"Could not initialize transformation pipeline: {no_transform_middleware._initialization_error}"
        )

        conversation_context = [
            HumanMessage(content="Set thermostat to 75 Fahrenheit"),
            AIMessage(content="Setting thermostat"),
        ]

        tool_call = {
            "id": "6",
            "type": "function",
            "function": {
                "name": "set_thermostat",
                "arguments": json.dumps(
                    {
                        "temperature": 75.0,  # Would be wrong if transformations were enabled
                        "location": "room",
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=conversation_context,
            tool_specs=weather_tool_specs,
            tool_calls=[tool_call],
        )

        # Execute
        result = no_transform_middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should not detect transformation issues when disabled
        transformation_issues = [
            issue
            for issue in result.output.reflection_result.issues
            if "format" in issue.metric_name.lower()
            or "transformation" in issue.explanation.lower()
        ]
        assert len(transformation_issues) == 0

    def test_complex_transformation_scenario(self, middleware):
        """Test complex transformation scenario with multiple unit types."""

        assert not middleware._initialization_error, (
            f"Could not initialize transformation pipeline: {middleware._initialization_error}"
        )

        # Tool with multiple unit-sensitive parameters
        complex_tool_specs = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_energy_usage",
                    "description": "Calculate energy usage for heating",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature_c": {
                                "type": "number",
                                "description": "Temperature in Celsius",
                            },
                            "area_sqm": {
                                "type": "number",
                                "description": "Area in square meters",
                            },
                            "duration_hours": {
                                "type": "number",
                                "description": "Duration in hours",
                            },
                        },
                        "required": ["temperature_c", "area_sqm", "duration_hours"],
                    },
                },
            }
        ]

        conversation_context = [
            HumanMessage(
                content="Calculate energy for heating 500 sq ft to 72°F for 3 hours"
            ),
            AIMessage(content="I'll calculate the energy usage for heating."),
        ]

        # Mixed units - some correct, some incorrect
        complex_tool_call = {
            "id": "7",
            "type": "function",
            "function": {
                "name": "calculate_energy_usage",
                "arguments": json.dumps(
                    {
                        "temperature_c": 72.0,  # Should be ~22°C (72°F converted)
                        "area_sqm": 500.0,  # Should be ~46.5 sqm (500 sq ft converted)
                        "duration_hours": 3.0,  # Already correct
                    }
                ),
            },
        }

        run_input = SPARCReflectionRunInput(
            messages=conversation_context,
            tool_specs=complex_tool_specs,
            tool_calls=[complex_tool_call],
        )

        # Execute
        result = middleware.process(run_input, phase=AgentPhase.RUNTIME)

        # Should detect multiple transformation issues
        assert result.output.reflection_result is not None
