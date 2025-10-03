from altk.toolkit_core.llm.types import GenerationMode
from altk.toolkit_core.llm import LLMClient
from altk.toolkit_core.core.toolkit import AgentPhase
from altk.pre_tool_reflection_toolkit.core.config import (
    RefractionConfig,
    RefractionMode,
)
from altk.pre_tool_reflection_toolkit.refraction.refraction import RefractionComponent
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
from altk.pre_tool_reflection_toolkit.core.types import (
    RefractionBuildInput,
    RefractionRunInput,
    RefractionRunOutput,
)

from typing import Any, Type, cast

import pytest
import json


class DummyLLMClient(LLMClient):
    def generate(self, *args, **kwargs) -> str:
        return '<tool_call>[{"name": "book_flight", "arguments": {"origin": "JFK", "destination": "LAX", "passengers": 1}, "label": "None"}]</tool_call>'

    @classmethod
    def provider_class(cls) -> Type:
        return Any  # type: ignore

    def _register_methods(self) -> None:
        """
        Register how to call watsonx methods:

        - 'text'       -> ModelInference.generate
        - 'text_async' -> ModelInference.agenerate
        - 'chat'       -> ModelInference.chat
        - 'chat_async' -> ModelInference.achat
        """
        self.set_method_config(GenerationMode.TEXT.value, "generate", "prompt")
        self.set_method_config(GenerationMode.TEXT_ASYNC.value, "agenerate", "prompt")
        self.set_method_config(GenerationMode.CHAT.value, "chat", "messages")
        self.set_method_config(GenerationMode.CHAT_ASYNC.value, "achat", "messages")

    def _parse_llm_response(self, raw: Any) -> str:
        return ""

    def _setup_parameter_mapper(self) -> None:
        """
        Setup parameter mapping for the provider. Override in subclasses to configure
        mapping from generic GenerationArgs to provider-specific parameters.
        """
        return None


class TestRefraction:
    """Test suite for semantic validation functionality."""

    @pytest.fixture
    def middleware(self) -> RefractionComponent:
        """Create middleware instance for testing."""
        return RefractionComponent()

    @pytest.fixture
    def tool_specs(self) -> list[dict[str, Any]]:
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

    def test_mappings_is_created_after_build(
        self, middleware: RefractionComponent, tool_specs: list[dict[str, Any]]
    ):
        build_input = RefractionBuildInput(
            tool_specs=tool_specs,
        )

        result = cast(
            Refractor, middleware.process(build_input, phase=AgentPhase.BUILDTIME)
        )

        assert result.mappings is not None

    def test_catalog_is_created_after_build(
        self, middleware: RefractionComponent, tool_specs: list[dict[str, Any]]
    ):
        build_input = RefractionBuildInput(
            tool_specs=tool_specs,
        )

        result = cast(
            Refractor, middleware.process(build_input, phase=AgentPhase.BUILDTIME)
        )

        assert result.catalog

    def test_determination_is_false_when_tool_calls_are_invalid(
        self, middleware: RefractionComponent, tool_specs: list[dict[str, Any]]
    ):
        invalid_tool_call = {
            "id": "1",
            "type": "function",
            "function": {
                "name": "book_flight",
                "arguments": json.dumps(
                    {
                        "origin": "JFK",
                        "destination": "LAX",
                        "passengers": 1,
                        # departure date is a missing parameter
                    }
                ),
            },
        }

        # Execute
        run_input = RefractionRunInput(
            tool_calls=[invalid_tool_call],
            tool_specs=tool_specs,
            memory_objects={},
            use_given_operators_only=True,
        )

        result = cast(
            RefractionRunOutput, middleware.process(run_input, phase=AgentPhase.RUNTIME)
        )

        assert result.result is not None
        assert result.result.report.determination is False

    def test_determination_is_true_when_tool_calls_are_valid(
        self, middleware: RefractionComponent, tool_specs: list[dict[str, Any]]
    ):
        valid_tool_call = {
            "id": "1",
            "type": "function",
            "function": {
                "name": "book_flight",
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

        run_input = RefractionRunInput(
            tool_calls=[valid_tool_call],
            tool_specs=tool_specs,
            memory_objects={},
            use_given_operators_only=True,
        )

        result = cast(
            RefractionRunOutput, middleware.process(run_input, phase=AgentPhase.RUNTIME)
        )

        assert result.result.report.determination is True

    def test_standalone_mode_does_not_call_llm(
        self, mocker, middleware: RefractionComponent, tool_specs: list[dict[str, Any]]
    ):
        llm_client = DummyLLMClient()
        spy = mocker.spy(llm_client, "generate")
        config = RefractionConfig(llm_client=llm_client, mode=RefractionMode.STANDALONE)

        middleware = RefractionComponent(config=config)

        invalid_tool_call = {
            "id": "1",
            "type": "function",
            "function": {
                "name": "book_flight",
                "arguments": json.dumps(
                    {
                        "origin": "JFK",
                        "destination": "LAX",
                        "passengers": 1,
                        # departure date is a missing parameter
                    }
                ),
            },
        }

        # Execute
        run_input = RefractionRunInput(
            tool_calls=[invalid_tool_call],
            tool_specs=tool_specs,
            memory_objects={},
            use_given_operators_only=True,
        )
        result = cast(
            RefractionRunOutput, middleware.process(run_input, phase=AgentPhase.RUNTIME)
        )

        assert result.result is not None
        assert spy.call_count == 0

    def test_with_llm_mode_calls_llm(
        self, mocker, middleware: RefractionComponent, tool_specs: list[dict[str, Any]]
    ):
        llm_client = DummyLLMClient()
        spy = mocker.spy(llm_client, "generate")
        config = RefractionConfig(llm_client=llm_client, mode=RefractionMode.WITH_LLM)

        middleware = RefractionComponent(config=config)

        invalid_tool_call = {
            "id": "1",
            "type": "function",
            "function": {
                "name": "book_flight",
                "arguments": json.dumps(
                    {
                        "origin": "JFK",
                        "destination": "LAX",
                        "passengers": 1,
                        # departure date is a missing parameter
                    }
                ),
            },
        }

        # Execute
        run_input = RefractionRunInput(
            tool_calls=[invalid_tool_call],
            tool_specs=tool_specs,
            memory_objects={},
            use_given_operators_only=True,
        )
        result = cast(
            RefractionRunOutput, middleware.process(run_input, phase=AgentPhase.RUNTIME)
        )

        assert result.result is not None
        assert spy.call_count != 0
