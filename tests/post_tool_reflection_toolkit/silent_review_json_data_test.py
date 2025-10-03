import os
from langchain_core.messages import HumanMessage, AIMessage
from altk.toolkit_core.core.toolkit import AgentPhase, ComponentConfig
from altk.toolkit_core.llm.base import get_llm
from altk.post_tool_reflection_toolkit.core.toolkit import SilentReviewRunInput
from altk.post_tool_reflection_toolkit.silent_review.silent_review import (
    SilentReviewForJSONDataComponent,
)
import pytest


def build_test_input() -> SilentReviewRunInput:
    return SilentReviewRunInput(
        messages=[
            HumanMessage(content="Tell me the weather"),
            AIMessage(content="Calling the weather tool now"),
        ],
        toolspec={
            "name": "get_weather",
            "description": "Gets weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        tool_response={
            "name": "get_weather",
            "result": {"city": "NYC", "temperature": "75F", "condition": "Sunny"},
        },
    )


def build_config():
    WATSONX_CLIENT = get_llm("watsonx")
    return ComponentConfig(
        llm_client=WATSONX_CLIENT(
            model_id="meta-llama/llama-3-3-70b-instruct",
            api_key=os.getenv("WX_API_KEY"),
            project_id=os.getenv("WX_PROJECT_ID"),
            url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
        )
    )


def test_silent_review_json():
    config = build_config()
    data = build_test_input()
    middleware = SilentReviewForJSONDataComponent(config=config)

    result = middleware.process(data=data, phase=AgentPhase.RUNTIME)
    assert result.outcome.value == 1.0


@pytest.mark.asyncio
async def test_silent_review_json_async():
    config = build_config()
    data = build_test_input()
    middleware = SilentReviewForJSONDataComponent(config=config)

    result = await middleware.aprocess(data=data, phase=AgentPhase.RUNTIME)
    assert result.outcome.value == 1.0
