"""Example of using ALTK with generic agent to check for silent errors.
This example uses the .env file in the root directory.
Copy the .env.example to .env and fill out the following variables:
LLM_PROVIDER = openai.sync
MODEL_NAME = o4-mini
OPENAI_API_KEY = *** openai api key ***
"""

import random

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

from altk.post_tool_reflection_toolkit.silent_review.silent_review import (
    SilentReviewForJSONDataComponent,
)
from altk.post_tool_reflection_toolkit.core.toolkit import SilentReviewRunInput, Outcome
from altk.toolkit_core.core.toolkit import AgentPhase

from dotenv import load_dotenv

load_dotenv()


@tool
def get_weather(city: str, state: Annotated[dict, InjectedState]) -> str:
    """Get weather for a given city."""
    if random.random() >= 0.500:
        # Simulates a silent error from an external service
        result = {"weather": "Weather service is under maintenance."}
    else:
        result = {"weather": f"It's sunny and 70F in {city}!"}

    # Use SilentReview component to check if it's a silent error
    review_input = SilentReviewRunInput(
        messages=state["messages"], tool_response=result
    )
    reviewer = SilentReviewForJSONDataComponent()
    review_result = reviewer.process(data=review_input, phase=AgentPhase.RUNTIME)

    if review_result.outcome != Outcome.ACCOMPLISHED:
        # Agent should retry tool call if silent error was detected
        print("Silent error detected, retry the get_weather tool!")
        return "Silent error detected, retry the get_weather tool!"
    else:
        return result


agent = create_react_agent(
    model="openai:o4-mini", tools=[get_weather], prompt="You are a helpful assistant"
)

# Runs the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
# Show the final result which should not be that the service is in maintenance.
print(result["messages"][-1].content)
