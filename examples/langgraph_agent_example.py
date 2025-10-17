"""Example of using ALTK with generic agent to check for silent errors.
This example uses the .env file in the root directory.
Copy the .env.example to .env and fill out the following variables:
ALTK_MODEL_NAME = anthropic/claude-sonnet-4-20250514
ANTHROPIC_API_KEY = *** anthropic api key ***

Note that this example will require installing langgraph, and langchain-anthropic.
"""

import random

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

from altk.post_tool.silent_review.silent_review import (
    SilentReviewForJSONDataComponent,
)
from altk.post_tool.core.toolkit import SilentReviewRunInput, Outcome
from altk.core.toolkit import AgentPhase

from dotenv import load_dotenv

load_dotenv()
retries = 0


@tool
def get_weather(city: str, state: Annotated[dict, InjectedState]) -> dict[str, str]:
    """Get weather for a given city."""
    global retries
    if random.random() >= (0.500 + retries * 0.25):
        # Simulates a silent error from an external service, less likely if retrying
        result = {"weather": "Weather service is under maintenance."}
    else:
        result = {"weather": f"It's sunny and {random.randint(50, 90)}F in {city}!"}

    # Use SilentReview component to check if it's a silent error
    review_input = SilentReviewRunInput(
        messages=state["messages"], tool_response=result
    )
    reviewer = SilentReviewForJSONDataComponent()
    review_result = reviewer.process(data=review_input, phase=AgentPhase.RUNTIME)

    if review_result.outcome == Outcome.NOT_ACCOMPLISHED:
        # Agent should retry tool call if silent error was detected
        print("(ALTK: Silent error detected, retry the get_weather tool!)")
        retries += 1
        return {"weather": "!!! Silent error detected, RETRY the get_weather tool !!!"}
    else:
        return result


agent = create_react_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

# Runs the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
print(result["messages"][-1].content)
if retries > 0:
    print(f"(get_weather was retried: {retries} times)")
