"""Example of using ALTK with generic agent to check for silent errors.
This example uses the .env file in the root directory.
Copy the .env.example to .env and fill out the following variables:
ALTK_MODEL_NAME = anthropic/claude-sonnet-4-20250514
ANTHROPIC_API_KEY = *** anthropic api key ***

Note that this example will require installing streamilt, langgraph, and langchain-anthropic.
Execute this demo with `streamlit run langgraph_agent_example_streamlit.py`
"""

import random

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState
import streamlit as st

from altk.post_tool.silent_review.silent_review import (
    SilentReviewForJSONDataComponent,
)
from altk.post_tool.core.toolkit import SilentReviewRunInput, Outcome
from altk.core.toolkit import AgentPhase

from dotenv import load_dotenv

load_dotenv()
tool_silent_error_raised = False
silent_error_raised = False
retries = 0


@tool
def get_weather(city: str, state: Annotated[dict, InjectedState]) -> dict[str, str]:
    """Get weather for a given city."""
    global retries
    if random.random() >= (0.500 + retries * 0.25):
        # Simulates a silent error from an external service, less likely if retrying
        result = {"weather": "Weather service is under maintenance."}
        global tool_silent_error_raised
        tool_silent_error_raised = True
    else:
        result = {"weather": f"It's sunny and {random.randint(50, 90)}F in {city}!"}

    if use_silent_review:
        # Use SilentReview component to check if it's a silent error
        review_input = SilentReviewRunInput(
            messages=state["messages"], tool_response=result
        )
        reviewer = SilentReviewForJSONDataComponent()
        review_result = reviewer.process(data=review_input, phase=AgentPhase.RUNTIME)

        if review_result.outcome == Outcome.NOT_ACCOMPLISHED:
            # Agent should retry tool call if silent error was detected
            print("Silent error detected, retry the get_weather tool!")
            global silent_error_raised
            silent_error_raised = True
            retries += 1
            return {
                "weather": "!!! Silent error detected, RETRY the get_weather tool !!!"
            }
        else:
            return result
    else:
        return result


agent = create_react_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    prompt="You are a helpful weather assistant.",
)

st.title("ALTK Chatbot example with Silent Review")
st.markdown(
    "This demo demonstrates using the ALTK to check for silent errors on an agent. The weather service will randomly silently fail. \
            \n- With Silent Error Review, the silent error is detected and then the agent is suggested to retry. \
            \n- Without Silent Review, the agent fails."
)

use_silent_review = st.checkbox("Use Silent Error Review")

if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input(
    "I can tell you the weather in a given city. But my weather service is being spotty..."
):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        inputs = {"messages": [("user", prompt)]}
        result = agent.invoke(inputs)

        if tool_silent_error_raised:
            with st.chat_message("tool"):
                st.write("Weather service: (Weather service is under maintenance.)")

        if silent_error_raised:
            with st.chat_message("altk"):
                st.write(
                    "ALTK: (Silent error detected, suggest agent to retry the get_weather tool.)"
                )

        response = f"Agent response : {result['messages'][-1].content} \n"
        if retries > 0:
            response += f"\n(number of retries: {retries})"
        st_response = st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
