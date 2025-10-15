"""Example of using ALTK with generic agent to check for pre-tool execution errors.
This example uses the .env file in the root directory.
Copy the .env.example to .env and fill out the following variables:
OPENAI_API_KEY = *** openai api key ***

Note that this example will require installing langgraph and langchain-openai
"""

import re
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from typing import TypedDict, List

from altk.pre_tool_reflection_toolkit.core import (
    SPARCReflectionRunInput,
    Track,
)
from altk.pre_tool_reflection_toolkit.sparc import SPARCReflectionComponent
from altk.toolkit_core.core.toolkit import AgentPhase, ComponentConfig
from altk.toolkit_core.llm import get_llm

from dotenv import load_dotenv

load_dotenv()


# Create SPARC reflector
OPENAI_CLIENT = get_llm("openai.sync.output_val")  # ValidatingLLMClient
config = ComponentConfig(
    llm_client=OPENAI_CLIENT(
        model_name="o4-mini",
    )
)
reflector = SPARCReflectionComponent(config=config, track=Track.FAST_TRACK)

# Tool specification, used by SPARC
tool_specs = [
    {
        "type": "function",
        "function": {
            "name": "send_scheduling_email",
            "description": "Send an email to recipients to schedule an event",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "agenda": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["to", "subject", "body", "agenda"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return weather for city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


@tool
def send_scheduling_email(
    to: str | List[str], subject: str, body: str, agenda: List[str]
) -> str:
    """Send an email to recipients."""
    # NOTE: To simulate incorrect tool calls, there are slight differences between this function and the tool spec
    email_body = body
    email_body += "\n\nAgenda: "
    for agenda_item in agenda:
        email_body += f"\n- {agenda_item}"
    email = email_service(to, subject, email_body)  # type: ignore
    return email


def email_service(to: List[str], subject: str, body: str):  # type: ignore
    # Simulates the email service API call, note the mismatch in the "to" parameter
    # Checks validity of email format in to list
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    for email in to:
        if re.match(email_pattern, email) is None:
            raise ValueError(f"invalid email address: {email}")
    email = "Emails will be sent to {} with the subject {} and body of\n\n{}.".format(
        ",".join(to), subject, body
    )
    return email


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def tool_pre_hook(state):
    # Creates a pre-tool node that runs the reflector, blocks and explains if the input is faulty
    reflect_input = SPARCReflectionRunInput(
        messages=state["messages"],
        tool_specs=tool_specs,
        tool_calls=state["messages"][-1].additional_kwargs["tool_calls"],
    )
    reflect_result = reflector.process(reflect_input, AgentPhase.RUNTIME)
    if reflect_result.output.reflection_result.decision == "approve":
        print("✅ Tool call approved")
        return {"next": "call_tool"}
    else:
        print("❌ Tool call rejected")
        issues = "Tool call rejected for the following reasons:"
        for issue in reflect_result.output.reflection_result.issues:
            issues += f"\n  - {issue.metric_name}: {issue.explanation}"
        print(issues)
        return {"next": "final_message", "messages": [HumanMessage(content=issues)]}


def final_message_node(state):
    return state


tools = [send_scheduling_email]
llm = ChatOpenAI(model="o4-mini")
llm_with_tools = llm.bind_tools(tools, tool_choice="send_scheduling_email")


def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# creates agent with pre-tool node that conditionally goes to tool node
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tool_pre_hook", tool_pre_hook)
builder.add_node("call_tool", ToolNode(tools))
builder.add_node("final_message", final_message_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    lambda state: "tool_pre_hook"
    if state["messages"][-1].tool_calls
    else "final_message",
)
builder.add_conditional_edges(
    "tool_pre_hook",
    lambda state: state["next"],
    {"call_tool": "call_tool", "final_message": "final_message"},
)
builder.add_edge("call_tool", END)
builder.add_edge("final_message", END)
agent = builder.compile()

# Runs the agent, this query should incorrectly parameterize the to field and be blocked
result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="send an email to team@company.com regarding the upcoming meeting"
            )
        ]
    }
)
print(result["messages"][-1].content)
