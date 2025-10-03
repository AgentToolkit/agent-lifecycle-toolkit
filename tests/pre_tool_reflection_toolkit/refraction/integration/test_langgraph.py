from typing import List
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
from nestful.data_handlers import get_nestful_catalog

try:
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolNode, create_react_agent
    from langgraph.graph import StateGraph, START, END
except ImportError:
    pass
from tests.utils.refraction.langgraph.utils import execute_tool_calls
from tests.utils.refraction.langgraph.rits_model import RITSModel
from tests.utils.refraction.tools.custom_tools_langgraph import (
    State,
    search_hotels,
    TripadvisorSearchLocation,
)

import pytest

pytestmark = pytest.mark.refract_extra


class TestLangGraph:
    def setup_method(self) -> None:
        catalog = get_nestful_catalog(executable=True)
        self.refractor = Refractor(catalog=catalog)

        self.tools: List[BaseTool] = [
            search_hotels,
            TripadvisorSearchLocation,
        ]

        self.tool_node = ToolNode(self.tools)

        self.model = RITSModel(
            model_name="ibm-granite/granite-3.3-8b-instruct",
            model_url="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-3-8b-instruct",
        )

    def test_direct_tool_call(self) -> None:
        message_with_single_tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "TripadvisorSearchLocation",
                    "args": {"query": "San Juan", "refractor": self.refractor},
                    "id": "5241421",
                    "type": "tool_call",
                }
            ],
        )

        result = self.tool_node.invoke({"messages": [message_with_single_tool_call]})

        messages: List[ToolMessage] = result.get("messages", [])
        assert messages[0].content == "San Juan: locations"

    @pytest.mark.skip(reason="I don't think this is possible")
    def test_chat_model(self) -> None:
        self.model.bind_tools(self.tools)

        result = self.model.invoke(
            [
                AIMessage(content="Search hotels"),
            ]
        )

        assert result.tool_calls == [
            {
                "args": {"query": "hotels"},
                "id": "0",
                "name": "TripadvisorSearchLocation",
                "type": "tool_call",
            }
        ]

        responses = execute_tool_calls(result.tool_calls, self.tools)
        assert responses[0]["var1"]["geoId"] == 123

    @pytest.mark.skip(reason="I don't think this is possible")
    def test_agentic(self) -> None:
        agent = create_react_agent(
            model=self.model,
            tools=self.tools,
        )

        result = agent.invoke(
            input={"messages": "search hotels", "refractor": self.refractor},
            config={"recursion_limit": 4},
        )

        tool_response: ToolMessage = result["messages"][2]
        assert tool_response.content == "hotel123"

    @pytest.mark.skip(reason="I don't think this is possible")
    def test_agentic_with_argument_usage(self) -> None:
        raise NotImplementedError()

    def test_workflow(self) -> None:
        workflow = StateGraph(State)
        workflow.add_node("search_hotels_node", search_hotels)

        workflow.add_edge(start_key=START, end_key="search_hotels_node")
        workflow.add_edge(start_key="search_hotels_node", end_key=END)

        chain = workflow.compile()

        memory = {"var1": {"geoId": 123}}

        state = chain.invoke(
            {
                "geoId": 234,
                "checkIn": "...",
                "checkOut": "...",
                "refractor": self.refractor,
                "memory": memory,
            }
        )
        assert state.get("response") == "234: hotels"

        state = chain.invoke(
            {
                "checkIn": "...",
                "checkOut": "...",
                "refractor": self.refractor,
                "memory": memory,
            }
        )

        assert state.get("response") == "123: hotels"
