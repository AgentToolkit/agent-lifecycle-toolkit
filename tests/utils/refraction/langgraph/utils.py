from typing import Dict, List, Any
from jinja2 import Template
from re import search, DOTALL
from json import loads, JSONDecodeError
from nestful.utils import get_token
from langchain_core.tools import BaseTool


PROMPT_TEMPLATE = Template("""
You are an expert in correcting tool calls. You are given a set of available tools, a query and an incorrect tool call that was meant to satisfy the query.

The user said: {{ query }}

You have access to the following tools:
<tools>{{ tools }}</tools>

The output must strictly adhere to the following format, and NO OTHER TEXT must be included:

<tool_call>[
{"name": "func_name1", "args": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>
""")


def generate_prompt(query: str, list_of_tools: List[Dict[str, Any]]) -> str:
    return str(PROMPT_TEMPLATE.render(query=query, tools=list_of_tools))


def extract_tool_calls(response: str) -> List[Dict[str, Any]]:
    match = search(r"(?P<tool_call>\[\s*\{.*?}\s*])", response, DOTALL)

    if match is None:
        return []

    try:
        tool_call_str = match.group("tool_call")
        tool_calls: List[Dict[str, Any]] = loads(tool_call_str)

        return tool_calls

    except (AttributeError, JSONDecodeError) as e:
        print(e)
        return []


def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
    tools: List[BaseTool],
) -> List[Dict[str, Any]]:
    list_of_responses: List[Dict[str, Any]] = []

    for index, call in enumerate(tool_calls):
        label = call.get("label", get_token(index + 1))
        tool = next(filter(lambda x: x.name == call["name"], tools), None)

        if tool is not None:
            response = {label: tool.invoke(call.get("args", {}))}
        else:
            response = {label: None}

        list_of_responses.append(response)

    return list_of_responses
