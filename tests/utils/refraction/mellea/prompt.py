from typing import Dict, List, Any
from jinja2 import Template


PROMPT_TEMPLATE = """
You are an expert in tool calling.
You are given a user query, a set of available tools, and the current working memory.
Your task is to generate a sequence of one or more tools calls to satisfy the query.

The user said: {{ query }}

You have access to the following tools:
<tools>{{ tools }}</tools>

Current working memory:
<memory>{{ memory }}</memory>

The output must strictly adhere to the following format, and NO OTHER TEXT must be included.
Each call is tagged with a label "var" followed by the index of that call in the sequence starting from 1 e.g.
the second call will have the label var2.
You can assign the value of a parameter either directly if its value is known or by referring to outputs
produced by previous calls or to variables in memory.
You can refer to output of previous calls using the labels e.g. $var1.name_of_output_variable$.
You can also refer to items in memory using the same format.
Here is an example:

<tool_calls>[
{"name": "func_name1", "args": {"argument1": "value1", "argument2": "$variable_in_memory$"}, "label": "var1"},
{"name": "func_name1", "args": {"argument1": "$var1.key.nested_key$", "argument2": "value2"}, "label": "var2"},
... (more tool calls as required)
]</tool_calls>
"""


RESPONSE_TEMPLATE = """
<tool_calls>{{ tool_calls | tojson }}</tool_calls>
"""


def generate_response(tool_calls: List[Dict[str, Any]]) -> str:
    return str(Template(RESPONSE_TEMPLATE).render(tool_calls=tool_calls))
