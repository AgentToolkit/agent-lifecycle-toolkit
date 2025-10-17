import os
from pathlib import Path

import markdown
from altk.core.llm import get_llm

from examples.calculator_example.example_tools import (
    add_tool,
    subtract_tool,
    multiply_tool,
    divide_tool,
)
from examples.tool_guard_example import ToolGuardExample

subdir_name = "work_dir_wx"
work_dir = Path.cwd() / subdir_name
policy_doc_path = os.path.join(str(Path.cwd()), "policy_document.md")
work_dir.mkdir(exist_ok=True)

OPENAILiteLLMClientOutputVal = get_llm("litellm.output_val")
validating_llm_client = OPENAILiteLLMClientOutputVal(
    model_path="gpt-4o-2024-08-06",
    custom_llm_provider="azure",
)

OPENAILiteLLMClient = get_llm("litellm")
llm_client = OPENAILiteLLMClient(
    model_path="gpt-4o-2024-08-06",
    custom_llm_provider="azure",
)
tool_funcs = [add_tool, subtract_tool, multiply_tool, divide_tool]
policy_text = open(policy_doc_path, "r", encoding="utf-8").read()
policy_text = markdown.markdown(policy_text)

tool_guard_example = ToolGuardExample(
    tools=tool_funcs,
    workdir=work_dir,
    policy_text=policy_text,
    validating_llm_client=validating_llm_client,
)
run_output = tool_guard_example.run_example(
    "Can you please calculate how much is 3/4?",
    "divide_tool",
    {"g": 3, "h": 4},
    llm_client,
)
print(run_output)
passed = not run_output.output.error_message
if passed:
    print("success!")
else:
    print("failure!")
run_output = tool_guard_example.run_example(
    "Can you please calculate how much is 5/0?",
    "divide_tool",
    {"g": 5, "h": 0},
    llm_client,
)
print(run_output)
passed = not run_output.output.error_message
if not passed:
    print("success!")
else:
    print("failure!")
