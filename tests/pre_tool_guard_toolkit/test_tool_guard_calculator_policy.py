import os

import dotenv
import pytest

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# from altk.pre_tool_guard_toolkit.pre_tool_guard.pre_tool_guard import PreToolGuardComponent
from altk.toolkit_core.llm import get_llm

from altk.pre_tool_guard_toolkit.core import (
    ToolGuardBuildInput,
    ToolGuardBuildInputMetaData,
    ToolGuardRunInput,
    ToolGuardRunInputMetaData,
)
# NOTE: ToolGuard broken until internal repo fixed
# from altk.pre_tool_guard_toolkit.pre_tool_guard import PreToolGuardComponent

import tempfile
import shutil

dotenv.load_dotenv()


@tool
def divide_tool(g: float, h: float) -> float:
    """
    Divide one number by another.

    Parameters
    ----------
    g : float
        The dividend.
    h : float
        The divisor (must not be zero).

    Returns
    -------
    float
        The result of a divided by b.
    """
    return g / h


AZURE_CREDS_AVAILABLE = all(
    [
        os.getenv("AZURE_OPENAI_API_KEY"),
        os.getenv("AZURE_API_BASE"),
        os.getenv("AZURE_API_VERSION"),
    ]
)


@pytest.mark.skip(reason="ToolGuard broken until internal repo is fixed")
@pytest.mark.skipif(
    not AZURE_CREDS_AVAILABLE, reason="Azure OpenAI credentials not set"
)
def test_tool_guard_calculator_policy():
    pass
    # work_dir = tempfile.mkdtemp()
    # tools = [divide_tool]
    # policy_text = "The calculator must not allow division by zero."

    # OPENAILiteLLMClientOutputVal = get_llm("litellm.output_val")
    # validating_llm_client = OPENAILiteLLMClientOutputVal(
    # 	model_path="gpt-4o-2024-08-06",
    # 	custom_llm_provider="azure",
    # )

    # OPENAILiteLLMClient = get_llm("litellm")
    # llm_client = OPENAILiteLLMClient(
    # 	model_path="gpt-4o-2024-08-06",
    # 	custom_llm_provider="azure",
    # )

    # middleware = PreToolGuardComponent(tools=tools, workdir=work_dir)
    # build_input = ToolGuardBuildInput(
    # 	metadata=ToolGuardBuildInputMetaData(
    # 		policy_text=policy_text,
    # 		short1=True,
    # 		validating_llm_client=validating_llm_client,
    # 	)
    # )
    # middleware._build(build_input)

    # test_options = [("Can you please calculate how much is 3/4?","divide_tool",{"g":3,"h":4},True),
    # 				("Can you please calculate how much is 5/0?", "divide_tool", {"g": 5, "h": 0},False)]
    # for user_query,tool_name,tool_params,expected in test_options:
    # 	conversation_context = [HumanMessage(content=user_query)]
    # 	run_input = ToolGuardRunInput(
    # 		messages=conversation_context,
    # 		metadata=ToolGuardRunInputMetaData(
    # 			tool_name=tool_name,
    # 			tool_parms=tool_params,
    # 			llm_client=llm_client
    # 		)
    # 	)

    # 	run_output = middleware._run(run_input)
    # 	print(run_output)
    # 	passed = not run_output.output.error_message
    # 	assert passed == expected

    # shutil.rmtree(work_dir)
