import dotenv

from langchain_core.messages import HumanMessage

from altk.pre_tool_guard_toolkit.core import (
    ToolGuardBuildInput,
    ToolGuardBuildInputMetaData,
    ToolGuardRunInput,
    ToolGuardRunInputMetaData,
)
from altk.pre_tool_guard_toolkit.pre_tool_guard import PreToolGuardComponent

# Load environment variables
dotenv.load_dotenv()


class ToolGuardExample:
    """
    Runs examples with a ToolGuard component and validates tool invocation against policy.
    """

    def __init__(self, tools, workdir, policy_text, validating_llm_client, short=True):
        self.tools = tools
        self.middleware = PreToolGuardComponent(tools=tools, workdir=workdir)

        build_input = ToolGuardBuildInput(
            metadata=ToolGuardBuildInputMetaData(
                policy_text=policy_text,
                short1=short,
                validating_llm_client=validating_llm_client,
            )
        )
        self.middleware._build(build_input)

    def run_example(
        self, user_message: str, tool_name: str, tool_params: dict, llm_client
    ):
        """
        Runs a single example through ToolGuard and checks if the result matches the expectation.
        """
        conversation_context = [HumanMessage(content=user_message)]

        run_input = ToolGuardRunInput(
            messages=conversation_context,
            metadata=ToolGuardRunInputMetaData(
                tool_name=tool_name, tool_parms=tool_params, llm_client=llm_client
            ),
        )

        run_output = self.middleware._run(run_input)
        return run_output
