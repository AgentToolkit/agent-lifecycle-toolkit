import asyncio
import json
import logging
import os
from typing import Set

from altk.toolkit_core.core.toolkit import AgentPhase, ComponentBase, ComponentOutput

from toolguard.__main__ import step2
from toolguard.llm.tg_llmevalkit import TG_LLMEval
from toolguard.stages_tptd.text_tool_policy_generator import step1_main_with_tools

from altk.pre_tool_guard_toolkit.core.types import (
    ToolGuardBuildInput,
    ToolGuardRunInput,
    ToolGuardRunOutput,
    ToolGuardRunOutputMetaData,
)

logger = logging.getLogger(__name__)


class PreToolGuardComponent(ComponentBase):
    def __init__(self, tools, workdir):
        super().__init__()
        self._tools = tools
        self._workdir = workdir
        self._step1_dir = os.path.join(self._workdir, "Step_1")
        self._step2_dir = os.path.join(self._workdir, "Step_2")

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        """Return the supported agent phases."""
        return {AgentPhase.BUILDTIME, AgentPhase.RUNTIME}

    def _build(self, data: ToolGuardBuildInput) -> ComponentOutput:
        llm = TG_LLMEval(data.metadata.validating_llm_client)
        step1_main_with_tools(
            data.metadata.policy_text,
            self._tools,
            self._step1_dir,
            llm,
            None,
            data.metadata.short1,
        )
        asyncio.run(step2(self._tools, self._step1_dir, self._step2_dir, None))

    def _run(self, data: ToolGuardRunInput) -> ToolGuardRunOutput:
        import sys

        sys.path.insert(0, self._step2_dir)
        tool_name = data.metadata.tool_name
        tool_parms = data.metadata.tool_parms
        import rt_toolguard

        app_guards = rt_toolguard.load(self._step2_dir)
        llm = rt_toolguard.MTKLLM(data.metadata.llm_client)
        app_guards.use_llm(llm)

        try:
            history_messages = [
                {"role": msg.type, "content": msg.content} for msg in data.messages
            ]
            app_guards.check_tool_call(tool_name, tool_parms, history_messages)
            error_message = False
        except Exception as e:
            error_message = (
                f"It is against the policy to invoke tool: {tool_name}({json.dumps(tool_parms)}) Error: "
                + str(e)
            )
        output = ToolGuardRunOutputMetaData(error_message=error_message)
        return ToolGuardRunOutput(output=output)
