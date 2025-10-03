import re
import json
import logging
from typing import Optional, Set
from nestful.schemas.sequences import SequenceStep, SequencingData
from altk.toolkit_core.core.toolkit import AgentPhase, ComponentBase
from altk.pre_tool_reflection_toolkit.core.config import (
    RefractionConfig,
    RefractionMode,
)
from altk.pre_tool_reflection_toolkit.core.types import (
    RefractionBuildInput,
    RefractionRunInput,
    RefractionRunOutput,
)
from altk.pre_tool_reflection_toolkit.refraction.src import generate_prompt
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    DebuggingResult,
    PromptType,
)


logger = logging.getLogger(__name__)


class RefractionComponent(ComponentBase):
    """
    Refraction for pre-tool-call evaluation
    """

    _config: RefractionConfig
    _refractor: Refractor

    def __init__(self, config: Optional[RefractionConfig] = None) -> None:
        if config is None:
            config = RefractionConfig()

        super().__init__(config=config)

        self._config = config
        self._refractor = Refractor(catalog=[])

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        return {AgentPhase.RUNTIME, AgentPhase.BUILDTIME}

    @staticmethod
    def _parse_tool_call_repair_response(
        model_response: str,
    ) -> Optional[SequencingData]:
        match = re.search(r"(?P<tool_call>\[\s*\{.*?}\s*])", model_response, re.DOTALL)

        if not match or not match.group("tool_call"):
            return None

        try:
            tool_call = match.group("tool_call").replace("'", '"')
            parsed_tool_call_sequence = json.loads(tool_call)

            sequence = SequencingData(
                output=[SequenceStep(**step) for step in parsed_tool_call_sequence]
            )

            return sequence

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool call from model response: {e}")
            return None

    def _repair_with_llm(
        self,
        result: DebuggingResult,
        data: RefractionRunInput,
        prompt_type: PromptType,
        *,
        max_retries: int = 3,
        use_given_operators_only: bool = False,
    ) -> DebuggingResult:
        num_tries = 0
        current_result: DebuggingResult = result

        assert current_result.backing_data is not None

        while num_tries < max_retries:
            num_tries += 1
            logger.info("Generating prompt...")
            prompt = generate_prompt(
                current_result,
                data.tool_calls,
                data.tool_specs,
                data.memory_objects or {},
                prompt_type,
            )

            logger.info("Sending prompt to LLM for repair...")

            if not self._config:
                raise Exception("Config needs to be specified")

            response = self._config.llm_client.generate(prompt)

            # extract the tool calls from the model response
            logger.info("Extracting tool calls from model response")
            repaired_sequence = self._parse_tool_call_repair_response(response)
            if not repaired_sequence:
                raise Exception("Failed to extract tool call from model response")

            logger.info("Running refract on tool call")
            current_result = self._refractor.refract(
                sequence=repaired_sequence,
                catalog=data.tool_specs,
                memory_objects=data.memory_objects,
                use_given_operators_only=use_given_operators_only,
            )

        return current_result

    def _build(self, data: RefractionBuildInput) -> Refractor:
        logger.info("Starting build phase ...")
        self._refractor.catalog = data.tool_specs

        if data.compute_maps is True:
            logger.info("Computing maps ...")
            self._refractor.initialize_maps(
                top_k=data.top_k, mapping_threshold=data.threshold
            )

        return self._refractor

    def _run(self, data: RefractionRunInput) -> RefractionRunOutput:
        try:
            logger.info("Running refraction on provided tool specs and tool calls...")

            if data.mappings:
                self._refractor.mappings = data.mappings

            result = self._refractor.refract(
                data.tool_calls,
                catalog=data.tool_specs,
                memory_objects=data.memory_objects,
                use_given_operators_only=data.use_given_operators_only,
            )

            if result.report.determination:
                logger.info("No issues detected after running refraction.")
                return RefractionRunOutput(result=result)

            logger.info("Found issues with the provided tool calls.")
            if self._config.mode == RefractionMode.STANDALONE:
                logger.info("Attempting resolution in standalone mode...")
                return RefractionRunOutput(result=result)

            elif self._config.mode == RefractionMode.WITH_LLM:
                logger.info("Attempting to repair issues with LLM...")
                llm_repair_result = self._repair_with_llm(
                    result, data, PromptType.WITH_SUGGESTIONS
                )
                return RefractionRunOutput(result=llm_repair_result)

            else:
                logger.info(
                    "Provided mode is not defined. Failed to attempt resolution..."
                )
                raise ValueError(f"'{self._config.mode}' mode is not defined")

        except Exception as e:
            logger.error(f"Run phase failed with error: {e}")
            return RefractionRunOutput(result=None)
