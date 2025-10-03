import asyncio
import logging
import time
from typing import Set, List, Optional, Dict, Any

from altk.toolkit_core.core.toolkit import AgentPhase, ComponentBase, ComponentConfig
from altk.toolkit_core.llm.output_parser import ValidatingLLMClient
from altk.toolkit_core.llm.providers.auto_from_env.auto_from_env import (
    AutoFromEnvLLMClient,
)
from altk.pre_tool_reflection_toolkit.sparc.function_calling.pipeline.pipeline import (
    ReflectionPipeline,
)
from altk.pre_tool_reflection_toolkit.sparc.function_calling.pipeline.types import (
    ToolSpec,
    ToolCall,
    PipelineResult,
)

from altk.pre_tool_reflection_toolkit.core import (
    SPARCReflectionConfig,
    Track,
    SPARCExecutionMode,
    DEFAULT_CONFIGS,
    SPARCReflectionRunInput,
    SPARCReflectionRunOutput,
    SPARCReflectionRunOutputSchema,
    SPARCReflectionResult,
    SPARCReflectionDecision,
    SPARCReflectionIssue,
    SPARCReflectionIssueType,
)


from langchain_core.messages import BaseMessage


logger = logging.getLogger(__name__)


def msgs_to_openai_dicts(msgs: list[BaseMessage]) -> list[dict]:
    return [{"role": msg.type, "content": msg.content} for msg in msgs]


class SPARCReflectionComponent(ComponentBase):
    """
    Component for SPARC reflection using LLMEvalKit's ReflectionPipeline.

    This component evaluates tool calls before they are executed, identifying
    potential issues and suggesting corrections or transformations.
    """

    config: ComponentConfig

    def __init__(
        self,
        config: Optional[ComponentConfig] = None,
        track: Optional[Track] = Track.FAST_TRACK,
        custom_config: Optional[SPARCReflectionConfig] = None,
        execution_mode: SPARCExecutionMode = SPARCExecutionMode.SYNC,
        verbose_logging: bool = False,
        **kwargs,
    ):
        """
        Initialize the component with track-based or custom configuration.

        Args:
            config: ComponentConfig containing LLM client
            track: Predefined configuration track (default: FAST_TRACK)
                   Ignored if custom_config is provided
            custom_config: Custom PreToolReflectionConfig with user-defined metric combinations
                          If provided, overrides the track parameter
            execution_mode: Execution mode (default: SYNC)
            verbose_logging: Enable verbose logging for debugging (default: False)
            **kwargs: Additional keyword arguments

        Examples:
            # Using predefined track
            middleware = PreToolReflectionMiddleware(
                config=config,
                track=Track.FAST_TRACK,
                execution_mode=PreToolExecutionMode.ASYNC
            )

            # Using custom metric configuration
            from llmevalkit.function_calling.consts import (
                METRIC_GENERAL_HALLUCINATION_CHECK,
                METRIC_PARAMETER_HALLUCINATION_CHECK,
            )

            custom_config = PreToolReflectionConfig(
                general_metrics=[METRIC_GENERAL_HALLUCINATION_CHECK],
                function_metrics=None,
                parameter_metrics=[METRIC_PARAMETER_HALLUCINATION_CHECK],
            )

            middleware = PreToolReflectionMiddleware(
                config=config,
                custom_config=custom_config,  # Overrides track parameter
                execution_mode=PreToolExecutionMode.ASYNC
            )
        """
        if config is None:
            config = ComponentConfig()

        super().__init__(config=config)

        # Validate that LLM client is of type ValidatingLLMClient
        if isinstance(config.llm_client, AutoFromEnvLLMClient):
            # special case for auto_from_env
            if not isinstance(config.llm_client._chosen_provider, ValidatingLLMClient):
                raise TypeError(
                    f"LLM client must be of type ValidatingLLMClient, "
                    f"got {type(config.llm_client._chosen_provider).__name__}. "
                    f"Please use a ValidatingLLMClient instance for proper output validation."
                )
        else:
            if not isinstance(config.llm_client, ValidatingLLMClient):
                raise TypeError(
                    f"LLM client must be of type ValidatingLLMClient, "
                    f"got {type(config.llm_client).__name__}. "
                    f"Please use a ValidatingLLMClient instance for proper output validation."
                )

        # Initialize internal state
        self._pipeline: Optional[ReflectionPipeline] = None
        self._tool_specs: List[ToolSpec] = []
        self._initialization_error: Optional[str] = None

        # Build configuration: use custom_config if provided, otherwise use track
        if custom_config is not None:
            # Use user-provided custom configuration
            base_config = custom_config
        else:
            # Use predefined track configuration
            if track is None:
                raise ValueError("Either 'track' or 'custom_config' must be provided")
            base_config = DEFAULT_CONFIGS[track.value]

        config_updates = {
            "execution_mode": execution_mode,
            "verbose_logging": verbose_logging,
        }

        # Apply any additional kwargs
        config_updates.update(kwargs)

        self._config = base_config.model_copy(update=config_updates)

        # Initialize the pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the reflection pipeline with the provided configuration."""
        try:
            if self._config.verbose_logging:
                logging.getLogger().setLevel(logging.DEBUG)

            # Use LLM client from config
            llm_client = self.config.llm_client
            # Special case for auto_from_env since component will check for type
            if isinstance(llm_client, AutoFromEnvLLMClient):
                llm_client = llm_client._chosen_provider
            logger.info(f"Using LLM client from config: {type(llm_client).__name__}")

            # Initialize reflection pipeline
            try:
                self._pipeline = ReflectionPipeline(
                    metrics_client=llm_client,
                    general_metrics=self._config.general_metrics,
                    function_metrics=self._config.function_metrics,
                    parameter_metrics=self._config.parameter_metrics,
                )
                logger.info("Reflection pipeline initialized successfully")

            except Exception as e:
                error_msg = f"Pipeline initialization failed: {str(e)}"
                logger.error(error_msg)
                self._initialization_error = error_msg

        except Exception as e:
            error_msg = f"Middleware initialization failed: {str(e)}"
            logger.error(error_msg)
            self._initialization_error = error_msg

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        """Return the supported agent phases."""
        return {AgentPhase.RUNTIME}

    def _run(self, data: SPARCReflectionRunInput) -> SPARCReflectionRunOutput:
        """
        Run pre-tool reflection on the provided tool call.

        Args:
            data: Run input containing tool call and context

        Returns:
            Reflection results and decision
        """
        start_time = time.time()

        try:
            # Validate that initialization was successful
            if self._initialization_error:
                raise RuntimeError(
                    f"Middleware initialization failed: {self._initialization_error}"
                )

            if not self._pipeline or not self._config:
                raise RuntimeError(
                    "Middleware not properly initialized. Ensure config is provided during initialization."
                )

            # Convert tool specs to ToolSpec objects if needed
            self._tool_specs = [
                ToolSpec.model_validate(spec) for spec in data.tool_specs
            ]

            # Create ToolCall object
            tool_call = ToolCall.model_validate(data.tool_calls[0])

            # Run reflection pipeline
            if self._config.execution_mode == SPARCExecutionMode.ASYNC:
                pipeline_result = asyncio.run(
                    self._run_async_pipeline(
                        msgs_to_openai_dicts(data.messages), tool_call
                    )
                )
            else:
                pipeline_result = self._run_sync_pipeline(
                    msgs_to_openai_dicts(data.messages), tool_call
                )

            # Process pipeline result
            reflection_result = self._process_pipeline_result(pipeline_result)

            # Calculate execution metrics
            execution_time = (time.time() - start_time) * 1000

            # Prepare output
            output = SPARCReflectionRunOutputSchema(
                reflection_result=reflection_result,
                execution_time_ms=execution_time,
            )

            if self._config.include_raw_response:
                output.raw_pipeline_result = (
                    pipeline_result.model_dump() if pipeline_result else None
                )

            return SPARCReflectionRunOutput(output=output)

        except Exception as e:
            logger.error(f"Run phase failed: {str(e)}")

            # Return a default rejection result on error
            error_issue = SPARCReflectionIssue(
                issue_type=SPARCReflectionIssueType.ERROR,
                metric_name="error_handling",
                explanation=f"Reflection pipeline error: {str(e)}",
            )

            error_result = SPARCReflectionResult(
                decision=SPARCReflectionDecision.ERROR,
                issues=[error_issue],
            )

            return SPARCReflectionRunOutput(
                output=SPARCReflectionRunOutputSchema(
                    reflection_result=error_result,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            )

    async def _arun(self, data: SPARCReflectionRunInput) -> SPARCReflectionRunOutput:
        """
        Run pre-tool reflection on the provided tool call asynchronously.

        Args:
            data: Run input containing tool call and context

        Returns:
            Reflection results and decision
        """
        start_time = time.time()

        try:
            # Validate that initialization was successful
            if self._initialization_error:
                raise RuntimeError(
                    f"Middleware initialization failed: {self._initialization_error}"
                )

            if not self._pipeline or not self._config:
                raise RuntimeError(
                    "Middleware not properly initialized. Ensure config is provided during initialization."
                )

            # Convert tool specs to ToolSpec objects if needed
            self._tool_specs = [
                ToolSpec.model_validate(spec) for spec in data.tool_specs
            ]

            # Create ToolCall object
            tool_call = ToolCall.model_validate(data.tool_calls[0])

            # Run reflection pipeline asynchronously
            pipeline_result = await self._run_async_pipeline(
                msgs_to_openai_dicts(data.messages), tool_call
            )

            # Process pipeline result
            reflection_result = self._process_pipeline_result(pipeline_result)

            # Calculate execution metrics
            execution_time = (time.time() - start_time) * 1000

            # Prepare output
            output = SPARCReflectionRunOutputSchema(
                reflection_result=reflection_result,
                execution_time_ms=execution_time,
            )

            if self._config.include_raw_response:
                output.raw_pipeline_result = (
                    pipeline_result.model_dump() if pipeline_result else None
                )

            return SPARCReflectionRunOutput(output=output)

        except Exception as e:
            logger.error(f"Async run phase failed: {e}")

            # Return a default rejection result on error
            error_issue = SPARCReflectionIssue(
                issue_type=SPARCReflectionIssueType.ERROR,
                metric_name="error_handling",
                explanation=f"Reflection pipeline error: {str(e)}",
            )

            error_result = SPARCReflectionResult(
                decision=SPARCReflectionDecision.ERROR,
                issues=[error_issue],
            )

            return SPARCReflectionRunOutput(
                output=SPARCReflectionRunOutputSchema(
                    reflection_result=error_result,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            )

    async def _run_async_pipeline(
        self, conversation_context: List[Dict[str, Any]], tool_call: ToolCall
    ) -> PipelineResult:
        """Run the reflection pipeline asynchronously."""
        return await self._pipeline.run_async(
            conversation_context,
            self._tool_specs,
            tool_call,
            continue_on_static=self._config.continue_on_static,
            transform_enabled=self._config.transform_enabled,
            retries=self._config.retries,
            max_parallel=self._config.max_parallel,
        )

    def _run_sync_pipeline(
        self, conversation_context: List[Dict[str, Any]], tool_call: ToolCall
    ) -> PipelineResult:
        """Run the reflection pipeline synchronously."""
        return self._pipeline.run_sync(
            conversation_context,
            self._tool_specs,
            tool_call,
            continue_on_static=self._config.continue_on_static,
            transform_enabled=self._config.transform_enabled,
        )

    def _process_pipeline_result(
        self, pipeline_result: PipelineResult
    ) -> SPARCReflectionResult:
        """Process the pipeline result into a structured reflection result."""
        issues = []
        decision = SPARCReflectionDecision.APPROVE

        # Check static issues
        if pipeline_result.static and not pipeline_result.static.final_decision:
            for metric_name, metric_result in pipeline_result.static.metrics.items():
                if not metric_result.valid:
                    issues.append(
                        SPARCReflectionIssue(
                            issue_type=SPARCReflectionIssueType.STATIC,
                            metric_name=metric_name,
                            explanation=metric_result.explanation
                            or "Syntax validation failed",
                            correction=metric_result.correction or None,
                        )
                    )

        # Check semantic issues
        if pipeline_result.semantic:
            # Function selection metrics
            function_selection_issues = False
            if pipeline_result.semantic.function_selection:
                for (
                    metric_name,
                    metric_result,
                ) in pipeline_result.semantic.function_selection.metrics.items():
                    if hasattr(metric_result, "is_issue") and metric_result.is_issue:
                        function_selection_issues = True
                        issues.append(
                            SPARCReflectionIssue(
                                issue_type=SPARCReflectionIssueType.SEMANTIC_FUNCTION,
                                metric_name=metric_name,
                                explanation=metric_result.raw_response.get(
                                    "explanation", ""
                                ),
                                correction=metric_result.raw_response.get("correction"),
                            )
                        )
            if not function_selection_issues:
                # General metrics
                if pipeline_result.semantic.general:
                    for (
                        metric_name,
                        metric_result,
                    ) in pipeline_result.semantic.general.metrics.items():
                        if (
                            hasattr(metric_result, "is_issue")
                            and metric_result.is_issue
                        ):
                            issues.append(
                                SPARCReflectionIssue(
                                    issue_type=SPARCReflectionIssueType.SEMANTIC_GENERAL,
                                    metric_name=metric_name,
                                    explanation=metric_result.raw_response.get(
                                        "explanation", ""
                                    ),
                                    correction=metric_result.raw_response.get(
                                        "correction"
                                    ),
                                )
                            )

                # Parameter metrics
                if pipeline_result.semantic.parameter:
                    for (
                        param_name,
                        param_metrics,
                    ) in pipeline_result.semantic.parameter.items():
                        for metric_name, metric_result in param_metrics.metrics.items():
                            if (
                                hasattr(metric_result, "is_issue")
                                and metric_result.is_issue
                            ):
                                issues.append(
                                    SPARCReflectionIssue(
                                        issue_type=SPARCReflectionIssueType.SEMANTIC_PARAMETER,
                                        metric_name=f"{param_name}.{metric_name}",
                                        explanation=metric_result.raw_response.get(
                                            "explanation", ""
                                        ),
                                        correction=metric_result.raw_response.get(
                                            "correction"
                                        ),
                                    )
                                )

                # Transform results
                if pipeline_result.semantic.transform:
                    for (
                        param_name,
                        param_info,
                    ) in pipeline_result.semantic.transform.items():
                        if param_info.correction:
                            issues.append(
                                SPARCReflectionIssue(
                                    issue_type=SPARCReflectionIssueType.TRANSFORM,
                                    metric_name=f"transform.{param_name}",
                                    explanation=f"Parameter transformation suggested for {param_name}.\n{param_info.correction}",
                                    correction={
                                        param_name: param_info.execution_output.get(
                                            "transformed", param_info.execution_output
                                        )
                                    },
                                )
                            )

        # Determine final decision
        if issues:
            decision = SPARCReflectionDecision.REJECT
        else:
            decision = SPARCReflectionDecision.APPROVE

        return SPARCReflectionResult(
            decision=decision,
            issues=issues,
        )
