from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from altk.toolkit_core.llm import ValidatingLLMClient
from altk.pre_tool_reflection_toolkit.sparc.function_calling.pipeline.static_checker import (
    evaluate_static,
)
from altk.pre_tool_reflection_toolkit.sparc.function_calling.pipeline.semantic_checker import (
    SemanticChecker,
)
from altk.pre_tool_reflection_toolkit.sparc.function_calling.pipeline.types import (
    FunctionCallMetric,
    FunctionCallInput,
    PipelineResult,
    StaticResult,
    SemanticResult,
    ToolSpec,
    ToolCall,
    StaticMetricResult,
)

# Default metric JSON paths
_METRICS_DIR = Path(__file__).parent.parent / "metrics"
_DEFAULT_GENERAL = _METRICS_DIR / "function_call" / "general_metrics.json"
_DEFAULT_GENERAL_RUNTIME = (
    _METRICS_DIR / "function_call" / "general_metrics_runtime.json"
)
_DEFAULT_FUNCSEL = (
    _METRICS_DIR / "function_selection" / "function_selection_metrics.json"
)
_DEFAULT_FUNCSEL_RUNTIME = (
    _METRICS_DIR / "function_selection" / "function_selection_metrics_runtime.json"
)
_DEFAULT_PARAM = _METRICS_DIR / "parameter" / "parameter_metrics.json"
_DEFAULT_PARAM_RUNTIME = _METRICS_DIR / "parameter" / "parameter_metrics_runtime.json"


class ReflectionPipeline:
    """
    High-level orchestration for function-call reflection.

    Modes:
      • static_only: schema checks
      • semantic_only: LLM metrics + transforms
      • run: full static -> semantic -> assemble -> PipelineResult

    Supports sync & async, custom JSON overrides, and any registered LLM.
    runtime_pipeline: if set to true, use faster prompts (no actionable recommendations, shorter explanations)
    """

    def __init__(
        self,
        metrics_client: ValidatingLLMClient,
        codegen_client: Optional[ValidatingLLMClient] = None,
        general_metrics: Optional[
            Union[Path, List[FunctionCallMetric], List[str]]
        ] = _DEFAULT_GENERAL_RUNTIME,
        function_metrics: Optional[
            Union[Path, List[FunctionCallMetric], List[str]]
        ] = _DEFAULT_FUNCSEL_RUNTIME,
        parameter_metrics: Optional[
            Union[Path, List[FunctionCallMetric], List[str]]
        ] = _DEFAULT_PARAM_RUNTIME,
        transform_enabled: Optional[bool] = False,
        runtime_pipeline: Optional[bool] = True,
        use_examples: Optional[bool] = True,
    ):
        self.metrics_client = metrics_client
        if codegen_client is None:
            self.codegen_client = metrics_client
        else:
            self.codegen_client = codegen_client

        self.general_metrics = general_metrics
        self.function_metrics = function_metrics
        self.parameter_metrics = parameter_metrics

        metrics_definitions = []

        for metrics, default_path in [
            (
                self.general_metrics,
                _DEFAULT_GENERAL_RUNTIME if runtime_pipeline else _DEFAULT_GENERAL,
            ),
            (
                self.function_metrics,
                _DEFAULT_FUNCSEL_RUNTIME if runtime_pipeline else _DEFAULT_FUNCSEL,
            ),
            (
                self.parameter_metrics,
                _DEFAULT_PARAM_RUNTIME if runtime_pipeline else _DEFAULT_PARAM,
            ),
        ]:
            if not metrics:
                metrics_definitions.append(None)
                continue

            # Handle metric names list
            if isinstance(metrics, list) and all(isinstance(x, str) for x in metrics):
                # Load the default JSON file
                if not default_path.is_file():
                    raise FileNotFoundError(
                        f"Default metrics file not found: {default_path}"
                    )

                with default_path.open("r") as f_in:
                    all_metrics = json.load(f_in)

                # Filter metrics by name
                filtered_metrics = [
                    metric for metric in all_metrics if metric.get("name") in metrics
                ]

                # Remove examples from prompts if requested
                if not use_examples:
                    for metric in filtered_metrics:
                        metric.pop("examples", None)

                if len(filtered_metrics) != len(metrics):
                    found_names = {metric.get("name") for metric in filtered_metrics}
                    missing = set(metrics) - found_names
                    raise ValueError(f"Metrics not found: {missing}")

                metrics_definitions.append(filtered_metrics)
                continue

            # Handle Path or List[FunctionCallMetric] (existing logic)
            if not isinstance(metrics, (Path, list)):
                raise TypeError(
                    "metrics must be Path, List[FunctionCallMetric], List[str], or None"
                )
            if isinstance(metrics, list) and all(
                isinstance(x, FunctionCallMetric) for x in metrics
            ):
                metrics_definitions.append([metric.model_dump() for metric in metrics])
            else:
                if not metrics.is_file():
                    raise FileNotFoundError(f"Metrics file not found: {metrics}")
                with metrics.open("r", encoding="utf8") as f:
                    metrics_definitions.append(json.load(f))

        gen_defs, fun_defs, par_defs = None, None, None

        if metrics_definitions:
            gen_defs = metrics_definitions[0]
            if len(metrics_definitions) >= 2:
                fun_defs = metrics_definitions[1]
                if len(metrics_definitions) >= 3:
                    par_defs = metrics_definitions[2]

        # 3) Initialize semantic checker
        self.semantic_checker = SemanticChecker(
            general_metrics=gen_defs,
            function_metrics=fun_defs,
            parameter_metrics=par_defs,
            metrics_client=self.metrics_client,
            codegen_client=self.codegen_client,
            transform_enabled=transform_enabled,
        )

    @staticmethod
    def static_only(
        inventory: List[ToolSpec],
        call: ToolCall,
    ) -> StaticResult:
        """
        Run schema-based static checks.

        Returns:
            StaticResult with per-check results and final_decision.
        """
        try:
            return evaluate_static(inventory, call)
        except Exception as e:
            return StaticResult(
                metrics={
                    "json_schema_validation": StaticMetricResult(
                        description="Invalid JSON schema",
                        valid=False,
                        explanation=f"error parsing JSON schema: {str(e)}",
                    )
                },
                final_decision=False,
            )

    def semantic_sync(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        inventory: List[ToolSpec],
        call: ToolCall,
        retries: Optional[int] = 2,
        transform_enabled: Optional[bool] = None,
    ) -> SemanticResult:
        """
        Synchronous LLM-based semantic metrics (+ optional transforms).
        """
        # delegate to SemanticChecker
        return self.semantic_checker.run_sync(
            inventory,
            call,
            conversation,
            retries=retries,
            transform_enabled=transform_enabled,
        )

    async def semantic_async(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        inventory: List[ToolSpec],
        call: ToolCall,
        retries: Optional[int] = 2,
        max_parallel: Optional[int] = 10,
        transform_enabled: Optional[bool] = None,
    ) -> SemanticResult:
        """
        Async LLM-based semantic metrics (+ optional transforms).
        """
        return await self.semantic_checker.run_async(
            inventory,
            call,
            conversation,
            retries=retries,
            max_parallel=max_parallel,
            transform_enabled=transform_enabled,
        )

    def run_sync(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        inventory: List[ToolSpec],
        call: ToolCall,
        continue_on_static: Optional[bool] = False,
        retries: Optional[int] = 1,
        transform_enabled: Optional[bool] = None,
    ) -> PipelineResult:
        """
        Full sync pipeline: static -> semantic -> assemble PipelineResult.
        """
        static_res = self.static_only(inventory, call)

        if not static_res.final_decision and not continue_on_static:
            inputs = FunctionCallInput(
                conversation_context=conversation,
                tools_inventory=inventory,
                tool_call=call,
            )
            return PipelineResult(
                inputs=inputs,
                static=static_res,
                semantic=SemanticResult(
                    general=None,
                    function_selection=None,
                    parameter=None,
                    transform=None,
                ),
                overall_valid=False,
            )

        semantic_res = self.semantic_sync(
            conversation, inventory, call, retries, transform_enabled
        )
        return PipelineResult(
            inputs=FunctionCallInput(
                conversation_context=conversation,
                tools_inventory=inventory,
                tool_call=call,
            ),
            static=static_res,
            semantic=semantic_res,
            overall_valid=True,
        )

    async def run_async(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        inventory: List[ToolSpec],
        call: ToolCall,
        continue_on_static: Optional[bool] = False,
        retries: Optional[int] = 1,
        max_parallel: Optional[int] = 10,
        transform_enabled: Optional[bool] = None,
    ) -> PipelineResult:
        """
        Full async pipeline: static -> semantic -> assemble PipelineResult.
        """
        static_res = self.static_only(inventory, call)

        if not static_res.final_decision and not continue_on_static:
            inputs = FunctionCallInput(
                conversation_context=conversation,
                tools_inventory=inventory,
                tool_call=call,
            )
            return PipelineResult(
                inputs=inputs,
                static=static_res,
                semantic=SemanticResult(
                    general=None,
                    function_selection=None,
                    parameter=None,
                    transform=None,
                ),
                overall_valid=False,
            )

        semantic_res = await self.semantic_async(
            conversation, inventory, call, retries, max_parallel, transform_enabled
        )
        return PipelineResult(
            inputs=FunctionCallInput(
                conversation_context=conversation,
                tools_inventory=inventory,
                tool_call=call,
            ),
            static=static_res,
            semantic=semantic_res,
            overall_valid=True,
        )
