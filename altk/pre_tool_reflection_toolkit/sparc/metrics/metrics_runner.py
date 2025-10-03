import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from altk.pre_tool_reflection_toolkit.sparc.prompt.runner import (
    PromptRunner,
    PromptResult,
    SyncGen,
    AsyncGen,
    Prompt,
    PromptAndSchema,
)
from altk.pre_tool_reflection_toolkit.sparc.metrics.prompt import MetricPrompt
from altk.pre_tool_reflection_toolkit.sparc.metrics.field import NumericField
from altk.pre_tool_reflection_toolkit.sparc.metrics.metric import Metric


class MetricRunResult(BaseModel):
    """
    Structured result for a single metric invocation.
    """

    metric_name: str
    jsonschema: Dict[str, Any]
    prompt: Prompt
    raw_response: Any
    numeric_thresholds_checks: Dict[str, bool]
    error: Optional[str]
    is_important: bool
    importance_reason: Optional[str]
    is_correct: bool
    correctness_reason: Optional[str]
    is_issue: bool


class MetricRunner:
    """
    Orchestrates running multiple metrics via LLM calls.
    """

    def __init__(
        self, entries: Optional[List[Tuple[MetricPrompt, Dict[str, Any]]]] = None
    ) -> None:
        """
        Args:
            entries: Optional list of (MetricPrompt, user_kwargs) pairs.
        """
        self.entries: List[Dict[str, Any]] = []
        if entries:
            for mp, kw in entries:
                self.add(mp, kw)

    def add(self, metric_prompt: MetricPrompt, user_kwargs: Dict[str, Any]) -> None:
        """
        Add a metric to run.

        Args:
            metric_prompt: MetricPrompt instance.
            user_kwargs: Dict of variables to render the user template.
        """
        messages = metric_prompt.build_messages(user_kwargs)
        self.entries.append(
            {
                "metric_prompt": metric_prompt,
                "user_kwargs": user_kwargs,
                "messages": messages,
                "schema": metric_prompt.metric.to_jsonschema(),
            }
        )

    def remove(self, index: int) -> None:
        """Remove the entry at the given index."""
        self.entries.pop(index)

    def clear(self) -> None:
        """Remove all entries."""
        self.entries.clear()

    def _assemble_prompts(self) -> List[PromptAndSchema]:
        return [(e["messages"], e["schema"]) for e in self.entries]

    def _process_results(
        self, prompt_results: List[PromptResult]
    ) -> List[MetricRunResult]:
        """
        Combine PromptResult with metric parsing, threshold checks,
        importance and correctness determinations.
        """
        results: List[MetricRunResult] = []

        for entry, pr in zip(self.entries, prompt_results):
            mp: MetricPrompt = entry["metric_prompt"]
            metric: Metric = mp.metric

            # default values
            numeric_thresholds_checks: Dict[str, bool] = {}
            err = pr.error
            is_imp = False
            imp_reason = None
            is_corr = False
            corr_reason = None
            data = None

            if pr.error is None:
                try:
                    # parse raw response into JSON-compatible dict
                    raw = pr.response
                    if isinstance(raw, str):
                        data = json.loads(raw)
                    else:
                        data = raw

                    # numeric threshold checks
                    for field in metric.fields:
                        if isinstance(field, NumericField):
                            val = data.get(field.name)
                            ok = False
                            if isinstance(val, (int, float)):
                                ok = field.is_within_threshold(val)
                            numeric_thresholds_checks[field.name] = ok

                    # importance and correctness
                    is_imp, imp_reason = metric.is_important(data)
                    is_corr, corr_reason = metric.is_correct(data)

                except Exception as e:
                    err = str(e)

            # Build the result model
            result = MetricRunResult(
                metric_name=metric.name,
                jsonschema=entry["schema"],
                prompt=pr.prompt,
                raw_response=pr.response,
                numeric_thresholds_checks=numeric_thresholds_checks,
                error=err,
                is_important=is_imp,
                importance_reason=imp_reason,
                is_correct=is_corr,
                correctness_reason=corr_reason,
                is_issue=is_imp and not is_corr,
            )
            results.append(result)

        return results

    def run_all(
        self,
        gen_fn: SyncGen,
        prompt_param_name: str = "prompt",
        schema_param_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[MetricRunResult]:
        """
        Run all metrics using a synchronous single-prompt generator.
        """
        prompts = self._assemble_prompts()
        runner = PromptRunner(prompts)
        pr_results = runner.run_all(
            gen_fn,
            prompt_param_name=prompt_param_name,
            schema_param_name=schema_param_name,
            **kwargs,
        )
        return self._process_results(pr_results)

    async def run_async(
        self,
        async_fn: AsyncGen,
        max_parallel: int = 10,
        prompt_param_name: str = "prompt",
        schema_param_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[MetricRunResult]:
        """
        Run all metrics using asynchronous single-prompt generation.
        """
        prompts = self._assemble_prompts()
        runner = PromptRunner(prompts)
        pr_results = await runner.run_async(
            async_fn,
            max_parallel=max_parallel,
            prompt_param_name=prompt_param_name,
            schema_param_name=schema_param_name,
            **kwargs,
        )
        return self._process_results(pr_results)
