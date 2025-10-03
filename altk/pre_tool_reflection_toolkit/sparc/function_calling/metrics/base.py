from abc import ABC

from altk.pre_tool_reflection_toolkit.sparc.metrics import Metric, MetricPrompt


class FunctionMetricsPrompt(MetricPrompt, ABC):
    """
    Abstract base for function-calling metric prompts.
    Subclasses must define class attrs:
      - system_template: str
      - user_template: str
    """

    system_template: str
    user_template: str

    def __init__(self, metric: Metric, task_description: str) -> None:
        super().__init__(
            metric=metric,
            system_template=self.system_template,
            user_template=self.user_template,
            system_kwargs_defaults={
                "task_description": task_description,
                "metric_jsonschema": metric.to_jsonschema(),
            },
        )
