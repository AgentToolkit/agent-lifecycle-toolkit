from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from altk.toolkit_core.core.toolkit import ComponentConfig
from altk.pre_tool_reflection_toolkit.core.consts import (
    METRIC_GENERAL_HALLUCINATION_CHECK,
    METRIC_GENERAL_VALUE_FORMAT_ALIGNMENT,
    METRIC_FUNCTION_SELECTION_APPROPRIATENESS,
    METRIC_AGENTIC_CONSTRAINTS_SATISFACTION,
    METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT,
    METRIC_PARAMETER_HALLUCINATION_CHECK,
)


class SPARCExecutionMode(str, Enum):
    """Execution mode for the reflection pipeline."""

    SYNC = "sync"
    ASYNC = "async"


class Track(str, Enum):
    """Predefined configuration tracks for the reflection pipeline."""

    SYNTAX = "syntax"
    FAST_TRACK = "fast_track"
    SLOW_TRACK = "slow_track"
    TRANSFORMATIONS_ONLY = "transformations_only"


class SPARCReflectionConfig(BaseModel):
    """Configuration for the SPARC Middleware."""

    # LLM Configuration
    model_path: str = Field(
        default="meta-llama/llama-3-3-70b-instruct",
        description="The model path/identifier for the reflection LLM",
    )
    custom_llm_provider: Optional[str] = Field(
        default=None,
        description="Custom LLM provider (e.g., 'openai', 'anthropic', 'litellm')",
    )

    # Pipeline Execution Configuration
    execution_mode: SPARCExecutionMode = Field(
        default=SPARCExecutionMode.SYNC,
        description="Whether to run the pipeline synchronously or asynchronously",
    )

    # Async Configuration (only used when execution_mode is ASYNC)
    retries: int = Field(
        default=3, description="Number of retries for async execution", ge=0, le=10
    )
    max_parallel: int = Field(
        default=7,
        description="Maximum parallel executions for async mode",
    )

    # Pipeline Configuration
    continue_on_static: bool = Field(
        default=False,
        description="Whether to continue pipeline execution even if static checks fail",
    )
    transform_enabled: bool = Field(
        default=False,
        description="Whether to enable parameter transformation in the pipeline",
    )

    # Metrics Configuration
    general_metrics: Optional[List[str]] = Field(
        default=[
            METRIC_GENERAL_HALLUCINATION_CHECK,
            METRIC_GENERAL_VALUE_FORMAT_ALIGNMENT,
        ],
        description="List of general metrics to evaluate",
    )
    function_metrics: Optional[List[str]] = Field(
        default=[
            METRIC_FUNCTION_SELECTION_APPROPRIATENESS,
            METRIC_AGENTIC_CONSTRAINTS_SATISFACTION,
        ],
        description="List of function-specific metrics to evaluate",
    )
    parameter_metrics: Optional[List[str]] = Field(
        default=[
            METRIC_PARAMETER_HALLUCINATION_CHECK,
            METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT,
        ],
        description="List of parameter-specific metrics to evaluate",
    )

    # Output Configuration
    include_raw_response: bool = Field(
        default=False,
        description="Whether to include raw reflection pipeline response in output",
    )

    verbose_logging: bool = Field(
        default=False, description="Enable verbose logging for debugging"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        extra = "forbid"


# Default configurations for common use cases
DEFAULT_CONFIGS = {
    "syntax": SPARCReflectionConfig(
        general_metrics=None,
        function_metrics=None,
        parameter_metrics=None,
    ),
    "fast_track": SPARCReflectionConfig(
        general_metrics=[METRIC_GENERAL_HALLUCINATION_CHECK],
        function_metrics=[METRIC_FUNCTION_SELECTION_APPROPRIATENESS],
        parameter_metrics=None,
    ),
    "slow_track": SPARCReflectionConfig(
        general_metrics=[
            METRIC_GENERAL_HALLUCINATION_CHECK,
            METRIC_GENERAL_VALUE_FORMAT_ALIGNMENT,
        ],
        function_metrics=[
            METRIC_FUNCTION_SELECTION_APPROPRIATENESS,
            METRIC_AGENTIC_CONSTRAINTS_SATISFACTION,
        ],
        parameter_metrics=None,
        transform_enabled=True,
    ),
    "transformations_only": SPARCReflectionConfig(
        general_metrics=None,
        function_metrics=None,
        parameter_metrics=None,
        transform_enabled=True,
    ),
}


class RefractionMode(str, Enum):
    """Refraction mode for the reflection pipeline."""

    STANDALONE = "standalone"
    WITH_LLM = "with_llm"


class RefractionConfig(ComponentConfig):
    """Configuration for the Refraction Middleware."""

    mode: RefractionMode = Field(
        default=RefractionMode.STANDALONE,
    )
