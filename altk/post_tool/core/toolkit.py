from enum import Enum
from typing import List, Optional, Dict, Union, Any
from collections.abc import Callable
from altk.core.toolkit import (
    ComponentBase,
    ComponentInput,
    ComponentOutput,
)

######### Reflection Component Interfaces ##############


class PostToolReflectionComponent(ComponentBase):
    pass


class PostToolReflectionBuildInput(ComponentInput):
    pass


class PostToolReflectionRunInput(ComponentInput):
    tool_response: Optional[Union[Dict, List[Dict]]] = None


class PostToolReflectionBuildOutput(ComponentOutput):
    pass


class PostToolReflectionRunOutput(ComponentOutput):
    pass


######### Silent Review Reflection Component Interfaces ##############


class SilentReviewBuildInput(PostToolReflectionBuildInput):
    pass


class SilentReviewBuildOutput(PostToolReflectionBuildOutput):
    pass


class SilentReviewRunInput(PostToolReflectionRunInput):
    tool_spec: Optional[Union[Dict, List[Dict]]] = None


class Outcome(float, Enum):
    """Used for Review"""

    NOT_ACCOMPLISHED = 0
    PARTIAL_ACCOMPLISH = 0.5
    ACCOMPLISHED = 1


class SilentReviewRunOutput(PostToolReflectionRunOutput):
    outcome: Outcome
    details: Dict

    def as_tuple(self) -> tuple[Outcome, dict]:
        return self.outcome, self.details


######### CodeGen Reflection Component Interfaces ##############
class CodeGenerationRunInput(PostToolReflectionRunInput):
    nl_query: str


class CodeGenerationRunOutput(PostToolReflectionRunOutput):
    result: Any


######### RAGRepair Component Interfaces ##############
class RAGRepairBuildInput(PostToolReflectionBuildInput):
    pass


class RAGRepairBuildOutput(PostToolReflectionBuildOutput):
    pass


class RAGRepairRunInput(PostToolReflectionRunInput):
    original_function: Optional[Callable] = None
    nl_query: str = ""
    tool_call: str
    error: Optional[str] = None


class RAGRepairRunOutput(PostToolReflectionRunOutput):
    result: Optional[Any] = None
    new_tool_call: str
    retrieved_docs: str = ""
