from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from altk.core.toolkit import ComponentInput, ComponentOutput


######### Policy Guard Middleware Interfaces ##############


class PolicyDetectorInput(ComponentInput):
    policies: List[str]
    prompt: str
    response: str


class PolicyDetectorSingleOutput(BaseModel):
    policy: str
    compliance: bool
    explanation: str


class PolicyDetectorOutput(ComponentOutput):
    policy_outputs: List[PolicyDetectorSingleOutput]


class RepairConfig(BaseModel):
    max_retry: int = 5
    max_sample: int = 5
    temperature: float = 0.5
    continue_iterations: bool = False
    no_degrade: bool = False
    allinone: bool = False


class PriorityTags(str, Enum):
    HIGH = "HIGH_PRIORITY"
    MEDIUM = "MEDIUM_PRIORITY"
    LOW = "LOW_PRIORITY"


class PolicyRepairerInput(ComponentInput):
    config: RepairConfig = Field(default_factory=RepairConfig)
    detection_input: PolicyDetectorInput
    detection_output: PolicyDetectorOutput
    weights: Optional[List[float]] = None
    ranks: Optional[List[int]] = None
    tags: Optional[List[PriorityTags]] = None


class PolicyRepairerOutput(ComponentOutput):
    repaired_text: str
    bestofn_attempts: Optional[List] = None
