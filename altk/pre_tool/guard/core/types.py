from typing import Union

from altk.toolkit_core.core.toolkit import ComponentInput, ComponentOutput
from altk.toolkit_core.llm import LLMClient
from pydantic import BaseModel, Field, ConfigDict


class ToolGuardBuildInputMetaData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    policy_text: str = Field(description="Text of the policy document file")
    short1: bool = Field(default=True, description="Run build short or long version. ")
    validating_llm_client: LLMClient = Field(
        description="ValidatingLLMClient for build time"
    )


class ToolGuardBuildInput(ComponentInput):
    metadata: ToolGuardBuildInputMetaData = Field(
        default_factory=lambda: ToolGuardBuildInputMetaData()
    )


class ToolGuardRunInputMetaData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    tool_name: str = Field(description="Tool name")
    tool_parms: dict = Field(default={}, description="Tool parameters")
    llm_client: LLMClient = Field(description="LLMClient for build time")


class ToolGuardRunInput(ComponentInput):
    metadata: ToolGuardRunInputMetaData = Field(
        default_factory=lambda: ToolGuardRunInputMetaData()
    )


class ToolGuardRunOutputMetaData(BaseModel):
    error_message: Union[str, bool] = Field(
        description="Error string or False if no error occurred"
    )


class ToolGuardRunOutput(ComponentOutput):
    output: ToolGuardRunOutputMetaData = Field(
        default_factory=lambda: ToolGuardRunOutputMetaData()
    )
