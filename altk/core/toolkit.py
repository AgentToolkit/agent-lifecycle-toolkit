"""Set of base classes that define the interfaces for Toolkit Components."""

from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Set, Dict, Any
from enum import Enum, auto
from altk.core.llm import get_llm


######### Toolkit Component Interfaces ##############
class AgentPhase(Enum):
    """Enum representing different phases of agent"""

    BUILDTIME = auto()
    RUNTIME = auto()


class ComponentConfig(BaseModel):
    llm_client: Any = Field(
        default_factory=get_llm("auto_from_env")
    )  # More flexible to accept different LLM client types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, ctx: Any) -> None:
        if isinstance(self.llm_client, str):
            # In this scenario, we assume that user supplied a litellm-style model name
            litellm_provider = get_llm("litellm")
            self.llm_client = litellm_provider(model_name=self.llm_client)


class ComponentInput(BaseModel):
    """
    Shared base input for component at runtime.
    """

    messages: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any] | BaseModel] = None


class ComponentOutput(BaseModel):
    """
    Shared base output for component at runtime.
    """

    output: Optional[Dict[str, Any] | BaseModel] = None


class ComponentBase(ABC, BaseModel):
    config: Optional[ComponentConfig] = Field(default_factory=ComponentConfig)

    @classmethod
    @abstractmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        """
        Returns a set of AgentPhase that the component supports.
        This method must be implemented by subclasses to specify which phases
        the middleware can handle.
        Returns:
            Set[AgentPhase]: A set of supported AgentPhase.
        """
        pass

    def process(self, data: ComponentInput, phase: AgentPhase) -> ComponentOutput:
        """
        Process the input data based on the specified phase.
        Args:
            data (ComponentInput): Input data for the component.
            phase (AgentPhase): The phase in which the component is being executed.

        Returns:
            ComponentOutput: Processed output data.
        Raises:
            ValueError: If the phase is not supported by the component.
            NotImplementedError: If the phase is not handled in the component.
        """
        if phase not in self.supported_phases():
            raise ValueError(
                f"{self.__class__.__name__} does not support phase {phase}"
            )

        if phase == AgentPhase.BUILDTIME:
            return self._build(data)
        elif phase == AgentPhase.RUNTIME:
            return self._run(data)

        raise NotImplementedError(f"Unhandled phase: {phase}")

    async def aprocess(
        self, data: ComponentInput, phase: AgentPhase
    ) -> ComponentOutput:
        """
        Async Process the input data based on the specified phase.
        Args:
            data (ComponentInput): Input data for the component.
            phase (AgentPhase): The phase in which the component is being executed.

        Returns:
            ComponentOutput: Processed output data.
        Raises:
            ValueError: If the phase is not supported by the component.
            NotImplementedError: If the phase is not handled in the component.
        """
        if phase not in self.supported_phases():
            raise ValueError(
                f"{self.__class__.__name__} does not support phase {phase}"
            )

        if phase == AgentPhase.BUILDTIME:
            return await self._abuild(data)
        elif phase == AgentPhase.RUNTIME:
            return await self._arun(data)

        raise NotImplementedError(f"Unhandled phase: {phase}")

    def _build(self, data: ComponentInput) -> ComponentOutput:
        """
        Default build method that can be overridden by subclasses.
        This method is called during the BUILDTIME phase.
        Args:
            data (ComponentInput): Input data for the build phase.
        """
        raise NotImplementedError

    def _run(self, data: ComponentInput) -> ComponentOutput:
        """
        Default run method that can be overridden by subclasses.
        This method is called during the RUNTIME phase.
        Args:
            data (ComponentInput): Input data for the run phase.
        """
        raise NotImplementedError

    async def _abuild(self, data: ComponentInput) -> ComponentOutput:
        """
        Async build method that can be overridden by subclasses.
        This method is called during the BUILDTIME phase.
        Args:
            data (ComponentInput): Input data for the build phase.
        """
        raise NotImplementedError

    async def _arun(self, data: ComponentInput) -> ComponentOutput:
        """
        Async run method that can be overridden by subclasses.
        This method is called during the RUNTIME phase.
        Args:
            data (ComponentInput): Input data for the run phase.
        """
        raise NotImplementedError
