from typing import Any, Dict, List, Optional, Union

from altk.core.toolkit import ComponentInput, ComponentOutput
from pydantic import BaseModel, Field


class SpotLightConfig(BaseModel):
    """Configuration for the SpotLight component."""

    model_path: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="The HuggingFace model path to use with Spotlight",
    )
    generation_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_new_tokens": 128,
            "do_sample": False,
            "output_attentions": True,
        }
    )


class SpotLightMetadata(BaseModel):
    """
    SpotLight parameters

    emph_strings: Spans to emphasize within the prompt. Multiple spans can be specified as list of lists.
    alpha: Target proportion of attention to emphasize the spans towards.
    """

    emph_strings: Optional[Union[str, List[str], List[List[str]]]] = None
    alpha: float = 0.2


class SpotLightRunInput(ComponentInput):
    """Input for a single SpotLight prediction."""

    metadata: SpotLightMetadata


class SpotLightOutputSchema(BaseModel):
    """Output schema for SpotLight"""

    prediction: str
    metadata: Optional[SpotLightMetadata] = None


class SpotLightRunOutput(ComponentOutput):
    """Output from SpotLight prediction."""

    output: Optional[SpotLightOutputSchema] = None
