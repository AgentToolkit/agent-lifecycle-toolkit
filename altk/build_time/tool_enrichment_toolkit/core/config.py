from pydantic import Field
from altk.core.toolkit import ComponentConfig
from altk.core.llm import GenerationMode, get_llm
import os


class PythonToolEnrichConfig(ComponentConfig):
    """Configuration for python tool enrichment in the Tool Enrichment Middleware."""

    gen_mode: GenerationMode = Field(
        default=GenerationMode.TEXT,
        description="LLM Generation Mode. Currently only GenerationMode.TEXT is supported",
    )
    enable_tool_description_enrichment: bool = Field(
        default=True,
        description="If the tool description needs to be enriched, set this to True, else False",
    )
    enable_tool_parameter_description_enrichment: bool = Field(
        default=True,
        description="If the tool parameter description needs to be enriched, set this to True, else False",
    )
    enable_tool_return_description_enrichment: bool = Field(
        default=True,
        description="If the tool return description needs to be enriched, set this to True, else False",
    )
    enable_tool_example_enrichment: bool = Field(
        default=True,
        description="If the parameter examples needs to be added as part of enrichment, set this to True, else False",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        extra = "forbid"


class MCPCFToolEnrichConfig(ComponentConfig):
    """Configuration for the MCP CF Tool Enrichment in the Tool Enrichment Middleware."""

    gen_mode: GenerationMode = Field(
        default=GenerationMode.TEXT,
        description="LLM Generation Mode. Currently only GenerationMode.TEXT is supported",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        extra = "forbid"


def get_llm_client_obj(model_name="mistralai/mistral-small-3-1-24b-instruct-2503"):
    WatsonXAIClient = get_llm("watsonx")
    client = WatsonXAIClient(
        model_name=model_name,
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
    )
    return client


DEFAULT_CONFIGS = {
    "python": PythonToolEnrichConfig(
        llm_client=get_llm_client_obj(
            model_name="mistralai/mistral-small-3-1-24b-instruct-2503"
        ),
        gen_mode=GenerationMode.TEXT,
        enable_tool_description_enrichment=True,
        enable_tool_parameter_description_enrichment=True,
        enable_tool_return_description_enrichment=True,
        enable_tool_example_enrichment=True,
    ),
    "mcpcf": MCPCFToolEnrichConfig(
        llm_client=get_llm_client_obj(
            model_name="mistralai/mistral-small-3-1-24b-instruct-2503"
        ),
        gen_mode=GenerationMode.TEXT,
    ),
}
