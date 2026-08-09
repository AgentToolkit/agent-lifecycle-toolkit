from pydantic import Field
from altk.core.toolkit import ComponentConfig
from altk.core.llm import GenerationMode


class TestCaseGenConfig(ComponentConfig):
    """Configuration for the Test Case Generation Middleware."""

    gen_mode: GenerationMode = Field(
        default=GenerationMode.TEXT,
        description="LLM Generation Mode. Currently only GenerationMode.TEXT is supported",
    )
    max_nl_utterances: int = Field(
        default=3,
        description="Maximum number of utterances to be generated for each test case",
        ge=0,
        le=10,
    )
    max_testcases: int = Field(
        default=3,
        description="Maximum number of test case scenarios to be generated",
        ge=0,
        le=10,
    )
    clean_nl_utterances: bool = Field(
        default=False, description="if cleanup required for the generated nl utterances"
    )
    negative_test_cases: bool = Field(
        default=False, description="Whether to generate negative test cases or not"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        extra = "forbid"


# def get_llm_client_obj(model_name="mistralai/mistral-small-3-1-24b-instruct-2503"):
#     WatsonXAIClient = get_llm("watsonx")
#     client = WatsonXAIClient(
#         model_name=model_name,
#         api_key=os.getenv("WX_API_KEY"),
#         project_id=os.getenv("WX_PROJECT_ID"),
#         url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
#     )
#     return client
#
#
# DEFAULT_CONFIGS = {
#     "standard": TestCaseGenConfig(
#         llm_client=get_llm_client_obj(model_name="mistralai/mistral-small-3-1-24b-instruct-2503"),
#         gen_mode=GenerationMode.TEXT,
#         max_nl_utterances=3,
#         max_testcases=3,
#         clean_nl_utterances=True,
#     )
# }
