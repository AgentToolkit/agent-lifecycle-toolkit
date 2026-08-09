import os
from altk.core.llm.base import get_llm
import pytest


@pytest.fixture
def llm_client():
    WatsonXAIClient = get_llm("watsonx")
    llm_client = WatsonXAIClient(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL"),
        hooks=[lambda ev, data: print(ev, data)],
    )
    return llm_client
