import random
import string
from typing import cast
from altk.core.toolkit import AgentPhase
from altk.post_tool.code_generation.code_generation import (
    CodeGenerationComponent,
    CodeGenerationComponentConfig,
)
from altk.post_tool.core.toolkit import (
    CodeGenerationRunInput,
    CodeGenerationRunOutput,
)
from altk.core.llm.base import get_llm
from dotenv import load_dotenv
import pytest
import os

load_dotenv()


class TestLongResponse:
    @pytest.fixture
    def llm_client(self):
        WATSONX_CLIENT = get_llm("watsonx")
        llm_client = WATSONX_CLIENT(
            model_id="meta-llama/llama-3-3-70b-instruct",
            api_key=os.getenv("WX_API_KEY"),
            project_id=os.getenv("WX_PROJECT_ID"),
            url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
        )
        yield llm_client

    def test_code_generation(self, llm_client):
        """Demonstrates how to use CodeGenerationComponent"""

        def generate_random_string(length: int = 10):
            """Generates a random string of specified length."""

            characters = string.ascii_letters + string.digits
            random_string = "".join(random.choice(characters) for _ in range(length))
            return random_string

        user_query = "What is the coldest city?"

        num_repetitions = 100
        response = [
            {
                "city": f"{generate_random_string(10)}",
                "temperature": f"{random.randint(5, 10)}°C",
            }
            for _ in range(num_repetitions)
        ]

        config = CodeGenerationComponentConfig(
            llm_client=llm_client, use_docker_sandbox=False
        )
        middleware = CodeGenerationComponent(config=config)

        input_data = CodeGenerationRunInput(
            messages=[], nl_query=user_query, tool_response=response
        )
        output = cast(
            CodeGenerationRunOutput, middleware.process(input_data, AgentPhase.RUNTIME)
        )

        print(output.result)
        assert output is not None
        assert output.result is not None
