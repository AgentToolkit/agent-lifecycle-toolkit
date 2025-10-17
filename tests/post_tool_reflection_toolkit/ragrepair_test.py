from typing import cast
from dotenv import load_dotenv
import os

import pytest
from altk.core.llm.base import get_llm
from altk.core.toolkit import AgentPhase
from altk.post_tool.rag_repair.rag_repair import RAGRepairComponent
from altk.post_tool.rag_repair.rag_repair_config import (
    RAGRepairComponentConfig,
)
from altk.post_tool.core.toolkit import (
    RAGRepairRunInput,
    RAGRepairRunOutput,
    RAGRepairBuildInput,
)


load_dotenv()


class TestRAGRepair:
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

    def test_no_docs(self, tmp_path, llm_client):
        # Set up RAG
        config = RAGRepairComponentConfig(
            llm_client=llm_client, persist_path=os.path.join(tmp_path, ".chroma")
        )
        repairer = RAGRepairComponent(docs_path="thispathdoesnotexist", config=config)
        with pytest.warns(UserWarning):
            repairer.process(RAGRepairBuildInput(), AgentPhase.BUILDTIME)

        query = "Get all alerts in the otel-demo namespace"
        cmd = "kubectl get alerts -n otel-demo"
        response = 'error: the server doesn\'t have a resource type "alerts"'
        input_data = RAGRepairRunInput(nl_query=query, tool_call=cmd, error=response)
        result = cast(
            RAGRepairRunOutput, repairer.process(input_data, AgentPhase.RUNTIME)
        )
        assert result.retrieved_docs == ""

    def test_docs(self, tmp_path, llm_client):
        config = RAGRepairComponentConfig(
            llm_client=llm_client, persist_path=os.path.join(tmp_path, ".chroma")
        )
        # NOTE: path may be brittle, this might fail in CI/CD
        repairer = RAGRepairComponent(
            docs_path=os.path.join(
                "tests", "post_tool_reflection_toolkit", "rag_test_docs"
            ),
            config=config,
        )
        repairer.process(RAGRepairBuildInput(), AgentPhase.BUILDTIME)

        query = "Get all alerts in the otel-demo namespace"
        cmd = "kubectl get alerts -n otel-demo"
        response = 'error: the server doesn\'t have a resource type "alerts"'
        input_data = RAGRepairRunInput(nl_query=query, tool_call=cmd, error=response)
        result = repairer.process(input_data, AgentPhase.RUNTIME)
        assert result.retrieved_docs is not None

    def test_messages(self, tmp_path, llm_client):
        config = RAGRepairComponentConfig(
            llm_client=llm_client, persist_path=os.path.join(tmp_path, ".chroma")
        )
        # NOTE: path may be brittle, this might fail in CI/CD
        repairer = RAGRepairComponent(
            docs_path=os.path.join(
                "tests", "post_tool_reflection_toolkit", "rag_test_docs"
            ),
            config=config,
        )
        repairer.process(RAGRepairBuildInput(), AgentPhase.BUILDTIME)

        messages = [
            {"role": "user", "content": "Check on otel-demo"},
            {
                "role": "assistant",
                "content": "Get all alerts in the otel-demo namespace",
            },
        ]
        cmd = "kubectl get alerts -n otel-demo"
        response = 'error: the server doesn\'t have a resource type "alerts"'
        input_data = RAGRepairRunInput(messages=messages, tool_call=cmd, error=response)
        result = repairer.process(input_data, AgentPhase.RUNTIME)
        assert result.retrieved_docs is not None

    def test_original_function(self, tmp_path, llm_client):
        def reprint(cmd: str):
            return cmd

        config = RAGRepairComponentConfig(
            llm_client=llm_client, persist_path=os.path.join(tmp_path, ".chroma")
        )
        # NOTE: path may be brittle, this might fail in CI/CD
        repairer = RAGRepairComponent(
            docs_path=os.path.join(
                "tests", "post_tool_reflection_toolkit", "rag_test_docs"
            ),
            config=config,
        )
        repairer.process(RAGRepairBuildInput(), AgentPhase.BUILDTIME)
        query = "Get all alerts in the otel-demo namespace"
        cmd = "kubectl get alerts -n otel-demo"
        response = 'error: the server doesn\'t have a resource type "alerts"'
        data = RAGRepairRunInput(
            original_function=reprint, nl_query=query, tool_call=cmd, error=response
        )
        result = repairer.process(data, AgentPhase.RUNTIME)
        # should be the same because the original_function just returns the tool call
        assert result.result == result.new_tool_call
