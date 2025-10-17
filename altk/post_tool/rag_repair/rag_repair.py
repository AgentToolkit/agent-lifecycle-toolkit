"""RAGRepair Component
Need to provide a path to documents, if there is a "doc" and/or "man" folder, will separate the docs into two collections,
otherwise will just use the "doc" collection.
"""

import logging
import json
from typing import Set, Any, Literal
import os
from glob import glob

from pydantic import Field

from altk.post_tool.rag_repair.prompts import RAG_REPAIR_PROMPT
from altk.core.toolkit import AgentPhase
from altk.post_tool.core.toolkit import (
    PostToolReflectionComponent,
    RAGRepairRunInput,
    RAGRepairRunOutput,
    RAGRepairBuildInput,
    RAGRepairBuildOutput,
)
from .retrievers import BM25RetrieverTool, ChromaDBRetrieverTool
from .rag_repair_config import RAGRepairComponentConfig

logger = logging.getLogger(__name__)
# if modifying this, be sure to also modify load_documents() in retrievers.py
DOC_TYPES = ("**/*.html", "**/*.pdf", "**/*.jsonl", "**/*.json")


class RAGRepairComponent(PostToolReflectionComponent):
    docs_path: str
    config: RAGRepairComponentConfig = Field(default_factory=RAGRepairComponentConfig)
    _retriever_man: Any | None = None
    _retriever_docs: Any | None = None

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        return {AgentPhase.BUILDTIME, AgentPhase.RUNTIME}

    def setup_rag(self):
        if self.config.docs_filter != "man":
            self.setup_db(doc_type="doc")
        if self.config.docs_filter != "doc":
            self.setup_db(doc_type="man")

    def setup_db(self, doc_type: Literal["doc", "man"] = "doc"):
        file_paths = []
        docs_path = self.docs_path
        if doc_type == "man":
            man_path = os.path.join(docs_path, "man")
            if os.path.isdir(man_path):
                for files in DOC_TYPES:
                    for fpath in glob(files, root_dir=man_path, recursive=True):
                        file_paths.append(os.path.join(man_path, fpath))
                if self.config.retrieval_type == "bm25":
                    self._retriever_man = BM25RetrieverTool(
                        file_paths, config=self.config
                    )
                else:
                    self._retriever_man = ChromaDBRetrieverTool(
                        file_paths, collection_name="man", config=self.config
                    )
        elif doc_type == "doc":
            other_doc_path = os.path.join(docs_path, "doc")
            if os.path.isdir(other_doc_path):
                for files in DOC_TYPES:
                    for fpath in glob(files, root_dir=other_doc_path, recursive=True):
                        file_paths.append(os.path.join(other_doc_path, fpath))
            else:
                # doc is also default doc_type, if no directories are defined, just use this
                for files in DOC_TYPES:
                    for fpath in glob(files, root_dir=docs_path, recursive=True):
                        file_paths.append(os.path.join(docs_path, fpath))
            if self.config.retrieval_type == "bm25":
                self._retriever_docs = BM25RetrieverTool(file_paths, config=self.config)
            self._retriever_docs = ChromaDBRetrieverTool(
                file_paths, collection_name="doc", config=self.config
            )

    def run_rag(self, question: str, doc_type: Literal["doc", "man"] = "doc"):
        # doc_type is "man" or "doc"
        k = 3
        if doc_type == "man":
            k = 1
            # possible to not have the retriever set up if no docs found
            if self._retriever_man:
                results = self._retriever_man.run(question, k=k)
            else:
                results = []
        else:
            if self._retriever_docs:
                results = self._retriever_docs.run(question, k=k)
            else:
                results = []

        agg_ret = ""
        # Print the results and concatenate retrieved chunks
        for idx, doc in enumerate(results):
            logger.info(f"Result {idx + 1}: {doc.page_content}")
            agg_ret += doc.page_content

        return agg_ret

    def repair(self, data: RAGRepairRunInput, **kwargs: Any) -> RAGRepairRunOutput:
        new_cmd = data.tool_call

        rag_man_result = ""
        rag_result = ""
        if self.config.docs_filter != "doc":
            rag_man_result = self.run_rag(data.tool_call, "man")
        if self.config.docs_filter != "man":
            rag_result = self.run_rag(f"fix the error: {data.tool_call} {data.error}")
        agent_scratchpad = ""
        error = ""
        if data.error:
            error = data.error
        retrieved_docs = rag_man_result + "\n" + rag_result
        retrieved_docs = retrieved_docs.strip()

        prompt_query = json.dumps(data.messages)
        if data.nl_query != "":
            prompt_query = data.nl_query

        pirates_rag_repair_prompt = RAG_REPAIR_PROMPT.format(
            rag_result=rag_result,
            rag_man_result=rag_man_result,
            query=prompt_query,
            cmd=data.tool_call,
            error=error,
            agent_scratchpad=agent_scratchpad,
        )
        result = self.config.llm_client.generate(pirates_rag_repair_prompt)
        for line in result.splitlines():
            if line.strip().startswith("Command: "):
                new_cmd = line.split("Command: ")[1]
                break
        new_result = None
        if data.original_function:
            new_result = data.original_function(new_cmd)
        return RAGRepairRunOutput(
            result=new_result, new_tool_call=new_cmd, retrieved_docs=retrieved_docs
        )

    def _build(self, data: RAGRepairBuildInput) -> RAGRepairBuildOutput:  # type: ignore
        self.setup_rag()
        return RAGRepairBuildOutput()

    def _run(self, data: RAGRepairRunInput) -> RAGRepairRunOutput:  # type: ignore
        result = self.repair(data)
        return result
