from pydantic import Field
from altk.core.toolkit import ComponentConfig
from typing import Literal


class RAGRepairComponentConfig(ComponentConfig):
    """Configuration for RAGRepair, note that LLMClient should already be provided"""

    # RAGRepair Configuration
    retrieval_type: Literal["chroma", "bm25"] = Field(
        default="chroma",
        description="Chooses retrieval type between embedding vector store or BM25 ranking",
    )
    persist_path: str = Field(
        default=".chroma",
        description="Path on local filesystem to create vector storage for RAG",
    )
    reload_docs: bool = Field(
        default=False,
        description="Set True to flush local storage and reload documents. Useful if local documents have changed.",
    )
    docs_filter: Literal["all", "doc", "man"] = Field(
        default="all",
        description="Select which types of documents to retrieve from RAG.",
    )

    # RAG Configuration
    chunk_size: int = Field(
        default=1500,
        description="Chunk size in characters used when splitting documents.",
    )
    chunk_overlap: int = Field(
        default=150,
        description="Number of characters that overlap between two adjacent chunks when splitting documents.",
    )
    embedding_name: str = Field(
        default="ibm-granite/granite-embedding-107m-multilingual",
        description="Embedding model used in vector storage. Expects a huggingface model id. Does nothing for BM25.",
    )
    distance_func: str = Field(
        default="l2",
        description="Distance function calculation used by vector storage. Does nothing for BM25.",
    )
