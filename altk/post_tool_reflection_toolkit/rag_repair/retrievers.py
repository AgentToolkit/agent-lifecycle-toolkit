import os
import logging
import warnings
from abc import ABC, abstractmethod

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from .rag_repair_config import RAGRepairComponentConfig

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    @abstractmethod
    def run(self, query, k):
        """This is the main function that does the actual retrieval."""
        pass

    def load_documents(self, file_paths):
        """Utility: Load documents from the file paths provided."""
        # NOTE: Be sure to modify this with the corresponding loader when adding new types of documents
        loaders = []
        for filepath in file_paths:
            if filepath.endswith(".pdf"):
                loaders.append(PyPDFLoader(filepath))
            elif filepath.endswith(".html"):
                loaders.append(BSHTMLLoader(filepath))
            elif filepath.endswith(".jsonl"):
                loaders.append(JSONLoader(filepath, jq_schema=".text", json_lines=True))
            elif filepath.endswith(".json"):
                loaders.append(JSONLoader(filepath, jq_schema=".text"))
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        return docs

    def split_documents(self, text_splitter, docs):
        """Utility: Split the documents into smaller chunks."""
        return text_splitter.split_documents(docs)


class BM25RetrieverTool(BaseRetriever):
    def __init__(self, file_paths, config: RAGRepairComponentConfig):
        # this is set to prevent HF embedding warning when using default tokenizer settings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.file_paths = file_paths
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        logger.info(
            "Initializing retriever with documents, this may take a few minutes..."
        )
        docs = self.load_documents(self.file_paths)
        splits = self.split_documents(self.text_splitter, docs)
        self.retriever = BM25Retriever.from_documents(splits)

    def run(self, query, k):
        self.retriever.k = k
        results = self.retriever.invoke(query)
        return results


class ChromaDBRetrieverTool(BaseRetriever):
    def __init__(self, file_paths, collection_name, config: RAGRepairComponentConfig):
        # this is set to prevent HF embedding warning when using default tokenizer settings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.file_paths = file_paths
        self.collection_name = collection_name
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.persist_directory = config.persist_path
        self.distance_func = config.distance_func
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.embedding = HuggingFaceEmbeddings(model_name=config.embedding_name)
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        if config.reload_docs:
            self.vectordb.reset_collection()
        if len(self.vectordb.get()["documents"]) == 0:
            self.__init_db__()

    def __init_db__(self):
        logger.info(
            "Initializing vector storage with documents, this may take a few minutes..."
        )
        docs = self.load_documents(self.file_paths)
        if not docs:
            return
        splits = self.split_documents(self.text_splitter, docs)
        self.vectordb = self.embed_and_store(splits)

    def load_documents(self, file_paths):
        """Load documents from the file paths provided."""
        if len(file_paths) == 0:
            warnings.warn(
                f"No documents provided for RAG Repair, aborting DB creation for {self.collection_name} collection!",
                stacklevel=2,
            )
            return None
        return super().load_documents(file_paths)

    def embed_and_store(self, splits):
        """Embed the document chunks and store them in a Chroma vector database."""
        collection_metadata = {"hnsw:space": self.distance_func}
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata=collection_metadata,
        )
        return vectordb

    def run(self, query, k=3):
        results = self.vectordb.similarity_search(query, k=k)
        return results
