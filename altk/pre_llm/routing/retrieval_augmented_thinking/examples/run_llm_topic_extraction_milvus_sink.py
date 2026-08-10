import json
import logging
import os
from pathlib import Path
import tempfile
from chromadb import EphemeralClient
from chromadb.config import Settings
from altk.pre_llm.routing.retrieval_augmented_thinking.milvus.topic_retriever import (
    MilvusTopicRetriever,
)
from altk.core.toolkit import AgentPhase
from altk.core.llm import get_llm
from altk.pre_llm.core.types import (
    TopicExtractionBuildOutput,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.llm import (
    LLMTopicExtractorOutput,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.settings import (
    TopicExtractionSettings,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.topic_extractor import (
    TopicExtractionMiddleware,
)

# To run this example you'll need set env vars with your watsonx credentials:
# export WX_API_KEY=...
# export WX_PROJECT_ID=...
# export WX_URL=https://us-south.ml.cloud.ibm.com⁄
# python altk/retrieval_augmented_thinking_toolkit/examples/run_llm_topic_extraction_milvus_sink.py


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    force=True,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

client = EphemeralClient(settings=Settings(allow_reset=True))

five_docs = json.loads((Path(__file__).parent / "chunks.json").read_text())[:5]

logger.info("Storing 5 chunks from source document into chroma")
collection = client.create_collection("sound")
collection.add(
    ids=[str(i) for i, _ in enumerate(five_docs)],
    documents=five_docs,
)
WatsonXAIClient = get_llm("watsonx")
llm_client = WatsonXAIClient(
    model_id="meta-llama/llama-3-3-70b-instruct",
    api_key=os.getenv("WX_API_KEY"),
    project_id=os.getenv("WX_PROJECT_ID"),
    url=os.getenv("WX_URL"),
)
with tempfile.NamedTemporaryFile(suffix=".db") as milvus_db:
    topic_extraction_settings = {
        "subject": "sound",
        "topic_extractor": {
            "name": "llm",
            "config": {
                "content_provider": {
                    "name": "altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider.ChromaDBProvider",
                    "config": {
                        "collection": "sound",
                        "instance": {"client": client},
                        "n_docs": 3,
                    },
                },
            },
        },
        "topics_sink": {
            "name": "altk.pre_llm.routing.retrieval_augmented_thinking.milvus.topic_sink.MilvusProvider",
            "config": {
                "collection": "sound_topics",
                "uri": milvus_db.name,
            },
        },
    }
    topic_extractor = TopicExtractionMiddleware.from_settings(
        TopicExtractionSettings(llm_client=llm_client, **topic_extraction_settings)
    )

    logger.info("Extracting topics from chunks")
    topic_extraction_output: TopicExtractionBuildOutput[LLMTopicExtractorOutput] = (
        topic_extractor.process(data=None, phase=AgentPhase.BUILDTIME)
    )
    if topic_extraction_output.error:
        logger.error(f"{topic_extraction_output.error=}")

    # Get the extracted topics using a Milvus Topic Retriever
    topic_retriever = MilvusTopicRetriever(
        milvus_db.name, metadata_fields=["expertise"], collection_name="sound_topics"
    )
    retrieved_topics = topic_retriever.get_topics("Sound perception")
    topics_lst = "\n".join(
        [
            f"{retrieved_topic.topic.topic}: {retrieved_topic.topic.subject} ({retrieved_topic.topic.expertise})"
            for retrieved_topic in retrieved_topics
        ]
    )
    logger.info(f"Topics related to 'Sound perception':\n{topics_lst}")
