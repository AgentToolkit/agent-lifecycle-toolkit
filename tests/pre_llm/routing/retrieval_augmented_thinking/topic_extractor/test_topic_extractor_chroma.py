import logging
import os
from typing import Callable, List, cast

import pytest
from chromadb import GetResult
from chromadb.api import ClientAPI

from altk.pre_llm.core.types import TopicInfo
from altk.pre_llm.routing.retrieval_augmented_thinking.chroma.common import (
    EphemeralChromaDBConfig,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider import (
    DEAULT_OFFSET,
    DEFAULT_PAGE_SIZE,
    ChromaDBProvider,
    ChromaDBProviderSettings,
)
from altk.pre_llm.core.types import (
    TopicExtractionBuildOutput,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.llm import (
    LLMTopicExtractor,
    LLMTopicExtractorOutput,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.settings import (
    ContentProviderSettings,
    LLMTopicExtractorSettings,
    TopicExtractionSettings,
    TopicExtractorSettings,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.topic_extractor import (
    TopicExtractionMiddleware,
    remove_dup_topics,
)
from altk.core.toolkit import AgentPhase, ComponentInput
from altk.core.llm import get_llm
from altk.core.llm.base import LLMClient


@pytest.mark.parametrize(
    "provider,model_id",
    [
        pytest.param(
            "litellm.rits",
            "ibm-granite/granite-3.3-8b-instruct",
            marks=pytest.mark.skipif(
                os.getenv("RUN_THIS", "false").lower() == "false",
                reason="RUN_THIS env var is false",
            ),
        ),
        ("watsonx", "meta-llama/llama-3-3-70b-instruct"),
        ("litellm.watsonx", "meta-llama/llama-3-3-70b-instruct"),
    ],
)
def test_topic_extractor_same_chromadb(
    caplog,
    chunks: List[str],
    create_chroma: Callable[..., ClientAPI],
    provider,
    model_id,
):
    caplog.set_level(logging.DEBUG, logger=ChromaDBProvider.__module__)
    caplog.set_level(logging.DEBUG, logger=TopicExtractionMiddleware.__module__)
    caplog.set_level(logging.DEBUG, logger=LLMTopicExtractor.__module__)

    # populate the DB
    client = create_chroma("sound", chunks[:10])

    chroma_db_provider = ChromaDBProvider(
        collection="sound",
        dest_collection="sound.topics",
        client=client,
        n_docs=5,
    )
    llm_client: LLMClient | None = None
    if provider == "litellm.rits":
        RITSLiteLLMClient = get_llm("litellm.rits")

        llm_client = RITSLiteLLMClient(
            model_name=model_id,
            model_url=model_id.split("/")[1]
            .replace(".", "-")
            .lower(),  # Llama-3.1-8B-Instruct -> llama-3-1-8b-instruct
            # hooks=[lambda event, payload: print(f"[RITS] {event}: {payload}")],
        )
    elif provider == "watsonx":
        WatsonXAIClient = get_llm("watsonx")
        llm_client = WatsonXAIClient(
            model_id=model_id,
            api_key=os.getenv("WX_API_KEY"),
            project_id=os.getenv("WX_PROJECT_ID"),
            url=os.getenv("WX_URL"),
            # hooks=[lambda event, payload: print(f"[SYNC HOOK] {event}: {payload}")],
        )
    elif provider == "litellm.watsonx":
        WatsonXLiteLLMClient = get_llm("litellm.watsonx")

        llm_client = WatsonXLiteLLMClient(
            model_name=model_id,
            # hooks=[lambda event, payload: print(f"[SYNC HOOK] {event}: {payload}")],
        )
    else:
        raise ValueError(f"Unsupported LLM model provider {provider}")

    topic_extractor = LLMTopicExtractor(
        content_provider=chroma_db_provider,
        llm_client=llm_client,
    )
    topic_extraction = TopicExtractionMiddleware[
        ComponentInput, LLMTopicExtractorOutput
    ](subject="sound", topics_sink=chroma_db_provider, topic_extractor=topic_extractor)
    topic_extraction_output: TopicExtractionBuildOutput[LLMTopicExtractorOutput] = (
        topic_extraction.process(data=None, phase=AgentPhase.BUILDTIME)
    )
    print(f"{topic_extraction_output.topic_extractor_output.chunks_processed=}")
    print(f"{topic_extraction_output.error=}")
    for (
        chunk_number,
        chunk_stat,
    ) in topic_extraction_output.topic_extractor_output.chunk_stats.items():
        if chunk_stat.error:
            print(f"Processing of chunk {chunk_number} had errors: {chunk_stat.error}")

    if "sound.topics" in [col.name for col in client.list_collections()]:
        # Retrieve the extracted topics from the target chroma db collection:
        topics: GetResult = client.get_collection("sound.topics").get()
        print()
        print("Extracted topics:")
        for doc, metadata in [
            (doc, metadata)
            for doc, metadata in zip(topics["documents"], topics["metadatas"])
        ]:
            print()
            print(f"Topic: {doc}")
            print(f"Metadata: {metadata}")
    else:
        print("sound.topics collection not found")


def test_topic_extractor_different_chromadb(
    caplog,
    chunks,
    create_chroma: Callable[..., ClientAPI],
    chroma: ClientAPI,
):
    caplog.set_level(logging.DEBUG, logger=ChromaDBProvider.__module__)
    caplog.set_level(logging.DEBUG, logger=TopicExtractionMiddleware.__module__)

    # populate the DB
    client = create_chroma("sound", chunks[:10])

    chroma_db_provider = ChromaDBProvider(
        collection="sound",
        client=client,
        n_docs=5,
    )
    chroma_db_sink = ChromaDBProvider(
        collection="sound.topics",
        client=chroma,
        n_docs=5,
    )
    WatsonXAIClient = get_llm("watsonx")
    llm_client = WatsonXAIClient(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL"),
    )
    topic_extractor = LLMTopicExtractor(
        content_provider=chroma_db_provider,
        llm_client=llm_client,
    )
    topic_extractor_extraction = TopicExtractionMiddleware(
        subject="sound",
        topic_extractor=topic_extractor,
        topics_sink=chroma_db_sink,
    )

    topic_extraction_output: TopicExtractionBuildOutput[LLMTopicExtractorOutput] = (
        topic_extractor_extraction.process(data=None, phase=AgentPhase.BUILDTIME)
    )

    print(f"{topic_extraction_output.topic_extractor_output.chunks_processed=}")
    print(f"{topic_extraction_output.error=}")
    for (
        chunk_number,
        chunk_stat,
    ) in topic_extraction_output.topic_extractor_output.chunk_stats.items():
        if chunk_stat.error:
            print(f"Processing of chunk {chunk_number} had errors: {chunk_stat.error}")

    if "sound.topics" in [col.name for col in client.list_collections()]:
        # list the extracted topics stored in the target collection
        topics: GetResult = chroma.get_collection("sound.topics").get()
        print()
        print("Extracted topics:")
        for doc, metadata in [
            (doc, metadata)
            for doc, metadata in zip(topics["documents"], topics["metadatas"])
        ]:
            print()
            print(f"Topic: {doc}")
            print(f"Metadata: {metadata}")
    else:
        print("sound.topics collection not found")


def test_topic_extractor_settings(caplog, create_chroma: Callable[..., ClientAPI]):
    caplog.set_level(logging.DEBUG, logger=ChromaDBProvider.__module__)
    caplog.set_level(logging.DEBUG, logger=TopicExtractionMiddleware.__module__)

    # populate the DB
    client = create_chroma("sound", ["chunk_1", "chunk_2"])

    WatsonXAIClient = get_llm("watsonx")
    llm_client = WatsonXAIClient(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL"),
    )

    # Use Pydantic model settings directly
    topic_extractor_settings = TopicExtractionSettings(
        llm_client=llm_client,
        subject="sound",
        topic_extractor=TopicExtractorSettings(
            name="llm",
            config=LLMTopicExtractorSettings(
                model_id="meta-llama/llama-3-3-70b-instruct",
                content_provider=ContentProviderSettings(
                    name="altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider.ChromaDBProvider",
                    config=ChromaDBProviderSettings(
                        collection="sound",
                        dest_collection="sound.topics",
                        instance=EphemeralChromaDBConfig(client=client),
                    ),
                ),
            ),
        ),
    )
    topic_extraction = TopicExtractionMiddleware.from_settings(topic_extractor_settings)
    topic_extractor: LLMTopicExtractor = topic_extraction.topic_extractor
    assert isinstance(topic_extractor.content_provider, ChromaDBProvider)
    assert topic_extraction.subject == "sound"
    assert topic_extraction.topics_sink is None
    assert topic_extractor.content_provider.chroma_client == client
    assert topic_extractor.content_provider.collection_name == "sound"
    assert topic_extractor.content_provider.dest_collection_name == "sound.topics"
    assert topic_extractor.content_provider.n_docs is None
    assert topic_extractor.content_provider.page_size == DEFAULT_PAGE_SIZE
    assert topic_extractor.content_provider.offset == DEAULT_OFFSET

    # Use dict settings
    topic_extractor_settings = {
        "subject": "sound",
        "topic_extractor": {
            "name": "llm",
            "config": {
                "content_provider": {
                    "name": "altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider.ChromaDBProvider",
                    "config": {
                        "collection": "sound",
                        "dest_collection": "sound.topics",
                        "instance": {"client": client},
                    },
                },
            },
        },
    }
    topic_extraction = TopicExtractionMiddleware.from_settings(
        TopicExtractionSettings(llm_client=llm_client, **topic_extractor_settings)
    )
    assert isinstance(
        cast(LLMTopicExtractor, topic_extraction.topic_extractor).content_provider,
        ChromaDBProvider,
    )


def test_topic_extractor_different_chromadb_using_dict_settings(
    caplog, chunks, create_chroma: Callable[..., ClientAPI], chroma: ClientAPI
):
    caplog.set_level(logging.DEBUG, logger=ChromaDBProvider.__module__)
    caplog.set_level(logging.DEBUG, logger=TopicExtractionMiddleware.__module__)

    # populate the DB
    client = create_chroma("sound", chunks[:10])

    topic_extractor_settings = {
        "subject": "sound",
        "topic_extractor": {
            "name": "llm",
            "config": {
                "content_provider": {
                    "name": "altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider.ChromaDBProvider",
                    "config": {
                        "collection": "sound",
                        "instance": {"client": client},
                        "n_docs": 5,
                    },
                },
            },
        },
        "topics_sink": {
            "name": "altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider.ChromaDBProvider",
            "config": {
                "collection": "sound.topics",
                "instance": {"client": client},
            },
        },
    }
    WatsonXAIClient = get_llm("watsonx")
    llm_client = WatsonXAIClient(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL"),
    )

    topic_extractor = TopicExtractionMiddleware.from_settings(
        TopicExtractionSettings(llm_client=llm_client, **topic_extractor_settings)
    )

    topic_extraction_output: TopicExtractionBuildOutput[LLMTopicExtractorOutput] = (
        topic_extractor.process(data=None, phase=AgentPhase.BUILDTIME)
    )
    print(f"{topic_extraction_output.topic_extractor_output.chunks_processed=}")
    print(f"{topic_extraction_output.error=}")
    for (
        chunk_number,
        chunk_stat,
    ) in topic_extraction_output.topic_extractor_output.chunk_stats.items():
        if chunk_stat.error:
            print(f"Processing of chunk {chunk_number} had errors: {chunk_stat.error}")

    # list the extracted topics returned in the output
    for topic in topic_extraction_output.topics:
        print(topic)

    # list the extracted topics stored in the target collection
    if "sound.topics" in [col.name for col in client.list_collections()]:
        topics: GetResult = client.get_collection("sound.topics").get()
        print()
        print("Extracted topics:")
        for doc, metadata in [
            (doc, metadata)
            for doc, metadata in zip(topics["documents"], topics["metadatas"])
        ]:
            print()
            print(f"Topic: {doc}")
            print(f"Metadata: {metadata}")
    else:
        print("sound.topics collection not found")


def test_topic_extractor_wo_levels_of_expertise(
    caplog, chunks, create_chroma: Callable[..., ClientAPI], chroma: ClientAPI
):
    caplog.set_level(logging.DEBUG, logger=ChromaDBProvider.__module__)
    caplog.set_level(logging.DEBUG, logger=TopicExtractionMiddleware.__module__)

    # populate the DB
    client = create_chroma("sound", chunks[:10])

    topic_extractor_settings = {
        "subject": "sound",
        "topic_extractor": {
            "name": "llm",
            "config": {
                "content_provider": {
                    "name": "altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider.ChromaDBProvider",
                    "config": {
                        "collection": "sound",
                        "instance": {"client": client},
                        "n_docs": 2,
                    },
                },
                "levels_of_expertise": False,
            },
        },
        "topics_sink": {
            "name": "altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider.ChromaDBProvider",
            "config": {
                "collection": "sound.topics",
                "instance": {"client": client},
            },
        },
    }
    WatsonXAIClient = get_llm("watsonx")
    llm_client = WatsonXAIClient(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL"),
    )

    topic_extraction = TopicExtractionMiddleware.from_settings(
        TopicExtractionSettings(llm_client=llm_client, **topic_extractor_settings)
    )
    assert not cast(
        LLMTopicExtractor, topic_extraction.topic_extractor
    ).levels_of_expertise

    topic_extraction_output: TopicExtractionBuildOutput[LLMTopicExtractorOutput] = (
        topic_extraction.process(data=None, phase=AgentPhase.BUILDTIME)
    )
    assert all([topic.expertise is None for topic in topic_extraction_output.topics])

    if "sound.topics" in [col.name for col in client.list_collections()]:
        # list the extracted topics stored in the target collection
        topics: GetResult = client.get_collection("sound.topics").get()
        print()
        print("Extracted topics:")
        for doc, metadata in [
            (doc, metadata)
            for doc, metadata in zip(topics["documents"], topics["metadatas"])
        ]:
            print()
            print(f"Topic: {doc}")
            print(f"Metadata: {metadata}")
    else:
        print("sound.topics collection not found")


def test_remove_topic_dups():
    topics = [
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="mentions", subject="collab_1"),
    ]
    topics_dedup = remove_dup_topics(topics)
    assert [
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="mentions", subject="collab_1"),
    ] == topics_dedup

    topics = [
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="mentions", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="mentions", subject="collab_1"),
    ]
    topics_dedup = remove_dup_topics(topics)
    assert [
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="mentions", subject="collab_1"),
    ] == topics_dedup

    topics = [
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="knowledge", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="knowledge", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="mentions", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="mentions", subject="collab_1"),
    ]
    topics_dedup = remove_dup_topics(topics)
    assert [
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="knowledge", subject="collab_1"),
    ] == topics_dedup

    topics = [
        TopicInfo(topic="topic_1", expertise="mentions", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="mentions", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="knowledge", subject="collab_1"),
        TopicInfo(topic="topic_2", expertise="knowledge", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
    ]
    topics_dedup = remove_dup_topics(topics)
    assert [
        TopicInfo(topic="topic_2", expertise="knowledge", subject="collab_1"),
        TopicInfo(topic="topic_1", expertise="expert", subject="collab_1"),
    ] == topics_dedup

    topics_dedup = remove_dup_topics([])
    assert topics_dedup == []


def test_remove_topic_dups_wo_levels_of_expertise():
    topics = [
        TopicInfo(topic="topic_1", subject="collab_1"),
    ]
    topics_dedup = remove_dup_topics(topics)
    assert [
        TopicInfo(topic="topic_1", subject="collab_1"),
    ] == topics_dedup

    topics = [
        TopicInfo(topic="topic_1", subject="collab_1"),
        TopicInfo(topic="topic_1", subject="collab_1"),
        TopicInfo(topic="topic_2", subject="collab_1"),
    ]
    topics_dedup = remove_dup_topics(topics)
    assert [
        TopicInfo(topic="topic_1", subject="collab_1"),
        TopicInfo(topic="topic_2", subject="collab_1"),
    ] == topics_dedup
