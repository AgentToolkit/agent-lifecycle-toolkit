# Retrieval Augmented Thinking

A middleware solution to improve routing accuracy for manager agents that contain "difficult-to-describe" collaborator agents, such as those implement RAG. Retrieval-Augmented Thinking is made up of both a  **build-time** component (extracting topics of expertise) and a **run-time** component (Retrieval Agumented Thinking) to improve collaborator selection.


## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [When to Use This Component](#when-to-use-this-component)
- [How it Works](#how-it-works)
- [Quick Start](#quick-start)
- [License](#license)
- [Under the Hood](#under-the-hood)

## Installation

Make sure the dependencies for the Routing components are included by running `pip install "agent-lifecycle-toolkit[routing]"`.

## Overview

Retrieval-Augmented Thinking is a technique that uses _dynamic instruction_ at runtime to give hints to the agent regarding which collaborator(s) might be best to consult to answer a user query.  This approach has proven highly effective for improving collaborator invocations for agents that have broad areas of expertise, such as RAG-based agents.

## When to Use this Component

There are three scenarios where Retrieval-Augmented Thinking (RAT) can be effective.

1. **RAG Agents**
Regrieval-Augmented Thinking has proven particularly effective for managing collaborator agents that perform RAG, because RAG agents can often converse about hundreds, or thousands of topics, as well as contain acronyms and domain-specific terminology that the base LLM does not understand.  It is not easy to describe RAG agent's expertise in short, human-written natural language description, thus these agents benefit greatly from Retrieval-Augmented Thinking.

1. **Agents with documentation that describes their capabilities**
Retrieval-Augmented Thinking can be used for collaborator agents that have some form of documentation that describes their capabilities and areas of expertise.  For example, [Wolfram Alpha](https://www.wolframalpha.com/) is a service that can answer questions about thousands of topics, and those topics are documented in detail on their web page.  If Wolfram Alpha were used as a Collaborator Agent, its documentation could be used to drive Retrieval-Augmented Thinking.

1. **Agents with logs of user queries**
If an agent does not properly describe its capabilities, it becomes more difficult to route to. However, if user logs are available it is still possible to extract topics of expertise from these user logs and agent responses.   If agent responses are not available, the agent can be executed offline to generate them, and the agent responses can be analyzed for topics of expertise


## How it Works

Retrieval-Augmented Thinking operates in two steps

1. **Build time**
During agent build, content (documents, etc) for each route must be made available to a build time topic extraction processor, which extracts topics of expertise and stores them in a vector store for later retrieval.  Given that documents are often stored in vector stores for RAG implementations, we provide a sample implementation of processing content directly from a ChromaDB vector store.

2. **Runtime**
Simply select a RAT-enabled WxO Style for your agent to leverage Retrieval-Augmented thinking in production.  When a query comes in, the vector store that has been populated with "Topics of Expertise" is searched, and the most relevant collaborator expertise is summarized and dynamically injected into the thinking prompt.  This provides more information to the AI Agent to make good routing choices and produce the best possible plan.

## Quick Start

1. Extract the topics of expertise for a hypothetical tool that knows about sound (`tool_sound`) from documents stored in a ChromaDB collection "sound" and store the extracted topics in the "topics" collection. In the topic extraction, the "subject" is the name of the tools for which the topics are being extracted.
```python
import os
from altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_sink_content_provider import (
    ChromaDBProvider,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.core.toolkit import (
    TopicExtractionBuildOutput,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.llm import (
    LLMTopicExtractor,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_extractor.topic_extractor import (
    TopicExtractionMiddleware,
)
from altk.core.llm.base import get_llm
from altk.core.toolkit import AgentPhase


WatsonXAIClient = get_llm("watsonx")
llm_client = WatsonXAIClient(
    model_id="meta-llama/llama-3-3-70b-instruct",
    api_key=os.getenv("WX_API_KEY"),
    project_id=os.getenv("WX_PROJECT_ID"),
    url=os.getenv("WX_URL"),
)
topic_extractor = LLMTopicExtractor(
    content_provider=ChromaDBProvider(
        collection="sound",
        db_path="/path/to/local/chroma",
    ),
    llm_client=llm_client,
)
topic_extractor_extraction = TopicExtractionMiddleware(
    subject="tool_sound",
    topic_extractor=topic_extractor,
    topics_sink=ChromaDBProvider(db_path="/path/to/local/chroma", collection="topics"),
)
topic_extraction_output: TopicExtractionBuildOutput = (
    topic_extractor_extraction.process(data=None, phase=AgentPhase.BUILDTIME)
)

print(f"{topic_extraction_output.error=}")
print(f"{topic_extraction_output.topics}")
```

If you need to extract topics for other tools used by the agent this process must be repeated creating a Topic Extractor for each tool. The following code asumes the collections containing the documents for each tool are all in the same Chroma DB instance:
```python
collections = {"tool_1": "coll_1", "tool_2": "coll_2"}
topics_sink = ChromaDBProvider(db_path="/path/to/local/chroma", collection="topics")
for tool in collections.keys():
    topic_extractor = LLMTopicExtractor(
        content_provider=ChromaDBProvider(
            collection=collections[tool], db_path="/path/to/local/chroma"
        ),
        llm_client=llm_client,
    )
    topic_extractor_extraction = TopicExtractionMiddleware(
        subject=tool,
        topic_extractor=topic_extractor,
        topics_sink=topics_sink,
    )
    topic_extraction_output: TopicExtractionBuildOutput = (
        topic_extractor_extraction.process(data=None, phase=AgentPhase.BUILDTIME)
    )
    print(f"Topic extraction for tool {tool}")
    print(f"{topic_extraction_output.error=}")
    print(f"{topic_extraction_output.topics}")
```

2. In your agent that uses the `tool_sound` tool, when a query comes in use a Topic Retriever to retrieve the topics of expertise that are similar to the query and use them to create tool hints that can be added to the agent prompt.

```python
from altk.pre_llm.routing.retrieval_augmented_thinking.chroma.topic_retriever import (
    ChromaDBTopicRetriever,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.core.toolkit import (
    TopicRetrievalRunInput,
    TopicRetrievalRunOutput,
)
from altk.pre_llm.routing.retrieval_augmented_thinking.topic_retriever.topic_retriever import (
    TopicRetrievalMiddleware,
)
from altk.core.toolkit import AgentPhase


topic_retriever = TopicRetrievalMiddleware(
    topic_retriever=ChromaDBTopicRetriever(
        db_path="/path/to/local/chroma", collection="topics"
    )
)
topic_retriever_ouput: TopicRetrievalRunOutput = topic_retriever.process(
    data=TopicRetrievalRunInput(
        messages=[
            {"role": "user", "content": "What are the basic characteristics of sound?"}
        ]
    ),
    phase=AgentPhase.RUNTIME,
)
hints = "\n".join(
    {
        retrieved_topic.topic.subject + ": " + retrieved_topic.topic.topic
        for retrieved_topic in topic_retriever_ouput.topics
    }
)
hints_prompt_fragment = f"Here are some hints to use when thinking about which tool might be best to help:\n{hints}"
print(hints_prompt_fragment)
```
The following is a prompt fragment with tool hints using the retrieved topics that can be injected into the agent's prompt:
```
Here are some hints to use when thinking about which tool might be best to help:
tool_sound: hearing a sound
tool_sound: sound perception
tool_sound: sound wave
tool_sound: loud sounds
tool_sound: quiet sounds
tool_sound: Understanding Sound and Hearing
tool_sound: sound waves
tool_sound: sound
tool_sound: absence of sounds
tool_sound: auditory perception
```


## License
Apache 2.0 - see LICENSE file for details.

## Under the Hood
For more details on the architecture, experimental results, and its usage, refer to our [documentation](https://agenttoolkit.github.io/agent-lifecycle-toolkit/concepts/components/rat/).
