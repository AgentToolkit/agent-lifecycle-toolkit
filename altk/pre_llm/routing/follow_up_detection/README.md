# Follow up detection

A component that detects if the last message in a conversation between a user and an AI agent is a follow up question, thus the conversation should remain in the current route, or a new topic that requires re-routing.

## Installation

Make sure the dependencies for the Routing components are included by running `pip install "agent-lifecycle-toolkit[routing]"`.

## Why this component is helpful
AI Agents often use routers or manager agents to identify which collaborators or tools are best to handle a given input.  These approaches work well for the first utterance in a conversation, or complete utterances that contain all the information needed to route. However agent interaction is often multi-turn, with multiple back-and-forth questions and answers from the user and AI Agent. The user may say things like "why", "shorter", or "convert it to metric". These scenarios require examining the conversation history to decide which route should be taken.

LLM-based manager nodes often add the conversation history in the context when routing, so it is considered by the LLM when routing; however, this approach often produces unsatisfying results. Separating out the follow-up question logic into its own decision process has several distinct advantages:

  1. LLMs perform better when asked to make one decision at a time, rather than compound requests with complex instructions
  2. A standalone follow-up prompt is easier to customize to best fit your application scenario
  3. Our standalone follow-up detector allows leveraging deterministic rules together with LLM inference to a decision.  For example, if your route has some kind of return code that says it is in the middle of a flow, the follow-up detector may want to check this bit and automatically route back to the same route.

## How to use it
The component supports different message formats by means of a `FollowUpDetector`.
Currently the component includes the following follow up detectors:
* A [default](#default-follow-up-detection) follow up detector that uses the standard message format supported by the toolkit, i.e. a sequence of LangChain's `AIMessage` or `HumanMessage`. This detector uses an LLM to determine if the last message in a conversation is a follow up.
* A [follow up detector for WatsonX Orchestrate](#watsonx-orchestrate-follow-up-detection) message format.

The component [can be extended](#how-to-extend-it) to support other message formats.

The following inputs are required:
* An instance of an [LLMClient](../toolkit-core/toolkit_core/llm/README.md) needed to access an LLM to do the follow up detection
* The list of messages with the conversation history between the user and the AI assitant
* The user query
An optional callback function can be provided that will be called by the component before calling the follow up detector. The callback function receives the conversation history and the user query and returns a boolean flag indicating the result of the follow up detection. If no follow up was detected by the callback then the component calls the follow up detector

The output contains the following fields
* A flag field `is_follow_up` indicating if the user query is a follow up
* A metadata field `detection_type` indicating the type of follow up detection used by which the follow up was detected. This value depends on the specific follow up detector but a common used detection type that all detectors might use is `llm`, indicating that the follow up was detected using an `llm`.

### Default follow up detection
In its default configuration a `DefaultFollowUpDetector` is used. The message type it handles is defined by the type alias `DefaultMessages` in [`follow-up-detection-toolkit/follow_up_detection_toolkit/follow_up.py`](./follow_up_detection_toolkit/follow_up.py):
```
DefaultMessages = Annotated[list[BaseMessage], Field(min_length=1)]
```

```python
import os
from altk.core.llm.base import get_llm
from altk.pre_llm.routing.follow_up_detection.core.toolkit import (
    FollowUpDetectionRunInput,
    FollowUpDetectionRunOutput,
)
from altk.pre_llm.routing.follow_up_detection.follow_up import FollowUpDetectionComponent
from langchain_core.messages import AIMessage, HumanMessage
from altk.core.toolkit import AgentPhase, ComponentConfig

WatsonXAIClient = get_llm("watsonx")
llm_client = WatsonXAIClient(
    model_id="meta-llama/llama-3-3-70b-instruct",
    api_key=os.getenv("WX_API_KEY"),
    project_id=os.getenv("WX_PROJECT_ID"),
    url=os.getenv("WX_URL"),
    hooks=[lambda ev, data: print(ev, data)],
)
follow_up_detection = FollowUpDetectionComponent(
    config=ComponentConfig(llm_client=llm_client)
)
follow_up_detection_result: FollowUpDetectionRunOutput = (
    follow_up_detection.process(
        data=FollowUpDetectionRunInput[DefaultMessages](
            conversation_history=[
                HumanMessage(content="I need the total sales for one year?"),
                AIMessage(content="For which year?"),
            ],
            user_query="2021",
        ),
        phase=AgentPhase.RUNTIME,
    )
)
assert follow_up_detection_result.is_follow_up
assert follow_up_detection_result.metadata.get("detection_type") == "llm"
```

In the following example a callback function to detect the follow up is provided:
```python
import os
from altk.core.llm.base import get_llm
from altk.pre_llm.routing.follow_up_detection.core.middleware import (
    FollowUpDetectionRunInput,
    FollowUpDetectionRunOutput,
)
from altk.pre_llm.routing.follow_up_detection.follow_up import FollowUpDetectionComponent
from langchain_core.messages import AIMessage, HumanMessage
from altk.core.toolkit import AgentPhase, ComponentConfig

WatsonXAIClient = get_llm("watsonx")
llm_client = WatsonXAIClient(
    model_id="meta-llama/llama-3-3-70b-instruct",
    api_key=os.getenv("WX_API_KEY"),
    project_id=os.getenv("WX_PROJECT_ID"),
    url=os.getenv("WX_URL"),
    hooks=[lambda ev, data: print(ev, data)],
)

follow_up_detection = FollowUpDetectionComponent(
    config=ComponentConfig(llm_client=llm_client)
)
follow_up_detection_result: FollowUpDetectionRunOutput = (
    follow_up_detection.process(
        data=FollowUpDetectionRunInput[DefaultMessages](
            conversation_history=[
                HumanMessage(content="I need the total sales for one year?"),
                AIMessage(content="For which year?"),
            ],
            user_query="2021",
            detect_follow_up=lambda messages, user_query: user_query.isdigit()
            and user_query == "2021",
        ),
        phase=AgentPhase.RUNTIME,
    )
)
# The follow up was detected by calling the detect_follow_up callback provided by the caller
assert follow_up_detection_result.is_follow_up
assert (
    follow_up_detection_result.metadata.get("detection_type")
    == "detect_follow_up_callback"
)

```

### WatsonX orchestrate follow up detection
The component includes a follow up detector that supports the WatsonX Orchestrate message format which is a dict like in the following example:
```python
{
    "role": "user",
    "content": [
        {
            "response_type": "text",
            "text": "what is ibm water policy",
        }
    ],
}
```
There is also a message type to represent options given by the assistant to the user, like in this example of an option message containing 4 options:
```python
{
    "role": "assistant",
    "content": [
        {
            "response_type": "option",
            "options": [
                {
                    "label": "ISC Opportunity ID",
                    "value": {
                        "input": {"text": "ISC Opportunity ID"}
                    },
                },
                {
                    "label": "ISC Account ID",
                    "value": {
                        "input": {"text": "ISC Account ID"}
                    },
                },
                {
                    "label": "Domestic Buying Group ID",
                    "value": {
                        "input": {
                            "text": "Domestic Buying Group ID"
                        }
                    },
                },
                {
                    "label": "Domestic Client ID",
                    "value": {
                        "input": {"text": "Domestic Client ID"}
                    },
                },
                {
                    "label": "Not related to an opportunity",
                    "value": {
                        "input": {
                            "text": "Not related to an opportunity"
                        }
                    },
                },
            ],
            "repeat_on_reprompt": True,
        }
    ]
}
```
The type of messages supported by the follow up detector is specified in `follow-up-detection-toolkit/follow_up_detection_toolkit/wx_orchestrate.py` with the type alias `WxOrchestrateMessages` that is defined as follows:
```
WxOrchestrateMessages = Annotated[
    list[dict[str, Any]],
    Field(min_length=1),
    AfterValidator(validate_conversation_history),
]
```
`validate_conversation_history` validates that the list of messages comply with the basic structure of an wxOrchestrate message, like a message must have a `role` and `content` fields.

The follow up detection for WatsonX Orchestrate messages consists of two steps:
1. If the last user message is a pick of one of the options sent previously by the assistant then the last user message is a follow up
1. If the first step didn't detect a follow up then the list of messages that represent the conversation and the current user utterance are sent to an LLM with a prompt that instruct it to detect a follow up.

Here's an example:
```python
from altk.pre_llm.routing.follow_up_detection.core.middleware import FollowUpDetectionRunOutput, FollowUpDetectionRunInput
from altk.pre_llm.routing.follow_up_detection.follow_up import FollowUpDetectionComponent
from altk.pre_llm.routing.follow_up_detection.wx_orchestrate import WxOrchestrateFollowUpDetector, WxOrchestrateMessages
from altk.core.toolkit import AgentPhase, ComponentConfig

WatsonXAIClient = get_llm("watsonx")
llm_client = WatsonXAIClient(
    model_id="meta-llama/llama-3-3-70b-instruct",
    api_key=os.getenv("WX_API_KEY"),
    project_id=os.getenv("WX_PROJECT_ID"),
    url=os.getenv("WX_URL"),
    hooks=[lambda ev, data: print(ev, data)],
)
follow_up_detection = FollowUpDetectionComponent(
    config=ComponentConfig(llm_client=llm_client),
    follow_up_detector=WxOrchestrateFollowUpDetector(),
)
follow_up_detection_result: FollowUpDetectionRunOutput = (
    follow_up_detection.process(
        data=FollowUpDetectionRunInput[WxOrchestrateMessages](
            conversation_history=[
                {
                    "role": "user",
                    "content": [
                        {
                            "response_type": "text",
                            "text": "what is ibm water policy",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "response_type": "text",
                            "text": "Thanks!\xa0Before I answer, it would be helpful for me if you could share some of the below information about the deal.",
                        },
                        {
                            "response_type": "text",
                            "text": "Which of the below can you share?",
                            "repeat_on_reprompt": False,
                        },
                        {
                            "response_type": "option",
                            "options": [
                                {
                                    "label": "ISC Opportunity ID",
                                    "value": {
                                        "input": {"text": "ISC Opportunity ID"}
                                    },
                                },
                                {
                                    "label": "ISC Account ID",
                                    "value": {"input": {"text": "ISC Account ID"}},
                                },
                                {
                                    "label": "Domestic Buying Group ID",
                                    "value": {
                                        "input": {
                                            "text": "Domestic Buying Group ID"
                                        }
                                    },
                                },
                                {
                                    "label": "Domestic Client ID",
                                    "value": {
                                        "input": {"text": "Domestic Client ID"}
                                    },
                                },
                                {
                                    "label": "Not related to an opportunity",
                                    "value": {
                                        "input": {
                                            "text": "Not related to an opportunity"
                                        }
                                    },
                                },
                            ],
                            "repeat_on_reprompt": True,
                        },
                    ],
                },
            ],
            user_query="Domestic Client ID",
        ),
        phase=AgentPhase.RUNTIME,
    )
)
assert follow_up_detection_result.is_follow_up
assert follow_up_detection_result.metadata.get("detection_type") == "choosen_option"
```
The metadata field `detection_type` can have the following values:
* `choosen_option` if the follow up was detected by the presence of an option choosen by the user
* `llm` if the follow up was detected using an LLM

## How to extend it
Define a class implementing the Protocol `FollowUpDetector` which means that the class must implement the following method:
```python
def detect_follow_up(self, config: ComponentConfig, data: FollowUpDetectionRunInput) -> FollowUpDetectionRunOutput
```
The type `FollowUpDetectionRunInput` can be parameterized with the actual type of the messages that are handled by the follow up detector implementation.
For example, the WatsonX orchestrate follow up detector uses WatsonX orchestrate messages that are dicts with `role` and `content` keys and it defines the type alias `WxOrchestrateMessages` that includes Pydating validations to validate that the messages comply with the basic wxOrchestrate message format:
```python
def detect_follow_up( self, config: ComponentConfig, data: FollowUpDetectionRunInput[WxOrchestrateMessages]) -> FollowUpDetectionRunOutput
```

You can then use the class when instantiating the follow up detection middleware component:
```python
follow_up_detection = FollowUpDetectionComponent(config=ComponentConfig(llm_client=llm_client), follow_up_detector=MyFollowUpDetectorClass())
follow_up_detection_result: FollowUpDetectionRunOutput = follow_up_detection.process(data=FollowUpDetectionRunInput[WxOrchestrateMessages](...), phase=AgentPhase.RUNTIME)
```

## How to run the tests
```
pytest -s tests/follow-up-detection
```
