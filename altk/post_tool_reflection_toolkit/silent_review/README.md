# Silent Error Review
A prompt-based approach to **identify** silent errors in tool calls (errors that do not produce any visible or explicit error message); determines whether the tool response is relevant, accurate and complete based on the user's query.

## Table of Contents
- [When it is recommended to Use This Component](#when-it-is-recommended-to-use-this-component)
- [Quick Start](#quick-start)
- [Interface](#interface)
- [License](#license)
- [Under the Hood](#under-the-hood)


## When it is recommended to Use This Component:
Best suited for tool responses that are verbose and/or based on tabular responses.


## Quick Start
The below example should give you an idea of how to plug in this component into your agent pipeline:
```python
from altk.post_tool_reflection_toolkit.silent_review.silent_review import SilentReviewForJSONDataComponent
from altk.post_tool_reflection_toolkit.core.toolkit import SilentReviewRunInput
from altk.toolkit_core.core.toolkit import AgentPhase

input_data = SilentReviewRunInput(
        messages=[
            HumanMessage(content="Tell me the weather"),
            AIMessage(content="Calling the weather tool now")
        ],
        toolspec={
            "name": "get_weather",
            "description": "Gets weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        },
        tool_response={
            "name": "get_weather",
            "result": {"city": "NYC", "temperature": "75F", "condition": "Sunny"}
        }
    )

reviewer = SilentReviewForJSONDataComponent()
result = reviewer.process(data=input_data, phase=AgentPhase.RUNTIME)
print(result.outcome.value)

# possible outcomes

# NOT_ACCOMPLISHED = 0
# PARTIAL_ACCOMPLISH = 0.5
# ACCOMPLISHED = 1

```


## Interface
Expected input:
- user query,
- tool response,
- tool specification,
- tool input,
- tool type

Expected output:
- Accomplished | Partially Accomplished | Not Accomplished


## License
Apache 2.0 - see LICENSE file for details.


## Under the Hood
For more details on how the technology behind this component, the architecture and experimental results, refer to our documentation website (Coming soon).
