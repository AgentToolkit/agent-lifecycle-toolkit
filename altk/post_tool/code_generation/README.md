# Code Generation for Tool Response (JSON) Processing
If the agent calls tools which generate complex JSON objects as responses, this component will use LLM-based Python code generation to process those responses and extract relevant information from them. See up to 20% improvement in accuracy on some queries even when using frontier models like GPT-4o.

## Table of Contents
- [When it is recommended to Use This Component](#when-it-is-recommended-to-use-this-component)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Examples](#examples)
  - [Input Format](#input-format)
  - [Output Format](#output-format)
- [Testing](#testing)
  - [Running Tests](#running-tests)
- [License](#license)
- [Under the Hood](#under-the-hood)



## When it is recommended to Use This Component:

Based on our evaluation, this approach to tool response processing is effective when the JSON tool responses are large and complex/highly nested and when the processing required is not trivial. For example, when the information to be extracted from the response is present as a single key, it might be easier for the model to extract it without code generation. But for more complex processing such as filtering and aggregation, code generation based processing is more accurate.


## Quick Start
Here is how you can call the code generation based tool response processing:

```Python

from altk.toolkit_core.core.toolkit import AgentPhase
from altk.post_tool.code_generation.code_generation import CodeGenerationComponent
from altk.post_tool.core.toolkit import CodeGenerationRunInput, CodeGenerationRunOutput

nl_query = "I need info about X from service Y."
response = {...}  # this is some tool or api response in JSON format
component = CodeGenerationComponent()

input_data = CodeGenerationRunInput(
    messages=[],
    nl_query=nl_query,
    tool_response=response
)
output = component.process(input_data, AgentPhase.RUNTIME)
```


## Configuration

The LLM model, provider and inference settings can be configured when creating the `CodeGenerationComponent` by supplying the relevant parameters. For example:

```python
from altk.post_tool.code_generation.code_generation import CodeGenerationComponent

component = CodeGenerationComponent(
    model_id="meta-llama/llama-3-405b-instruct",
    provider="openai.sync",  # can be one of: openai, watsonx, litellm
    model_kwargs={
        "temperature": 0.5,
        "max_tokens": 1000,
        "min_tokens": 20,
        "seed": 42,
    }
)
```

- `use_docker_sandbox` when set to `True` will use the `DOCKER_HOST` in the environment variables to run the generated code in a container, setting to `False` will use a restricted Python interpreter locally. Running the generated code locally while faster carries some security risks.

## Examples

```python
from altk.toolkit_core.core.toolkit import AgentPhase
from altk.post_tool.code_generation.code_generation import CodeGenerationComponent
from altk.post_tool.core.toolkit import CodeGenerationRunInput, CodeGenerationRunOutput

component = CodeGenerationComponent()

input_data = CodeGenerationRunInput(
    messages=[],
    user_query=user_query,
    tool_responses=response
)
output = component.process(input_data, AgentPhase.RUNTIME)
```

### Input Format
The class post_tool_reflection_toolkit.core.toolkit.CodeGenerationRunInput expects two main inputs as follows:

1. `nl_query`: str, this is the natural language description hinting at what information needs to be extracted from the response. It can be the agent's thought corresponding to the tool call or the user query directly.

2. `tool_response`: Any, this is the JSON response from the tool.

### Output Format
The output is a `CodeGenerationRunOutput` object with a result property that contains the data that was extracted using the generated code.

## Testing
This component includes comprehensive test suites:
### Running Tests
```
# Run all tests
uv run pytest tests/post-tool-reflection/

# Run specific test categories
uv run pytest tests/post-tool-reflection/codegen_test.py
```

## License
Apache 2.0 - see LICENSE file for details.

## Under the Hood
For more details on how the technology behind this component, the architecture and experimental results, refer to our documentation website (Coming soon).
