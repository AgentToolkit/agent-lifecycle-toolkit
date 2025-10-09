# ToolGuards for Enforcing Agentic Policy Adherence
An agent lifecycle solution for enforcement of business policies adherence in agentic workflows. See up to 20 points gain in accuracy in end to end agent performance when enabling this component.

## Table of Contents
- [When to Use This Component](#when-it-is-recommended-to-use-this-component)
- [Quick Start](#quick-start)
- [Parameters](#parameters)
  - [Constructor Parameters](#constructor-parameters)
  - [Run Phase Input Format](#run-phase-input-format)
  - [Run Phase Output Format](#run-phase-output-format)
- [License](#license)
- [Under the Hood](#under-the-hood)

## When it is Recommended to Use This Component:

Policies addressed in this study are those directly protecting tool invocation (pre-tool activation level), hence help preventing altering a system state in a way that contradicts business guidelines.


## Quick Start


### Configuration

```bash
export AZURE_OPENAI_API_KEY=<open ai azure key>
export AZURE_API_BASE="https://eteopenai.azure-api.net"
export AZURE_API_VERSION="2024-08-01-preview"
export PROG_AI_PROVIDER=azure
export PROG_AI_MODEL=gpt-4o-2024-08-06
```
### Example
A simple use case can be found under pre-tool-guard-toolkit/examples/calculator_example

```python
import dotenv

from langchain_core.messages import HumanMessage

from altk.pre_tool_guard_toolkit.core import (
    ToolGuardBuildInput,
    ToolGuardBuildInputMetaData,
    ToolGuardRunInput,
    ToolGuardRunInputMetaData,
)
from altk.pre_tool_guard_toolkit.pre_tool_guard import PreToolGuardComponent

# Load environment variables
dotenv.load_dotenv()


class ToolGuardExample:
    """
    Runs examples with a ToolGuard component and validates tool invocation against policy.
    """

    def __init__(self, model, tools, workdir, policy_doc_path, short=True, tools2run=None):
        self.model = model
        self.tools = tools
        self.toolguard = PreToolGuardComponent(model=model, tools=tools, workdir=workdir)

        build_input = ToolGuardBuildInput(
            metadata=ToolGuardBuildInputMetaData(
                policy_doc_path=policy_doc_path,
                tools2run=tools2run,
                short1=short,
            )
        )
        self.toolguard._build(build_input)

    def run_example(self, user_message: str, tool_name: str, tool_params: dict, should_pass: bool):
        """
        Runs a single example through ToolGuard and checks if the result matches the expectation.
        """
        conversation_context = [HumanMessage(content=user_message)]

        run_input = ToolGuardRunInput(
            messages=conversation_context,
            metadata=ToolGuardRunInputMetaData(
                tool_name=tool_name,
                tool_parms=tool_params
            )
        )

        run_output = self.toolguard._run(run_input)
        print(f"run_output: {run_output}")

        passed = not run_output.output.error_message
        if passed == should_pass:
            print("✅ Success!")
        else:
            print("❌ Failed!")

```

## Parameters

### Constructor Parameters

```python
PreToolGuardComponent(tools, workdir)
```

| Parameter | Type             | Description                                                                                                                                                                       |
| --------- | ---------------- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tools`   | `list[Callable]` | List of tool functions (or `langchain` tools) that the component should manage. Each tool must have a well-defined signature and docstring describing parameters and return type. |
| `workdir` | `str` or `Path`  | Path to a writable working directory where intermediate build and runtime artifacts will be stored (`Step_1/` and `Step_2/` folders). Must exist or be creatable.                 |

###  Build Phase input format

```python
ToolGuardBuildInput(
    metadata=ToolGuardBuildInputMetaData(
        policy_text="<HTML or Markdown policy string>",
        short1=True,
        validating_llm_client=<LLMClient instance>
    )
)
```

| Field                   | Type        | Description                                                                                                   |
| ----------------------- | ----------- |---------------------------------------------------------------------------------------------------------------|
| `policy_text`           | `str`       | The policy text. Can be plain text, Markdown, or HTML. This defines the rules and constraints for tool usage. |
| `short1`                | `bool`      | If `True`, runs a faster, summarized build process. If `False`, runs a full build.                            |
| `validating_llm_client` | `LLMClient` | A Validation LLM client used during build-time validation of the policy.                                      |

### Run phase Input Format
```python
ToolGuardRunInput(
    metadata=ToolGuardRunInputMetaData(
        tool_name="divide_tool",
        tool_parms={"g": 3, "h": 4},
        llm_client=<LLMClient instance>
    ),
    messages=[
        {"role": "user", "content": "Can you please calculate how much is 3/4?"}
    ]
)
```

| Field        | Type         | Description                                                                           |
| ------------ | ------------ | ------------------------------------------------------------------------------------- |
| `tool_name`  | `str`        | The name of the tool being invoked (must match one provided in `tools` during build). |
| `tool_parms` | `dict`       | Parameters for the tool call. Must match the tool’s expected argument names.          |
| `llm_client` | `LLMClient`  | Runtime LLM client used for evaluation.                                               |
| `messages`   | `list[dict]` | Conversation history in `{role, content}` format.                                     |

### Run phase Output Format
```python
ToolGuardRunOutput(
    output=ToolGuardRunOutputMetaData(
        error_message=False  # or string with violation reason
    )
)
```
| Field           | Type            | Description                                                                                          |
| --------------- | --------------- | ---------------------------------------------------------------------------------------------------- |
| `error_message` | `bool` or `str` | `False` if tool call passed validation, or a string with an error message if it violated the policy. |



## License
Apache 2.0 - see LICENSE file for details.

## Under the Hood
For more details on how the technology behind this component, the architecture and experimental results, refer to our documentation website (Coming soon).
