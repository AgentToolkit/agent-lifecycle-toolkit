# Refraction - Syntactic Validation of Tool Calls
Refraction is a low-cost (no LLMs!), low-latency, domain-agnostic, data-agnostic, model-agnostic approach towards validation and repair for a sequence of tool calls, based on classical AI planning techniques. We have seen as much as 48% error correction in certain scenarios.

## Table of Contents
- [When it is recommended to use this component](#when-it-is-recommended-to-use-this-component)
- [Quick Start](#quick-start)
- [License](#license)
- [Under the Hood](#under-the-hood)

## When it is recommended to use this component

You can use refraction API to fix individual tool calls and tool call sequences, especially when:
- you have access to catalog or tool specs
- you are validating sequence of tools calls together
- you are validating data flow from memory or between tool calls
- you don't want to use an LLM

## Quick Start

```python
import os
from altk.pre_tool_reflection_toolkit.refraction.refraction import RefractionComponent
from altk.pre_tool_reflection_toolkit.core.types import RefractionBuildInput, RefractionRunInput
from altk.pre_tool_reflection_toolkit.core.config import RefractionConfig, RefractionMode
from altk.toolkit_core.core.toolkit import AgentPhase

config = RefractionConfig(
    mode=RefractionMode.STANDALONE
)

refraction = RefractionComponent(config=config)

tool_specs = [{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to recipients",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        }
    }
}]
tool_call = {
    "id": "1",
    "type": "function",
    "function": {
        "name": "send_email",
        "arguments": {
            "to": ["team@company.com"],
            "subject": "Meeting Update",
            "body": "Meeting scheduled for tomorrow."
        }
    }
}

# build the mappings using the build time phase for faster processing later
build_input = RefractionBuildInput(
    tool_specs=tool_specs
)

memory = {}
output = refraction.process(build_input, phase=AgentPhase.BUILDTIME)

# and/or go straight to processing some input
run_input = RefractionRunInput(
    tool_calls=[tool_call],
    tool_specs=tool_specs,
    mappings=output.mappings, # be sure to use the mappings here if they were built
    memory_objects={},
    use_given_operators_only=True,
)
output = refraction.process(run_input, phase=AgentPhase.RUNTIME)

if output.result.report.determination:
    print("✅ Tool call approved")
else:
    print("❌ Tool call rejected")
    # you can then check if refraction was able to generate a fix for the rejected tool call
    corrected_function_call = output.result.corrected_function_call(memory, output.catalog)
    if corrected_function_call.is_executable:
        print("✅ Tool call repaired")
    else:
        print("❌ Tool call not repaired")
```

## License
Apache 2.0 - see LICENSE file for details.

## Under the Hood
For more details on how the technology behind this component, the architecture and experimental results, refer to our documentation website (Coming soon).
