## SpotLight - Improve LLM's Instruction Following

<p align="center">
<a href="https://arxiv.org/abs/2505.12025" target="_blank">
  <img src="https://img.shields.io/badge/arXiv link-SpotLight-blue" alt="arXiv paper badge" />
</a>
</p>

SpotLight enables users to emphasize important spans within their prompt and steers the LLMs attention towards those spans. Expect to see between 5 and 40 point improvement on accuracy.


## Table of Contents
- [When to Use This Component](#when-to-use-this-component)
- [Quick Start](#quick-start)
- [License](#license)
- [Under the Hood](#under-the-hood)


## When to Use This Component
Use Spotlight when your LLM is failing to follow critical instructions in complex prompts. It is an inference-time hook and does not involve any training or changes to model weights.

> [!IMPORTANT]
> SpotLight only works with locally loaded HuggingFace Transformer models


## Quick Start

1. Initialize the SpotLight config and component objects.
```python
from altk.spotlight_toolkit.core.config import SpotLightConfig
from altk.spotlight_toolkit.spotlight.spotlight import SpotLightComponent

# SpotLightConfig accepts the HF model path and generation arguments
# NOTE: torch may require the PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable
config = SpotLightConfig(model_path="Qwen/Qwen2.5-1.5B-Instruct",
                         generation_kwargs={
                            'max_new_tokens=': 128,
                            'do_sample': False,
                            }
                         )
spotlight = SpotLightComponent(config=config)
```

2. Define the input messages and spans within the prompt to emphasize. Provide these while running SpotLight, along with an optional `alpha` parameter, that defines the amount of emphasis the LLM should place on the span.
> [!NOTE]
> To maintain consistency with the rest of this framework, we use LangChain's message format. SpotLight does support the traditional HF chat format as well.

```python
from langchain_core.messages import HumanMessage, AIMessage
from altk.toolkit_core.core.toolkit import AgentPhase
from altk.spotlight_toolkit.core.config import SpotLightMetadata, SpotLightRunInput

messages = [HumanMessage(content="List the capitals of the following countries - USA, Italy, Greece. Always give me the answer in JSON format.")]
emph_span = ["Always give me the answer in JSON format."]

run_input = SpotLightRunInput(
    messages=messages,
    metadata=SpotLightMetadata(emph_strings=emph_span, alpha=0.1),
)
result = spotlight.process(run_input, phase=AgentPhase.RUNTIME)
prediction = result.output.prediction
print(prediction)
```
> [!NOTE]
> If the emphasized span is not present in the prompt, SpotLight will raise an error

```bash
"""
{
  "capitals": {
    "USA": "Washington D.C.",
    "Italy": "Rome",
    "Greece": "Athens"
  }
}
"""
```

> [!TIP]
> You can emphasize multiple spans by providing them as a list of lists -- `emph_span = [[span_1], [span_2]]`

## License
Apache 2.0 - see LICENSE file for details.

## Under the Hood
For more details on how the technology behind this component, the architecture and experimental results, refer to our documentation website (Coming soon).
