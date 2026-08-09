# Tool Enrichment Component

This component performs python tool enrichment using the metadata information in the tool definition for improved tool calling and tool input formation.

## Table of Contents
  - [When it is Recommended to Use This Component](#when-it-is-recommended-to-use-this-component)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Input](#input)
  - [License](#license)

## When it is Recommended to Use This Component
This component enriches python tool docstrings using the metadata information in the tool definition. This toolkit is designed to enhance the tool docstrings for improved tool learning - i.e., tool selection and tool calling by the LLM.
## Features
This component accepts as input the tool definition python file as string (consisting of @tool decorator) and outputs the enriched python tool string with modified docstring which consists of the following enrichments:
- Enriched tool description using metadata like tool signature, tool method body, current tool description, etc
- Improved tool parameter descriptions using metadata like existing parameter descriptions, declarations of global variables and class definitions and other metadata.
- New examples corresponding to each parameter of the tool based on parameter descriptions, tool body details, etc.
- Enriched tool return descriptions using existing return descriptions, tool body, global variable declarations and other metadata in the tool file.

## Quick Start

### Configuration
Initialize the Python Tool Enrichment config.

Note: Only TEXT Generation mode is supported (CHAT support coming soon).
```python
from altk.build_time.tool_enrichment_toolkit.core.config import PythonToolEnrichConfig
from altk.core.llm import get_llm, GenerationMode
import os

def get_llm_client_obj(model_id='mistralai/mistral-small-3-1-24b-instruct-2503'):
    WatsonXAIClient = get_llm("watsonx")
    client=WatsonXAIClient(
                model_id=model_id,
                api_key=os.getenv("WX_API_KEY"),
                project_id=os.getenv("WX_PROJECT_ID"),
                url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com")
            )
    return client

config = PythonToolEnrichConfig(llm_client=get_llm_client_obj(model_id='mistralai/mistral-small-3-1-24b-instruct-2503'),
                              gen_mode=GenerationMode.TEXT,
                              enable_tool_description_enrichment=True,
                              enable_tool_parameter_description_enrichment=True,
                              enable_tool_return_description_enrichment=True,
                              enable_tool_example_enrichment=True
                              )
```

### Examples
```python
from altk.core.toolkit import AgentPhase
from altk.build_time.tool_enrichment_toolkit.core.toolkit import PythonToolEnrichBuildInput
from altk.build_time.tool_enrichment_toolkit.utils.tool_enrichment import PythonToolEnrichComponent

def test_tool_enrichment_with_new_middleware_interface():
    tool_enrich_input = PythonToolEnrichBuildInput(
        python_tool='''import requests
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from model_utils import load_github_token

@tool
def listIssues(owner: str, repo: str, requestBody: Optional[Dict[str, Any]] = None):
    """
    Get all the relevant issues.

    """
    token = load_github_token()
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.get(url, headers=headers, params=requestBody or {})
    response.raise_for_status()
    return response.json())''')
    # use config defined above
    tool_enrich_middleware = PythonToolEnrichComponent()
    result =  tool_enrich_middleware.process(data=tool_enrich_input, config=config,
                                                     phase=AgentPhase.BUILDTIME)
    assert result.enriched_python_tool is not None and type(result.enriched_python_tool) == str
```


## Input
The toolkit expects the Python langchain Tool loaded as string as the input that follows google docstring format.

### Default Configurations

The component provides a pre-configured generation profile:

#### `standard`
- **Model**: `mistralai/mistral-small-3-1-24b-instruct-2503`
- **Generation Mode**: `TEXT`
- **enable_tool_description_enrichment**: `True`
- **enable_tool_parameter_description_enrichment**: `True`
- **enable_tool_return_description_enrichment**: `True`
- **enable_tool_example_enrichment**: `True`

## License

MIT License - see LICENSE file for details.
