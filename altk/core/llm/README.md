# LLMClient Library

This directory contains a flexible, extensible framework for working with any large-language-model (LLM) provider in a uniform way, including:

- **Unified interface** for multiple LLM providers (OpenAI, Azure OpenAI, IBM WatsonX, LiteLLM)
- **Tool calling support** across all providers with standardized response format
- **Structured output validation** with JSON Schema and Pydantic models
- **Optional dependencies** for each provider to keep installations lean
- **Robust error handling** and retry logic
- **Sync and async support** throughout
- **Observability hooks** for monitoring and debugging

---

## Quick Start

There are three options for getting LLM providers:
1. Instantiating specific LLM providers via `get_llm`
2. Through environment variables and the `auto_from_env` provider
3. Directly passing into `ComponentConfig`

Both examples will use Anthropic as an LLM provider using the claude-sonnet-4 model.

### 1. Using `get_llm` to instantiate an LLM provider

```python
from altk.core.llm import get_llm
from altk.core.toolkit import ComponentConfig

# Get an LLM provider
LiteLLMClient = get_llm("litellm")

client = LiteLLMClient(
    model_name="anthropic/claude-sonnet-4-20250514",
    api_key="*** anthropic api key ***"  # or define the env var ANTHROPIC_API_KEY
)

# Generate text
response = client.generate("Explain quantum computing")
print(response)

# Provide LLMClient to ComponentConfig for usage by component
config = ComponentConfig(llm_client=client)
```

### 2. Using environment variables and the `auto_from_env` provider
By default, ComponentConfig will use this option. Set the following environment variables:
- `ALTK_MODEL_NAME=anthropic/claude-sonnet-4-20250514`
- `ANTHROPIC_API_KEY=*** anthropic api key ***`

```python
from altk.core.llm import get_llm
from altk.core.toolkit import ComponentConfig

# Get an LLM provider
client = get_llm("auto_from_env")()

# Generate text
response = client.generate("Explain quantum computing")
print(response)

# ComponentConfig by default uses this option
config = ComponentConfig()
print(config.llm_client.generate("Explain quantum computing"))
```

By default, the LiteLLM provider is used. Optionally, `ALTK_LLM_PROVIDER` can be set to instantiate a specific provider.
Environment variables needed will differ depending on the desired provider, some examples:

_Ollama_:
```bash
ALTK_LLM_PROVIDER=litellm.ollama
ALTK_MODEL_NAME=granite3.3
```

_WatsonX_:
```bash
ALTK_LLM_PROVIDER=watsonx
ALTK_MODEL_NAME=ibm/granite-3.3-8b-instruct
WX_API_KEY=(watsonx api key)
WX_PROJECT_ID=(watsonx project id)
WX_URL=https://us-south.ml.cloud.ibm.com
```

### 3. Directly passing into `ComponentConfig`
`ComponentConfig` also supports directly passing in a LiteLLM-style model name into the `llm_client` parameter. This automatically creates a LiteLLM provider with the corresponding model name. Set the following environment variables:
- `ANTHROPIC_API_KEY=*** anthropic api key ***`

```python
from altk.core.toolkit import ComponentConfig

# Directly pass the LiteLLM-style model name into ComponentConfig
config = ComponentConfig(llm_client="anthropic/claude-sonnet-4-5-20250929")
response = config.llm_client.generate("Explain quantum computing")
print(response)
```


---

You can list all available registered llm clients with:

```python
from altk.core.llm import list_available_llms

print(f"Available LiteLLM providers: {list_available_llms()}")
```

Example output:

```python
"Available LiteLLM providers: ['auto_from_env', 'litellm', 'litellm.output_val', 'litellm.watsonx', 'litellm.watsonx.output_val', 'litellm.ollama', 'litellm.ollama.output_val', 'openai.sync', 'openai.async', 'openai.sync.output_val', 'openai.async.output_val', 'azure_openai.sync', 'azure_openai.async', 'azure_openai.sync.output_val', 'azure_openai.async.output_val', 'watsonx', 'watsonx.output_val']"
```
---

## Examples

Comprehensive examples for each client are available in the `llm_examples/` directory:

- **`azure_openai.py`** - Azure OpenAI client examples with Azure-specific configurations
- **`litellm_ollama.py`** - LiteLLM Ollama client examples with local models
- **`litellm_watsonx.py`** - LiteLLM WatsonX client examples with Granite models
- **`ibm_watsonx_ai.py`** - IBM WatsonX AI client examples with native IBM SDK

Each example file demonstrates:
- Basic text generation (with and without output validation)
- Tool calling functionality
- Async and sync usage patterns
- Error handling and observability hooks
- Structured output with Pydantic models and JSON schemas

To run an example:
```bash
python toolkit-core/llm_examples/azure_openai_example.py
```

---

## Provider-Agnostic Generation Parameters

All providers support a unified `GenerationArgs` interface for consistent parameter handling across different LLM APIs. This allows you to write provider-agnostic code that works seamlessly with any supported provider.

### GenerationArgs

```python
from altk.core.llm.types import GenerationArgs

# Create generation parameters
gen_args = GenerationArgs(
    min_tokens=10,           # Minimum tokens to generate (supported only by ibm-watsonx-ai)
    max_tokens=100,           # Maximum tokens to generate
    temperature=0.7,          # Sampling temperature (0.0-2.0)
    top_p=0.9,               # Nucleus sampling threshold
    top_k=50,                # Top-k sampling (where supported)
    frequency_penalty=0.1,    # Frequency penalty (-2.0 to 2.0)
    presence_penalty=0.1,     # Presence penalty (-2.0 to 2.0)
    repetition_penalty=1.1,   # Repetition penalty (where supported)
    stop_sequences=["END"],   # Stop sequences
    seed=42,                  # Random seed for reproducibility
    decoding_method="sample", # "greedy" or "sample" token selection
    stream=False,             # Enable streaming
    echo=False,               # Echo input in output (text mode)
    logprobs=True,            # Return log probabilities
    top_logprobs=5,           # Number of top logprobs to return
    timeout=30.0              # Request timeout in seconds
)

# Use with any provider
response = client.generate(
    "Explain quantum computing",
    generation_args=gen_args
)
```

### Parameter Mapping by Provider

Each provider automatically maps `GenerationArgs` to their native parameter formats:

#### **OpenAI/Azure OpenAI**
- **Direct mapping**: Most parameters map 1:1 (e.g., `temperature` → `temperature`)
- **Stop sequences**: `stop_sequences` → `stop`
- **Decoding method**: `decoding_method="greedy"` → `temperature=0.0`
- **Text vs Chat**: Different parameter support (e.g., `echo` only in text mode, `tools` only in chat mode)

#### **IBM WatsonX AI**
- **Parameter wrapping**: All parameters wrapped in `params` dict for IBM ModelInference API
- **Text mode mappings**: `max_tokens` → `max_new_tokens`, `seed` → `random_seed`
- **Chat mode mappings**: `max_tokens` → `max_tokens`, `seed` → `seed`
- **Stop sequences**: `stop_sequences` → `stop_sequences` (text), `stop_sequences` → `stop` (chat)
- **Decoding method**: Native support for `decoding_method` parameter

#### **LiteLLM**
- **Chat-only**: Only supports chat modes (no text completion)
- **Parameter passthrough**: Most parameters passed directly to underlying model
- **Top-k handling**: `top_k` → `{"top_k": value}` for models that support it
- **Repetition penalty**: `repetition_penalty` → `{"repetition_penalty": value}`
- **Decoding method**: `decoding_method="greedy"` → `temperature=0.0`

### Usage Examples

```python
from altk.core.llm import get_llm
from altk.core.llm.types import GenerationArgs

# Same parameters work across all providers
gen_args = GenerationArgs(
    max_tokens=150,
    temperature=0.7,
    top_p=0.9,
    seed=42,
    decoding_method="sample"
)

# IBM WatsonX
watsonx_client = get_llm("watsonx")(
    model_name="meta-llama/llama-3-3-70b-instruct",
    api_key="...",
    project_id="..."
)
watsonx_response = watsonx_client.generate(
    "What is AI?",
    generation_args=gen_args
)

# LiteLLM
litellm_client = get_llm("litellm")(
    model_name="meta-llama/llama-3-3-70b-instruct"
)
litellm_response = litellm_client.generate(
    "What is AI?",
    generation_args=gen_args
)
```

---

## Structured output validation

```python
# Use structured output
WatsonXLiteLLMClientOutputVal = get_llm("openai.sync.output_val")
structured_client = WatsonXLiteLLMClientOutputVal(
    model_name="meta-llama/llama-3-3-70b-instruct",
    include_schema_in_system_prompt=True, # Whether to add the Json Schema to the system prompt, or not
)

from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int

person = structured_client.generate(
    "Extract: John Doe, 30 years old",
    schema=Person,
    max_retries=2
)
print(f"Name: {person.name}, Age: {person.age}")
```

---

## Core Components

### `base.py`

- **`LLMClient`**
  The abstract foundation for any provider.
  - Manages a registry of implementations (`register_llm`, `get_llm`).
  - Handles initialization of the underlying SDK client.
  - Exposes four main methods:
    - `generate` (sync single)
    - `generate_async` (async single)
  - Emits observability hooks around every call (`before_generate`, `after_generate`, `error`).
  - Requires subclasses to register their own `MethodConfig` entries (mapping "chat", "text", etc., to real SDK methods) and to implement a `_parse_llm_response(raw)` method to extract plain-text from the provider's raw response.

### `output_parser.py`

- **`ValidatingLLMClient`**
  An extension of `LLMClient` that adds:
  1. **Output enforcement** against a schema (JSON Schema dict, Pydantic model, or basic Python type).
  2. Automatic **prompt injection** of system-level instructions ("Only output JSON matching this schema").
  3. **Cleaning** of raw responses (stripping Markdown, extracting fenced JSON).
  4. **Retries** for malformed outputs—only the bad items are retried.
  5. Methods mirror `LLMClient` but return fully-parsed Python objects (or Pydantic instances) instead of raw text.

---

## Available Providers

All provider adapters live under `providers/`. They subclass either `LLMClient` (plain) or `ValidatingLLMClient` (with output validation), and register themselves with a name you can pass to `get_llm(...)`.

### OpenAI Adapter
**Path:** `providers/openai/openai.py`
**Registered names:**
- `openai.sync` -> synchronous client
- `openai.async` -> asynchronous client
- `openai.sync.output_val` -> synchronous client with output validation
- `openai.async.output_val` -> asynchronous client with output validation

**Features:**
- Wraps `openai.OpenAI` SDK.
- Supports text & chat, sync & async.
- Tool calling support with structured responses.
- Streaming support.
- **Parameter mapping**: Direct 1:1 mapping for most parameters, with `decoding_method` mapped to temperature control

**Environment:**
Set `OPENAI_API_KEY` in your environment, or pass it to the constructor.

**Example:**
```python
from altk.core.llm import get_llm
from altk.core.llm.types import GenerationArgs

# Basic usage
client = get_llm("openai.sync")(api_key="your-key")
response = client.generate("Hello, world!", model="gpt-4o")

# With GenerationArgs
gen_args = GenerationArgs(
    max_tokens=100,
    temperature=0.7,
    decoding_method="sample"
)
response = client.generate("Hello, world!", generation_args=gen_args, model="gpt-4o")

# With output validation
client = get_llm("openai.sync.output_val")(api_key="your-key")
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int

person = client.generate(
  "Create a person",
  model="gpt-4o",
  schema=Person,
  include_schema_in_system_prompt=True, # Whether to add the Json Schema to the system prompt, or not
)
```

### Azure OpenAI Adapter
**Path:** `providers/openai/openai.py`
**Registered names:**
- `azure_openai.sync` -> synchronous client
- `azure_openai.async` -> asynchronous client
- `azure_openai.sync.output_val` -> synchronous client with output validation
- `azure_openai.async.output_val` -> asynchronous client with output validation

**Features:**
- Wraps `openai.AzureOpenAI` SDK.
- Supports Azure-specific configurations (endpoint, API version, deployment).
- Tool calling support with structured responses.
- Streaming support.
- **Parameter mapping**: Same as OpenAI with direct 1:1 mapping and `decoding_method` support

**Example:**
```python
from altk.core.llm import get_llm
from altk.core.llm.types import GenerationArgs

client = get_llm("azure_openai.sync")(
    api_key="your-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-08-01-preview"
)

# With GenerationArgs
gen_args = GenerationArgs(
    max_tokens=100,
    temperature=0.5,
    decoding_method="sample"
)
response = client.generate("Hello, world!", generation_args=gen_args, model="gpt-4o-2024-08-06")
```

### LiteLLM Adapter
**Path:** `providers/litellm/litellm.py`
**Registered names:**
- `litellm` -> plain text adapter
- `litellm.output_val` -> validating adapter

**Features:**
- Wraps any model served by the `litellm` SDK.
- Supports chat, text APIs, both sync & async.
- The **plain** adapter returns raw strings; the **output-val** adapter enforces JSON schemas, Pydantic models, or basic types with retries.
- Streaming support.

**Parameter Mapping:**
LiteLLM uses custom parameter transforms for some GenerationArgs parameters:
- `top_k` is mapped to LiteLLM using a custom transform function
- `repetition_penalty` is mapped directly
- `decoding_method` is transformed to `temperature` (sample=0.7, greedy=0.0)
- All other standard parameters (max_tokens, temperature, etc.) are mapped directly

**Features:**
- Subclasses the **validating** LiteLLM adapter.
- Automatically sets:
  - `model_name="hosted_vllm/{model_name}"`
  - `guided_decoding_backend=XGRAMMAR`

### Watsonx-Hosted LiteLLM Adapter
**Path:** `providers/litellm/watsonx.py`
**Registered names:**
- `litellm.watsonx` -> plain text adapter
- `litellm.watsonx.output_val` -> validating adapter

**Features:**
- Automatically prefixes `model_name="watsonx/{model_name}"`.
- Inherits all the validation and retry logic from the validating LiteLLM base class.

**Parameter Mapping:**
Watsonx inherits the same parameter mapping as the base LiteLLM adapter.

**Usage with GenerationArgs:**
```python
from altk.core.llm import GenerationArgs, get_llm

generation_args = GenerationArgs(
    max_tokens=200,
    temperature=0.5,
    top_k=30,
    decoding_method="sample"
)

client = get_llm("litellm.watsonx")
response = client.generate("Explain artificial intelligence", generation_args=generation_args, model_name="granite-3.1-8b-instruct")
```

**Environment variables:**
- `WX_API_KEY`
- `WX_PROJECT_ID` or `WX_SPACE_ID`
- `WX_URL`

**Example:**
```python
from altk.core.llm import get_llm

client = get_llm("litellm.watsonx.output_val")(
    model_name="meta-llama/llama-3-3-70b-instruct"
)

class Weather(BaseModel):
    city: str
    temperature_c: float
    condition: str

weather = client.generate(
    "Return weather for Rome with 25C and sunny condition.",
    schema=Weather,
    max_retries=2,
    include_schema_in_system_prompt=True, # Whether to add the Json Schema to the system prompt, or not
)
```

### IBM WatsonX AI Adapter
**Path:** `providers/ibm_watsonx_ai/ibm_watsonx_ai.py`
**Registered names:**
- `watsonx` -> plain text adapter
- `watsonx.output_val` -> validating adapter

**Features:**
- Wraps the native IBM WatsonX AI SDK.
- Advanced generation parameters (temperature, etc.).

**Parameter Mapping:**
IBM WatsonX AI uses its own parameter mapping for GenerationArgs:
- Most parameters (max_tokens, temperature, top_p, etc.) are mapped directly
- `decoding_method` is mapped to WatsonX's native decoding method parameter
- Provider uses the official IBM WatsonX parameter names for optimal compatibility

**Usage with GenerationArgs:**
```python
from altk.core.llm import GenerationArgs, get_llm

generation_args = GenerationArgs(
    max_tokens=150,
    temperature=0.7,
    top_p=0.9,
    decoding_method="sample"
)

client = get_llm("watsonx")
response = client.generate("What is AI?", generation_args=generation_args, model_name="meta-llama/llama-3-3-70b-instruct")
```

**Example:**
```python
from altk.core.llm import get_llm

client = get_llm("watsonx")(
    model_name="meta-llama/llama-3-3-70b-instruct",
)
response = client.generate("Explain quantum computing")
```

---

## Adding Your Own Provider

1. **Subclass** either `LLMClient` (for plain text) or `ValidatingLLMClient` (if you need schema enforcement).
2. **Implement**
   - `@classmethod provider_class() -> your SDK client class`
   - `_register_methods()` to map `"chat"`, `"chat_async"`, (if available `"text"`, `"text_async"`).
   - `_parse_llm_response(raw)` to pull a single string out of the provider's raw response.

3. **Register** your class:
   ```python
   @register_llm("myprovider")
   class MyClient(LLMClient):
       ...
   ```

4. **Use** it via the registry:
   ```python
   from altk.core.llm import get_llm

   Client = get_llm("myprovider")
   client = Client(api_key="…", other_args=…)
   text = client.generate("Hello world")
   ```

---

## Tips & Best Practices

- **Hooks** let you tap into every call for logging, tracing, or metrics:
  ```python
  client = MyClient(..., hooks=[lambda ev, data: print(ev, data)])
  ```
- **Retries** in validating mode help you guard against model hallucinations in structured outputs—set `max_retries=1` or `2` for quick corrections.
- **Keep schemas small**: only require the fields you care about to avoid brittle failures when the model adds extra metadata.
- **Tool calling** works consistently across all providers with the same interface.
- **Environment variables** can be used for API keys and configuration to keep secrets out of code.

---

## Supported Features by Provider

| Feature | OpenAI | Azure OpenAI | LiteLLM | LiteLLM WatsonX | IBM WatsonX AI |
|---------|---------|--------------|---------|--------------|-----------------|
| Basic Generation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Async Generation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool Calling | ✅ | ✅ | ✅ | ✅ | ✅ |
| Structured Output | ✅ | ✅ | ✅ | ✅ | ✅ |
| Observability Hooks | ✅ | ✅ | ✅ | ✅ | ✅ |
| Retry Logic | ✅ | ✅ | ✅ | ✅ | ✅ |
