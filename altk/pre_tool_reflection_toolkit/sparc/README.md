# Semantic Pre-execution Analysis for Reliable Calls (SPARC)

This component evaluates tool calls before execution, identifying potential issues and suggesting corrections or transformations across multiple validation layers.

## Table of Contents
- [When it is recommended to use this component](#when-it-is-recommended-to-use-this-component)
- [Features](#features)
- [Quick Start](#quick-start)
- [Input Format](#input-format)
- [Configuration](#configuration)
- [Validation Layers](#validation-layers)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [License](#license)
- [Under the Hood](#under-the-hood)

## When it is recommended to use this component:

- **Critical Applications**: Tool execution can change state, databases, or have significant consequences
- **Complex Parameters**: Your tools have many parameters or complex parameter relationships
- **Unit Conversions**: Tools require specific units/formats that can't be validated statically
- **Parameter Validation**: You need to verify that parameters are grounded in conversation context
- **Quality Assurance**: You want to catch hallucinated or incorrect tool calls before execution
- **Production Systems**: Where tool call accuracy is critical for user experience
- **Financial/Medical/Legal**: Domains where incorrect tool calls could have serious consequences

### Configuration Guidelines by Use Case:

- **No parameters or simple parameters**: Use `Track.SYNTAX` for fast static validation only
- **Single-turn or Multi-turn agentic conversations (performance-sensitive)**: Use `Track.FAST_TRACK` for basic semantic validation
- **Single-turn or Multi-turn conversations (high accuracy)**: Use `Track.SLOW_TRACK` for comprehensive validation
- **Unit conversion focus**: Use `Track.TRANSFORMATIONS_ONLY` for transformation-specific validation

## Features

### 🔍 Multi-Layer Validation
- **Static Analysis**: JSON schema validation, type checking, required parameter verification (Python-based, fast, 100% accurate)
- **Semantic Evaluation**: Intent alignment, parameter grounding, function selection appropriateness (LLM-based)
- **Parameter-Level Analysis**: Individual parameter validation and hallucination detection (1 LLM call per metric)
- **Transformation Validation**: Complex value conversion verification using code generation (1 + N LLM calls, where N = parameters needing transformation)

### ⚡ Flexible Execution
- **Sync/Async Support**: Choose execution mode based on your application needs
- **Parallel Execution**: All semantic metrics can run in parallel with ASYNC mode
- **Configurable Metrics**: Select specific validation metrics for your use case
- **Error Handling**: Graceful degradation with detailed error reporting
- **Performance Optimized**: Configurable retry logic and parallel execution limits

### 🛠 Easy Integration
- **Lifecylce Pattern**: Clean integration with existing agentic frameworks
- **Flexible Input**: Supports OpenAI-compatible tool specifications and conversation history
- **Comprehensive Output**: Detailed validation results with explanations and corrections
- **Track-Based API**: Pre-configured validation profiles for different use cases

## Quick Start

```python
from altk.pre_tool_reflection_toolkit.core import (
    SPARCReflectionRunInput,
    Track,
    SPARCExecutionMode,
)
from altk.pre_tool_reflection_toolkit.sparc import SPARCReflectionComponent
from altk.toolkit_core.core.toolkit import AgentPhase, ComponentConfig
from langchain_core.messages import HumanMessage, AIMessage
from altk.toolkit_core.llm import get_llm

# Build ComponentConfig with ValidatingLLMClient (REQUIRED)
# NOTE: This example assumes the OPENAI_API_KEY environment variable is set
def build_config():
    """Build ComponentConfig with OpenAI ValidatingLLMClient."""
    OPENAI_CLIENT = get_llm("openai.sync.output_val")  # ValidatingLLMClient
    # Other validating LLMs: litellm.ollama.output_val, watsonx.output_val
    return ComponentConfig(
        llm_client=OPENAI_CLIENT(
            model_name="o4-mini",
        )
    )

# Initialize reflector with ComponentConfig and Track-based API
config = build_config()
reflector = SPARCReflectionComponent(
    config=config,  # ComponentConfig with ValidatingLLMClient
    track=Track.FAST_TRACK,  # Choose appropriate track
    execution_mode=SPARCExecutionMode.ASYNC,
)

# Check initialization
if reflector._initialization_error:
    print(f"Failed to initialize: {reflector._initialization_error}")
    exit(1)

# Define your tool specification (OpenAI format)
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

# Prepare conversation context
messages = [
    HumanMessage(content="Send an email to team@company.com about the meeting"),
    AIMessage(content="I'll send that email for you.")
]

# Tool call to validate (OpenAI format)
tool_call = {
    "id": "1",
    "type": "function",
    "function": {
        "name": "send_email",
        "arguments": '{"to": ["teams@company.com"], "subject": "Meeting Update", "body": "Meeting scheduled for tomorrow."}'
    }
}

# Run reflection
run_input = SPARCReflectionRunInput(
    messages=messages,
    tool_specs=tool_specs,
    tool_calls=[tool_call]
)

result = reflector.process(run_input, phase=AgentPhase.RUNTIME)

# Check results
if result.output.reflection_result.decision == "approve":
    print("✅ Tool call approved")
else:
    print("❌ Tool call rejected")
    for issue in result.output.reflection_result.issues:
        print(f"  - {issue.metric_name}: {issue.explanation}")
```

## Input Format

The component expects three main inputs in OpenAI-compatible formats:

### 1. Conversational History (messages)
List of messages representing the conversation context:

```python
from langchain_core.messages import HumanMessage, AIMessage

messages = [
    HumanMessage(content="What's the weather in New York?"),
    AIMessage(content="I'll check the weather for you."),
    HumanMessage(content="Make sure to use Fahrenheit please"),
    AIMessage(content="I'll get the weather in New York using Fahrenheit.")
]
```

### 2. Tool Specifications (OpenAI format)
Array of tool specifications following OpenAI function calling format:

```python
tool_specs = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location", "units"]
            }
        }
    }
]
```

### 3. Generated Tool Call (OpenAI format)
The tool call generated by your agent that needs validation:

```python
tool_call = {
    "id": "call_123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "New York, NY", "units": "fahrenheit"}'
    }
}
```

## ValidatingLLMClient Requirements

**IMPORTANT**: The SPARCReflectionComponent requires a `ValidatingLLMClient` instance for proper output validation and structured response handling.

### Supported ValidatingLLMClient Types

```python
from altk.toolkit_core.llm import get_llm

# OpenAI with output validation
OPENAI_CLIENT = get_llm("openai.sync.output_val")

# Ollama (using litellm) with output validation
OLLAMA_CLIENT = get_llm("litellm.ollama.output_val")

# WatsonX (using ibm-watsonx-ai) with output validation
WATSONX_CLIENT = get_llm("watsonx.output_val")

# Azure OpenAI with output validation
AZURE_CLIENT = get_llm("azure_openai.output_val")
```

### Configuration Examples

#### OpenAI ValidatingLLMClient
```python
def build_config():
    OPENAI_CLIENT = get_llm("openai.sync.output_val")
    return ComponentConfig(
        llm_client=OPENAI_CLIENT(
            model_name="o4-mini",
        )
    )
```

### Type Validation

The component automatically validates the LLM client type during initialization:

```python
# ✅ Correct - ValidatingLLMClient
config = build_config()  # Returns ValidatingLLMClient
sparc = SPARCReflectionComponent(config=config, track=Track.SYNTAX)

# ❌ Error - Regular LLMClient (not ValidatingLLMClient)
regular_client = get_llm("openai.sync")  # Returns regular LLMClient
config = ComponentConfig(llm_client=regular_client(...))
sparc = SPARCReflectionComponent(config=config, track=Track.SYNTAX)
# TypeError: LLM client must be of type ValidatingLLMClient
```

## Configuration

### Track-Based Configuration

The component uses a Track enum for simplified configuration. Each track represents a predefined set of validation metrics optimized for specific use cases:

#### `Track.SYNTAX` - Fast Static Validation Only
- **LLM Calls**: 0
- **Validates**: JSON schema, types, required parameters
- **Use Case**: Fast validation, development/testing, simple tools
- **Performance**: Fastest (Python-based validation only)
- **Model Required**: No

```python
# SYNTAX track doesn't require LLM client (static validation only)
sparc = SPARCReflectionComponent(
    config=None,  # No config needed for static-only validation
    track=Track.SYNTAX,
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

#### `Track.FAST_TRACK` - Basic Semantic Validation
- **LLM Calls**: 2
- **Validates**: General hallucination check + function selection appropriateness (executed in parallel)
- **Use Case**: Single-turn or multi-turn conversations, performance-sensitive applications
- **Performance**: Very fast
- **Model Required**: Yes

```python
config = build_config()  # ValidatingLLMClient required
sparc = SPARCReflectionComponent(
    config=config,
    track=Track.FAST_TRACK,
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

#### `Track.SLOW_TRACK` - Comprehensive Validation
- **LLM Calls**: 5 + N (where N = parameters needing transformation, executed in parallel)
- **Validates**: General hallucination check + function selection appropriateness + value format + agentic constraint satisfaction + transformations
- **Use Case**: Single-turn or multi-turn conversations requiring high accuracy
- **Performance**: Moderate (depends on parameter count)
- **Model Required**: Yes

```python
config = build_config()  # ValidatingLLMClient required
sparc = SPARCReflectionComponent(
    config=config,
    track=Track.SLOW_TRACK,
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

#### `Track.TRANSFORMATIONS_ONLY` - Unit/Format Conversion Focus
- **LLM Calls**: 1 + N (where N = parameters needing transformation, executed in parallel)
- **Validates**: Units conversion, format transformations
- **Use Case**: Systems with complex parameter transformation needs
- **Performance**: Variable (depends on transformation complexity)
- **Model Required**: No

```python
# TRANSFORMATIONS_ONLY track doesn't require LLM client for basic transformations
sparc = SPARCReflectionComponent(
    config=None,  # No config needed for transformation-only validation
    track=Track.TRANSFORMATIONS_ONLY,
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

### Track Metrics Breakdown

Each track includes specific validation metrics optimized for different use cases:

#### `Track.SYNTAX`
- **Static validation only** (Python-based, no LLM calls)
- JSON schema compliance
- Type validation
- Required parameter verification
- Constraint validation (min/max, enum values)

#### `Track.FAST_TRACK`
- **METRIC_GENERAL_HALLUCINATION_CHECK**: Detects hallucinated or fabricated parameter values
- **METRIC_FUNCTION_SELECTION_APPROPRIATENESS**: Validates function choice matches user intent

#### `Track.SLOW_TRACK`
- **METRIC_GENERAL_HALLUCINATION_CHECK**: Detects hallucinated or fabricated parameter values
- **METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT**: Validates parameter format requirements
- **METRIC_FUNCTION_SELECTION_APPROPRIATENESS**: Validates function choice matches user intent
- **METRIC_AGENTIC_CONSTRAINTS_SATISFACTION**: Validates adherence to agentic conversation constraints and context
- **Transform enabled**: Unit/format conversions when needed

#### `Track.TRANSFORMATIONS_ONLY`
- **Transform enabled**: Focus on unit/format conversions
- **METRIC_TRANSFORMATION_DETECTION**: Identifies parameters needing transformation
- **Code generation**: Creates transformation code for each parameter requiring conversion
- **Supported conversions**: Temperature (F/C), distance (miles/km), weight (lbs/kg), and custom transformations

### Custom Configuration

For creating custom metric combinations instead of using predefined tracks, you can use the `custom_config` parameter with a `SPARCReflectionConfig` object:

```python
from altk.pre_tool_reflection_toolkit.core import SPARCReflectionConfig, SPARCExecutionMode

# Create custom configuration
custom_config = SPARCReflectionConfig(
    execution_mode=SPARCExecutionMode.ASYNC,  # Enable parallel execution
    general_metrics=["function_intent_alignment", "parameter_value_grounding"],
    function_metrics=["function_selection_appropriateness"],
    parameter_metrics=["parameter_hallucination_check"],
    transform_enabled=True,  # Enable transformation validation
    verbose_logging=True,
    retries=3,
    max_parallel=5  # Control parallel execution limit
)

# LLM client still comes from ComponentConfig
config = build_config()  # ComponentConfig with ValidatingLLMClient
sparc = SPARCReflectionComponent(
    config=config,  # LLM client configuration
    custom_config=custom_config,  # Validation configuration
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

### Custom Metric Configuration

For advanced users who need specific combinations of validation metrics, you can bypass predefined tracks and create custom configurations:

#### Available Metrics

```python
from llmevalkit.function_calling.consts import (
    METRIC_GENERAL_HALLUCINATION_CHECK,        # Detects hallucinated parameter values
    METRIC_GENERAL_VALUE_FORMAT_ALIGNMENT,     # Validates parameter format requirements
    METRIC_FUNCTION_SELECTION_APPROPRIATENESS, # Validates function choice matches intent
    METRIC_AGENTIC_CONSTRAINTS_SATISFACTION,   # Validates agentic conversation constraints
    METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT,   # Validates parameter format requirements
    METRIC_PARAMETER_HALLUCINATION_CHECK,      # Per-parameter hallucination detection
)
```

#### Custom Configuration Examples

##### Example 1: Function Selection Only
```python
from altk.pre_tool_reflection_toolkit.core import SPARCReflectionConfig

# Only validate function selection appropriateness
custom_config = SPARCReflectionConfig(
    general_metrics=None,
    function_metrics=[METRIC_FUNCTION_SELECTION_APPROPRIATENESS],
    parameter_metrics=None,
)

config = build_config()
sparc = SPARCReflectionComponent(
    config=config,
    custom_config=custom_config,  # Use custom config instead of track
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

##### Example 2: Comprehensive Parameter Validation
```python
# Focus on parameter-level validation with hallucination detection
custom_config = SPARCReflectionConfig(
    general_metrics=None,
    function_metrics=None,
    parameter_metrics=[
        METRIC_PARAMETER_HALLUCINATION_CHECK,
        METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT,
    ],
)

sparc = SPARCReflectionComponent(
    config=config,
    custom_config=custom_config,
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

##### Example 3: Minimal Agentic Validation
```python
# Only agentic constraints for multi-turn conversations
custom_config = SPARCReflectionConfig(
    general_metrics=None,
    function_metrics=[METRIC_AGENTIC_CONSTRAINTS_SATISFACTION],
    parameter_metrics=None,
)

sparc = SPARCReflectionComponent(
    config=config,
    custom_config=custom_config,
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

##### Example 4: All Metrics (Maximum Validation)
```python
# Use all available metrics for maximum validation coverage
custom_config = SPARCReflectionConfig(
    general_metrics=[
        METRIC_GENERAL_HALLUCINATION_CHECK,
        METRIC_GENERAL_VALUE_FORMAT_ALIGNMENT,
    ],
    function_metrics=[
        METRIC_FUNCTION_SELECTION_APPROPRIATENESS,
        METRIC_AGENTIC_CONSTRAINTS_SATISFACTION,
    ],
    parameter_metrics=[
        METRIC_PARAMETER_HALLUCINATION_CHECK,
        METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT,
    ],
    transform_enabled=True,  # Enable transformations
)

sparc = SPARCReflectionComponent(
    config=config,
    custom_config=custom_config,
    execution_mode=SPARCExecutionMode.ASYNC,
)
```

#### Metric Categories

- **General Metrics**: Applied to the overall tool call context
  - `METRIC_GENERAL_HALLUCINATION_CHECK`: Detects fabricated or hallucinated information
  - `METRIC_GENERAL_VALUE_FORMAT_ALIGNMENT`: Validates parameter format requirements

- **Function Metrics**: Applied to function selection and appropriateness
  - `METRIC_FUNCTION_SELECTION_APPROPRIATENESS`: Validates function choice matches user intent
  - `METRIC_AGENTIC_CONSTRAINTS_SATISFACTION`: Validates adherence to agentic conversation constraints

- **Parameter Metrics**: Applied to individual parameters (executed per parameter)
  - `METRIC_PARAMETER_HALLUCINATION_CHECK`: Detects hallucinated parameter values
  - `METRIC_PARAMETER_VALUE_FORMAT_ALIGNMENT`: Validates parameter format requirements

#### When to Use Custom Configuration

- **Specific Requirements**: Your use case needs a unique combination of metrics
- **Performance Optimization**: You want to minimize LLM calls for specific metrics
- **Experimental Validation**: Testing different metric combinations
- **Domain-Specific Needs**: Certain metrics are more important for your domain
- **Cost Optimization**: Reducing LLM usage while maintaining necessary validation

## Validation Layers

### 1. Static Validation (Python-based, 100% accurate, fast)

Validates tool call structure without LLM calls:

- **JSON Schema Compliance**: Validates tool call structure against OpenAI function calling format
- **Type Validation**: Ensures parameter types match schema definitions
- **Required Parameters**: Verifies all mandatory parameters are present
- **Constraint Validation**: Checks min/max values, string lengths, enum values
- **Format Validation**: Validates email formats, date formats, etc.

**Example Issues Detected:**
```python
# Missing required parameter
{"to": ["user@example.com"]}  # Missing "subject" and "body"

# Type mismatch
{"participants": "user@example.com"}  # Should be array, not string

# Constraint violation
{"priority": "urgent"}  # Not in enum ["low", "normal", "high"]
```

### 2. Semantic Analysis (LLM-based, configurable metrics)

LLM-powered evaluation of tool call appropriateness. Each metric is an LLM call that can be executed in parallel.

**Example Issues Detected:**
```python
# Intent misalignment
User: "What's the weather?"
Tool Call: book_flight(origin="JFK", destination="LAX")  # Wrong function

# Parameter grounding issue
User: "Call my mom at +1234567890"
Tool Call: send_sms(phone_number="+9876543210")  # Different number

# Parameter hallucination
User: "Check weather in Miami"
Tool Call: get_weather(location="Miami Beach, FL, USA, Downtown District")  # Too specific
```

### 3. Transformation Validation (Code generation-based)

**LLM Calls**: 1 + N (where N = number of parameters requiring transformation)
**Execution**: All transformation checks can run in parallel

Advanced validation using code generation for parameter transformations:

#### Supported Transformations
- **Unit Conversions**: Temperature (F and C), distance (miles and km), weight (lbs and kg)
- **Custom Transformations**: Extensible framework for domain-specific conversions

#### How It Works
1. **Detection**: Identifies potential transformation needs in conversation (1 LLM call)
2. **Code Generation**: LLM generates Python code to perform transformation (per parameter)
3. **Execution**: Safely executes transformation code in sandboxed environment
4. **Validation**: Compares original vs transformed values
5. **Suggestion**: Provides corrected parameter values if transformation needed

**Example Transformations:**
```python
# Temperature conversion
User: "Set thermostat to 75 degrees"  # Assumes Fahrenheit in US context
Tool Spec: {"temperature": {"description": "Temperature in Celsius"}}
Transformation: 75°F → 23.9°C

# Distance conversion
User: "Calculate travel time for 50 miles at 60 mph"
Tool Spec: {"distance_km": ..., "speed_kmh": ...}
Transformation: 50 miles → 80.47 km, 60 mph → 96.56 km/h
```

## Examples

### Static Validation Example

```python
# examples/static_issues_example.py
from altk.pre_tool_reflection_toolkit.core import Track, SPARCExecutionMode
from altk.pre_tool_reflection_toolkit.sparc import SPARCReflectionComponent

def run_static_validation():
    # Initialize with SYNTAX track for fast static validation
    sparc = SPARCReflectionComponent(
        track=Track.SYNTAX,
        execution_mode=SPARCExecutionMode.ASYNC,
    )

    # Example: Missing required parameters
    tool_call = {
        "id": "1",
        "type": "function",
        "function": {
            "name": "send_email",
            "arguments": '{"to": ["user@example.com"]}'
            # Missing required "subject" and "body"
        }
    }

    # Will detect: Missing required parameters "subject", "body"
```

### Semantic Validation Example

```python
# examples/semantic_issues_example.py
from altk.pre_tool_reflection_toolkit.core import Track, SPARCExecutionMode
from altk.pre_tool_reflection_toolkit.sparc import SPARCReflectionComponent
from langchain_core.messages import HumanMessage, AIMessage

def run_semantic_validation():
    # Initialize with FAST_TRACK for agentic conversation validation
    sparc = SPARCReflectionComponent(
        track=Track.FAST_TRACK,
        execution_mode=SPARCExecutionMode.ASYNC,
        model_path="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    )

    # Example: Function selection misalignment
    conversation = [
        HumanMessage("What's the weather in New York?"),
        AIMessage("I'll check the weather for you.")
    ]

    tool_call = {
        "id": "1",
        "type": "function",
        "function": {
            "name": "book_flight",  # Wrong function for weather query
            "arguments": '{"origin": "JFK", "destination": "LAX"}'
        }
    }

    # Will detect: Function intent misalignment
```

### Units Conversion Example

```python
# examples/units_conversion_error_example.py
from altk.pre_tool_reflection_toolkit.core import Track, SPARCExecutionMode
from altk.pre_tool_reflection_toolkit.sparc import SPARCReflectionComponent
from langchain_core.messages import HumanMessage, AIMessage

def run_transformation_validation():
    # Initialize with TRANSFORMATIONS_ONLY track for unit conversion focus
    sparc = SPARCReflectionComponent(
        track=Track.TRANSFORMATIONS_ONLY,
        execution_mode=SPARCExecutionMode.ASYNC,
    )

    # Example: Temperature unit conversion
    conversation = [
        HumanMessage("Set thermostat to 75 degrees Fahrenheit"),
        AIMessage("I'll set the thermostat.")
    ]

    tool_call = {
        "id": "1",
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "arguments": '{"temperature": 75.0, "location": "living room"}'
            # Should be ~24°C, not 75
        }
    }

    # Will detect: Temperature conversion needed (75°F → 23.9°C)
    # Will suggest: {"temperature": 23.9}
```

## API Reference

### Core Types

#### `SPARCReflectionConfig`
Configuration class for the component.

```python
class SPARCReflectionConfig(BaseModel):
    model_path: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    execution_mode: SPARCExecutionMode = SPARCExecutionMode.SYNC
    general_metrics: Optional[List[str]] = None
    function_metrics: Optional[List[str]] = None
    parameter_metrics: Optional[List[str]] = None
    transform_enabled: bool = False
    verbose_logging: bool = False
    retries: int = 3  # For async mode
    max_parallel: int = 7  # For async mode
    # ... additional fields
```

#### `SPARCReflectionResult`
Result of the reflection analysis.

```python
class SPARCReflectionResult(BaseModel):
    decision: SPARCReflectionDecision  # APPROVE, REJECT, ERROR
    issues: List[SPARCReflectionIssue] = []

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0
```

#### `SPARCReflectionIssue`
Individual issue found during validation.

```python
class SPARCReflectionIssue(BaseModel):
    issue_type: SPARCReflectionIssueType  # STATIC, SEMANTIC_*, TRANSFORM, ERROR
    metric_name: str
    explanation: str
    correction: Optional[Dict[str, Any]] = None
```

## Performance Considerations

### Async vs Sync Mode
```python
# Async mode - enables parallel execution of all metrics
config.execution_mode = SPARCExecutionMode.ASYNC
config.max_parallel = 7  # Control parallel execution limit
config.retries = 3  # Retry failed LLM calls

# Sync mode - sequential execution (slower but simpler)
config.execution_mode = SPARCExecutionMode.SYNC
```

## Error Handling

The component provides comprehensive error handling:

### Initialization Errors
```python
sparc = SPARCReflectionComponent(config=config)
if sparc._initialization_error:
    print(f"Initialization failed: {sparc._initialization_error}")
    # Handle initialization failure
```

### Runtime Errors
```python
result = sparc.process(run_input, phase=AgentPhase.RUNTIME)
if result.output.reflection_result.decision == SPARCReflectionDecision.ERROR:
    # Handle error case
    error_issues = [issue for issue in result.output.reflection_result.issues
                   if issue.issue_type == SPARCReflectionIssueType.ERROR]
```

### Graceful Degradation
- Static validation failures can optionally continue to semantic checks (`continue_on_static=True`)
- Individual metric failures don't stop the entire pipeline
- Transformation errors fall back to original values with warnings
- Async mode includes automatic retry logic for failed LLM calls

## Testing

The component includes comprehensive test suites:

### Running Tests
```bash
# Run all tests
uv run pytest tests/pre_tool_reflection_toolkit/

# Run specific test categories
uv run pytest tests/pre_tool_reflection_toolkit/sparc/semantic_validation_test.py
uv run pytest tests/pre_tool_reflection_toolkit/semantic_validation_test.py
uv run pytest tests/pre_tool_reflection_toolkit/units_conversion_test.py
```

### Test Categories
- **Static Validation Tests**: Schema validation, type checking, constraint violations
- **Semantic Validation Tests**: Intent alignment, parameter grounding, hallucination detection
- **Units Conversion Tests**: Temperature, distance, and format transformation validation



## License
Apache 2.0 - see LICENSE file for details.

## Under the Hood
For more details on how the technology behind this component, the architecture and experimental results, refer to our documentation website (Coming soon).
