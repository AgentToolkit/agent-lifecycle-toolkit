# LLM Tests

This directory contains comprehensive test suites for various LLM providers supported by the agent-lifecycle-toolkit.

## ⚠️ Important: These tests are EXPENSIVE and require opt-in

LLM tests in this directory are marked with `@pytest.mark.llm` and are **excluded by default** when running `pytest` or `pytest tests/`. They only run when explicitly requested due to:

- **API Costs**: These tests make real API calls to LLM providers
- **Network Dependencies**: Require internet access and valid API keys
- **Time**: Can be slow depending on LLM response times

## Running LLM Tests

### Opt-in to LLM Tests
```bash
# Run ONLY LLM tests
pytest -m llm

# Run specific LLM provider tests
pytest tests/llm/test_watsonx_ai.py -m llm
pytest tests/llm/test_azure_openai.py -m llm

# Run LLM tests with verbose output
pytest -m llm -v -s

# Run specific test within LLM suite
pytest tests/llm/test_watsonx_ai.py::TestWatsonXAI::test_simple_completion -m llm
```

### Running Both Regular and LLM Tests
```bash
# Remove the default exclusion to run all tests
pytest -m ""

# Or run specific combinations
pytest -m "not llm" && pytest -m llm
```

## Test Coverage

### 1. IBM WatsonX AI (`test_watsonx_ai.py`)
Tests for the native IBM WatsonX AI provider.

**Required Environment Variables:**
- `WX_API_KEY`: IBM WatsonX API key
- `WX_PROJECT_ID`: IBM WatsonX project ID
- `WX_URL`: IBM WatsonX URL (optional, defaults to https://us-south.ml.cloud.ibm.com)

**Required Package:**
- `ibm_watsonx_ai`

### 2. LiteLLM WatsonX (`test_litellm_watsonx.py`)
Tests for the LiteLLM WatsonX provider.

**Required Environment Variables:**
- `WX_API_KEY`: IBM WatsonX API key
- `WX_PROJECT_ID`: IBM WatsonX project ID
- `WX_URL`: IBM WatsonX URL

**Required Package:**
- `litellm`

### 3. Azure OpenAI (`test_azure_openai.py`)
Tests for the Azure OpenAI provider.

**Required Environment Variables:**
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_API_BASE`: Azure API base URL
- `AZURE_API_VERSION`: Azure API version

**Required Package:**
- `openai`

## Test Types

Each test file includes the following test categories:

### Basic Generation Tests
- Simple text generation
- Chat-style generation
- Generation with custom parameters
- Different generation modes (text vs chat)

### Tool Calling Tests
- Basic tool calling functionality
- Tool calling with specific tools
- Tool calling with auto selection
- Multiple tool scenarios

### Structured Output Tests
- Pydantic model validation
- JSON schema validation
- Complex schema validation
- Retry mechanisms for validation

### Async Tests
- Async generation
- Async structured output
- Async tool calling
- Concurrent request handling

### Integration Tests
- Client initialization
- Error handling
- Hooks functionality
- Multiple model support
- Configuration testing

## Running Tests

### Run All LLM Tests
```bash
pytest tests/llm/
```

### Run Specific Provider Tests
```bash
# WatsonX AI tests
pytest tests/llm/test_watsonx_ai.py

# LiteLLM WatsonX tests
pytest tests/llm/test_litellm_watsonx.py

# Azure OpenAI tests
pytest tests/llm/test_azure_openai.py
```

### Run with Verbose Output
```bash
pytest tests/llm/ -v
```

### Run with Coverage
```bash
pytest tests/llm/ --cov=core.llm
```

## Test Skipping

Tests are automatically skipped in the following scenarios:

1. **Missing Environment Variables**: Tests requiring specific environment variables will be skipped if those variables are not set.

2. **Missing Dependencies**: Tests requiring specific packages (e.g., `openai`, `litellm`, `ibm_watsonx_ai`) will be skipped if those packages are not installed.

3. **Skip Examples**:
   ```python
   # Skip if environment variables not set
   @pytest.mark.skipif(
       not all([os.getenv("WX_API_KEY"), os.getenv("WX_PROJECT_ID")]),
       reason="WX_API_KEY and WX_PROJECT_ID environment variables not set"
   )

   # Skip if package not available
   @pytest.mark.skipif(
       not OPENAI_AVAILABLE,
       reason="openai package not available"
   )
   ```

## Environment Setup

### For WatsonX AI and LiteLLM WatsonX
```bash
export WX_API_KEY="your-watsonx-api-key"
export WX_PROJECT_ID="your-watsonx-project-id"
export WX_URL="https://us-south.ml.cloud.ibm.com"  # Optional
```

### For Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2024-08-01-preview"
```

## Test Data Models

The tests use several Pydantic models for structured output validation:

### Common Models
```python
class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    condition: str
    humidity: Optional[int] = None
```

### Provider-Specific Models
```python
class BusinessInsight(BaseModel):
    insight: str
    confidence: float
    supporting_data: List[str]
    recommendation: str

class TechnicalAnalysis(BaseModel):
    topic: str
    complexity_level: str
    key_points: List[str]
    confidence_score: float
```

## Troubleshooting

### Common Issues

1. **ImportError**: Install missing dependencies
   ```bash
   pip install openai litellm ibm-watsonx-ai
   ```

2. **Environment Variables**: Ensure all required environment variables are set
   ```bash
   # Check if variables are set
   echo $WX_API_KEY
   echo $AZURE_OPENAI_API_KEY
   ```

3. **API Connectivity**: Verify API endpoints and credentials are correct

### Debug Mode
Run tests with additional debugging:
```bash
pytest tests/llm/ -v -s --tb=short
```

## Contributing

When adding new LLM provider tests:

1. Follow the existing test structure
2. Include proper skip conditions for environment variables and dependencies
3. Test all major functionality: basic generation, tool calling, structured output, async
4. Add comprehensive integration tests
5. Update this README with new provider information

## Test Fixtures

Each test file includes reusable fixtures for client initialization:

```python
@pytest.fixture
def provider_client():
    """Initialize provider client"""
    # Client initialization logic
    return client

@pytest.fixture
def provider_structured_client():
    """Initialize provider client with output validation"""
    # Structured client initialization logic
    return structured_client
```

This approach ensures consistent client configuration across all tests while allowing for easy customization when needed.
