import pytest
import os


# Global test configuration
@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure the test environment with proper settings."""
    # Set environment variables for testing if not already set
    test_env_vars = {
        "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "PYTEST_CURRENT_TEST": "true",
    }

    for key, value in test_env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    yield

    # Cleanup after tests
    for key in test_env_vars:
        if key in os.environ and os.environ[key] == test_env_vars[key]:
            del os.environ[key]
