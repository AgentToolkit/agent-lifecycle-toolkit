# Test Case Generation Toolkit

A toolkit for agentic systems that performs Test Case Generation. This component first generates the test case values and then generates an NL Utterance that formulates the test case values into a user query for validating robustness of tools and agents.

## Table of Contents

  - [When it is Recommended to Use This Component](#when-it-is-recommended-to-use-this-component)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Input Format](#input)
  - [License](#license)

## When it is Recommended to Use This Component
This toolkit generates testcases for the parameters present in the python tool  adhering to data types, data
formats and any internal parameter dependencies. This toolkit is designed to perform robust testing of tools.

## Features
- Generates a test case with all mandatory and all optional parameters.
- Generates a test case with all mandatory parameters.
- Generates remaining test cases with all mandatory and some optional parameters.
- Enables user to provide test case values which is then used by the system to generate testcases in the above order.

## Quick Start

### Configuration
Initialize the Test Case Generation config.

Note: Only TEXT Generation mode is supported (CHAT support coming soon).
```python
from altk.build_time.test_case_generation_toolkit.core.config import TestCaseGenConfig
from altk.core.llm import GenerationMode, get_llm
import os

WatsonXAIClient = get_llm("watsonx")
client = WatsonXAIClient(
    model_id='mistralai/mistral-small-3-1-24b-instruct-2503',
    api_key=os.getenv("WX_API_KEY"),
    project_id=os.getenv("WX_PROJECT_ID"),
    url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com")
)

config = TestCaseGenConfig(
    llm_client=client,
    gen_mode=GenerationMode.TEXT,
    max_nl_utterances=5,
    max_testcases=5,
    clean_nl_utterances=True,  # clean special characters if present in the generated nl utterance
    negative_test_cases=True
)
```

### Examples
1. Automatic test case values and NL utterance generation

```python
from altk.core.toolkit import AgentPhase
from altk.build_time.test_case_generation_toolkit.core.toolkit import TestCaseGenBuildInput
from altk.build_time.test_case_generation_toolkit.utils.test_case_generation import NLTestCaseGenComponent

def test_case_generation_with_toolkit_interface():
    test_case_gen_input = TestCaseGenBuildInput(
    python_tool_str='''import requests
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
    return response.json()'''
    )
    # use config defined above

    test_case_generation_toolkit = NLTestCaseGenComponent()
    result = test_case_generation_toolkit.process(data=test_case_gen_input, config=config, phase=AgentPhase.BUILDTIME)
    print(result.nl_test_cases)
```

2. With external test case values and automatic NL utterance generation

```python
from altk.core.toolkit import AgentPhase
from altk.build_time.test_case_generation_toolkit.core.toolkit import TestCaseGenBuildInput
from altk.build_time.test_case_generation_toolkit.utils.test_case_generation import NLTestCaseGenComponent


def test_case_generation_values_with_toolkit_interface():
    test_case_gen_input = TestCaseGenBuildInput(
    python_tool_str='''from enum import Enum
from typing import ClassVar, Optional, Type
import datetime
from dataclasses import dataclass
from langchain_core.tools import tool

class TimeOffTypes(Enum):
    """Represents the time off event types in SAP SuccessFactors."""

    ABSENCE = "ABSENCE"
    PUBLIC_HOLIDAY = "PUBLIC_HOLIDAY"
    NON_WORKING_DAY = "NON_WORKING_DAY"

@dataclass
class UpcomingTimeOff:
    """Represents an upcoming time off event in SAP SuccessFactors."""

    title: str
    start_date: str
    end_date: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration: int
    time_unit: str
    cross_midnight: bool
    type: str
    status_formatted: Optional[str]
    absence_duration_category: Optional[str]

    #Schema: ClassVar[Type[Schema]] = Schema

@dataclass
class UpcomingTimeOffResponse:
    """Represents the response from getting a user's upcoming time off."""

    time_off_events: list[UpcomingTimeOff]

    #Schema: ClassVar[Type[Schema]] = Schema


@tool
def get_upcoming_time_off(
    user_id: str, start_date: str, end_date: str, time_off_types: list[str]
) -> UpcomingTimeOffResponse:
    """
    Retrieves the user's upcoming time off details from SAP SuccessFactors.

    """

    # Check for date format
    try:
        datetime.datetime.strptime(start_date, "%Y-%m-%d")
        datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}

    time_off_events=[]

    for  time_type in time_off_types:
    # Hard-coded values for testing
        if time_type == "ABSENCE":
            time_off_events.append(
                UpcomingTimeOff(
                    title="Vacation",
                    start_date="2024-01-01",
                    end_date="2024-01-05",
                    start_time=None,
                    end_time=None,
                    duration=5,
                    time_unit="DAYS",
                    cross_midnight=False,
                    type="ABSENCE",
                    status_formatted=None,
                    absence_duration_category=None,
                ) )
        elif time_type == "PUBLIC_HOLIDAY":
            time_off_events.append(
                UpcomingTimeOff(
                    title="New Year's Day",
                    start_date="2024-01-01",
                    end_date="2024-01-01",
                    start_time=None,
                    end_time=None,
                    duration=1,
                    time_unit="DAYS",
                    cross_midnight=False,
                    type="PUBLIC_HOLIDAY",
                    status_formatted=None,
                    absence_duration_category=None,
                ))
            time_off_events.append (UpcomingTimeOff(
                    title="Christmas Day",
                    start_date="2024-12-25",
                    end_date="2024-12-25",
                    start_time=None,
                    end_time=None,
                    duration=1,
                    time_unit="DAYS",
                    cross_midnight=False,
                    type="PUBLIC_HOLIDAY",
                    status_formatted=None,
                    absence_duration_category=None,
                ))

        elif time_type == "NON_WORKING_DAY":
            time_off_events.append(
                UpcomingTimeOff(
                    title="Weekend",
                    start_date="2024-01-06",
                    end_date="2024-01-07",
                    start_time=None,
                    end_time=None,
                    duration=2,
                    time_unit="DAYS",
                    cross_midnight=False,
                    type="NON_WORKING_DAY",
                    status_formatted=None,
                    absence_duration_category=None,
                ))


    if (len(time_off_events)==0):
        return {"status_code": 500, "error": "Invalid values for time off types, valid values are ABSENCE, PUBLIC_HOLIDAY, NON_WORKING_DAY"}
    else:
        return UpcomingTimeOffResponse(time_off_events)''',
        test_case_values=[
            {
                "user_id": ["user123"],
                "start_date": ["2023-10-01"],
                "end_date": ["2023-10-31"],
                "time_off_types": [["ABSENCE", "PUBLIC_HOLIDAY"]],
            }
        ]
    )
    # use config defined above

    test_case_generation_toolkit = NLTestCaseGenComponent()
    result = test_case_generation_toolkit.process(data=test_case_gen_input, config=config,
                                                     phase=AgentPhase.BUILDTIME)
    assert result.nl_test_cases is not None and type(result.nl_test_cases) == dict
    assert "Test_scenarios" in result.nl_test_cases
    assert len(result.nl_test_cases["Test_scenarios"]) > 0 and len(result.nl_test_cases["Test_scenarios"]) <=3
    assert len(result.nl_test_cases["Test_scenarios"][0]["nl_utterance"]) > 0 and len(result.nl_test_cases["Test_scenarios"][0]["nl_utterance"]) <=3
```

## Input
The toolkit expects the Python langchain Tool loaded as string as the input that follows google docstring format.

### Default Configurations

The toolkit provides the following default configuration:

- **Model**: `mistralai/mistral-small-3-1-24b-instruct-2503`
- **Generation Mode**: `TEXT`
- **Maximum number of Test Cases**: `3`
- **Maximum number of NL Utterances**: `3`
- **NL Utterance cleanup required** (clean special characters if present in the generated nl utterance): `False`

## License

MIT License - see LICENSE file for details.
