from altk.core.toolkit import AgentPhase
from altk.build_time.test_case_generation_toolkit.core.toolkit import (
    TestCaseGenBuildInput,
)
from altk.build_time.test_case_generation_toolkit.utils.test_case_generation import (
    NLTestCaseGenComponent,
)
from altk.build_time.test_case_generation_toolkit.core.config import TestCaseGenConfig
from altk.core.llm import get_llm, GenerationMode
import os


def get_llm_client_obj(model_name="mistralai/mistral-small-3-1-24b-instruct-2503"):
    WatsonXAIClient = get_llm("watsonx")
    client = WatsonXAIClient(
        model_name=model_name,
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
    )
    return client


def example_testcase_generation_with_toolkit(python_tool_str, values, config):
    test_case_gen_input = TestCaseGenBuildInput(
        python_tool_str=python_tool_str, test_case_values=values
    )
    test_case_generation_toolkit = NLTestCaseGenComponent()
    result = test_case_generation_toolkit.process(
        data=test_case_gen_input, config=config, phase=AgentPhase.BUILDTIME
    )
    return result


python_tool_str = '''
from enum import Enum
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
        return UpcomingTimeOffResponse(time_off_events)'''

values = [
    {
        "user_id": ["user123"],
        "start_date": ["2023-10-01"],
        "end_date": ["2023-10-31"],
        "time_off_types": [["ABSENCE", "PUBLIC_HOLIDAY"]],
    }
]

config = TestCaseGenConfig(
    llm_client=get_llm_client_obj(
        model_name="mistralai/mistral-small-3-1-24b-instruct-2503"
    ),
    gen_mode=GenerationMode.TEXT,
    max_nl_utterances=3,
    max_testcases=3,
    clean_nl_utterances=True,
    negative_test_cases=True,
)

result = example_testcase_generation_with_toolkit(python_tool_str, values, config)
print(result.nl_test_cases)
