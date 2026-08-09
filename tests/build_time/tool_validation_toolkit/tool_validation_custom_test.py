import sys
from altk.core.toolkit import AgentPhase
import os
from altk.build_time.tool_validation_toolkit.core.toolkit import ToolValidationInput
from altk.build_time.tool_validation_toolkit.core.config import ToolValidationConfig
from altk.build_time.tool_validation_toolkit.utils.tool_validation import (
    ToolValidationComponent,
    PythonToolValidationComponent,
)

# creating react agent with python tool
import importlib.util


def get_python_tool(python_tool_string, python_tool_name):
    spec = importlib.util.spec_from_loader("tool_py", loader=None)
    tool_py = importlib.util.module_from_spec(spec)
    exec(python_tool_string, tool_py.__dict__)
    return tool_py.__getattribute__(python_tool_name)


# adding react agent llm code for WATSONX
from ibm_watsonx_ai import Credentials as wx_credentials
from langchain_ibm import ChatWatsonx
from langgraph.prebuilt import create_react_agent


def get_agent_llm(agent_llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503"):
    WATSONX_URL = os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com")
    WATSONX_API_KEY = os.getenv("WX_API_KEY", "")
    WATSONX_PROJECT = os.getenv("WX_PROJECT_ID", "")

    # set "WATSONX_API_KEY" env variable as required by ChatWatsonx Model
    os.environ["WATSONX_APIKEY"] = WATSONX_API_KEY
    # also set "WATSONX_URL" env variable for ChatWatsonx
    os.environ["WATSONX_URL"] = WATSONX_URL
    credentials = wx_credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    project_id = WATSONX_PROJECT
    try:
        llm_parameters = {
            "decoding_method": "greedy",
            "max_new_tokens": 800,
            "min_new_tokens": 1,
        }
        wx_chat_llm = ChatWatsonx(
            model_id=agent_llm_model_id,
            url=WATSONX_URL,
            project_id=project_id,
            credentials=credentials,
            params=llm_parameters,
        )
        return wx_chat_llm
    except ValueError as e:
        print(
            "Please check if all WatsonX related environment varaibles - WX_URL , WX_API_KEY , WX_PROJECT_ID are set"
        )
        print("Error in react agent llm configuration - ", e)


def get_react_agent(
    python_tool_string,
    python_tool_name,
    agent_llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
):
    tool = get_python_tool(python_tool_string, python_tool_name)
    tools = [tool]
    agent_llm = get_agent_llm(agent_llm_model_id)
    react_agent_with_tools = create_react_agent(agent_llm, tools)
    print("Created react agent with tools")
    return react_agent_with_tools


python_tool_name = "getApiV2Tickets"
python_tool_string = """
import json
import requests
from typing import *
from langchain_core.tools import tool
from pydantic import BaseModel


class getApiV2TicketsInput(BaseModel):
	email: Optional[str] = None
	filter: Optional[str] = None
	include: Optional[str] = None
	order_type: Optional[str] = None
	page: Optional[int] = None
	per_page: Optional[int] = None
	requester_id: Optional[str] = None
	type: Optional[str] = None
	updated_since: Optional[str] = None
	workspace_id: Optional[int] = None


@tool
def getApiV2Tickets(getApiV2TicketsInput):
	'''List all tickets in Freshservice
	'''


	header = {
    	'accept': 'application/json',
    	'content-type': 'application/x-www-form-urlencoded'

    	}


	tool_input = getApiV2TicketsInput
	#print(tool_input)
	queryParam = {
	'email': tool_input.get('email', None),
	'filter': tool_input.get('filter', None),
	'include': tool_input.get('include', None),
	'order_type': tool_input.get('order_type', None),
	'page': tool_input.get('page', None),
	'per_page': tool_input.get('per_page', None),
	'requester_id': tool_input.get('requester_id', None),
	'type': tool_input.get('type', None),
	'updated_since': tool_input.get('updated_since', None),
	'workspace_id': tool_input.get('workspace_id', None)
	}

	api_url = "https://ibm.freshservice.com/api/v2/tickets"
	response = requests.get(api_url, headers=header, data=queryParam)

	return response.json()
"""
tool_test_cases = [
    {
        "id": "TC_1",
        "mandatory_params": [],
        "input": "",
        "nl_utterance": ["Show me all tickets in Freshservice."],
    },
    {
        "id": "TC_2",
        "mandatory_params": [],
        "input": "email((string)) := test@example.com \nfilter((string)) := new \ninclude((string)) := requester \norder_type((string)) := asc \npage((integer)) := 1 \nper_page((integer)) := 10 \nrequester_id((string)) := 12345 \ntype((string)) := incident \nupdated_since((string)) := 2023-01-01T00:00:00Z \nworkspace_id((integer)) := 67890 \n",
        "nl_utterance": [
            "Show me the new incident tickets in Freshservice with requester id 12345, email test@example.com, updated since 2023-01-01T00:00:00Z, and workspace id 67890. I want to see the requester details and display 10 tickets per page, sorted in ascending order. Start from page 1."
        ],
    },
    {
        "id": "TC_3",
        "mandatory_params": [],
        "input": "email((string)) := user@example.com \nfilter((string)) := open \ninclude((string)) := stats \norder_type((string)) := desc \npage((integer)) := 2 \nper_page((integer)) := 20 \nrequester_id((string)) := 67890 \ntype((string)) := service_request \nupdated_since((string)) := 2023-02-01T00:00:00Z \nworkspace_id((integer)) := 12345 \n",
        "nl_utterance": [
            "Show me all service requests in Freshservice where the requester's email is user@example.com, the status is open, and the updated time is after 2023-02-01T00:00:00Z. I would like to see the stats and order the results by descending order. Please display 20 results per page, starting from page 2, and only show me the requests with requester\\_id 67890 and workspace\\_id 12345."
        ],
    },
    {
        "id": "TC_4",
        "mandatory_params": [],
        "input": "email((string)) := admin@example.com \nfilter((string)) := closed \ninclude((string)) := requester,stats \norder_type((string)) := asc \npage((integer)) := 3 \nper_page((integer)) := 30 \nrequester_id((string)) := 24680 \ntype((string)) := problem \nupdated_since((string)) := 2023-03-01T00:00:00Z \nworkspace_id((integer)) := 98765 \n",
        "nl_utterance": [
            "Can you list all problem tickets in the workspace with id 98765, where the requester id is 24680, the email is admin@example.com, the filter is closed, the order type is ascending, the page is 3, the number of tickets per page is 30, and the tickets were updated since 2023-03-01T00:00:00Z? Additionally, can you include the requester, stats, and type information in the results?"
        ],
    },
    {
        "id": "TC_5",
        "mandatory_params": [],
        "input": "email((string)) := support@example.com \nfilter((string)) := pending \ninclude((string)) := requester \norder_type((string)) := desc \npage((integer)) := 4 \nper_page((integer)) := 40 \nrequester_id((string)) := 36925 \ntype((string)) := change \nupdated_since((string)) := 2023-04-01T00:00:00Z \nworkspace_id((integer)) := 45678 \n",
        "nl_utterance": [
            "Can you list all tickets in Freshservice where the email is support@example.com, the filter is pending, the include is requester, the order type is descending, the page is 4, the number of tickets per page is 40, the requester ID is 36925, the type is change, the tickets updated since is 2023-04-01T00:00:00Z, and the workspace ID is 45678?"
        ],
    },
    {
        "id": "TC_6",
        "mandatory_params": [],
        "input": "email((string)) := help@example.com \nfilter((string)) := resolved \ninclude((string)) := stats \norder_type((string)) := asc \npage((integer)) := 5 \nper_page((integer)) := 50 \nrequester_id((string)) := 85274 \ntype((string)) := release \nupdated_since((string)) := 2023-05-01T00:00:00Z \nworkspace_id((integer)) := 32109 \n",
        "nl_utterance": [
            "Can you list all tickets in Freshservice where the email is help@example.com, the filter is resolved, the include is stats, the order\\_type is asc, the page is 5, the per\\_page is 50, the requester\\_id is 85274, the type is release, the updated\\_since is 2023-05-01T00:00:00Z, and the workspace\\_id is 32109?"
        ],
    },
    {
        "id": "TC_7",
        "mandatory_params": [],
        "input": "email((string)) := info@example.com \nfilter((string)) := all \ninclude((string)) := requester,stats \norder_type((string)) := desc \npage((integer)) := 6 \nper_page((integer)) := 60 \nrequester_id((string)) := 14725 \ntype((string)) := task \nupdated_since((string)) := 2023-06-01T00:00:00Z \nworkspace_id((integer)) := 85274 \n",
        "nl_utterance": [
            'Can you list all tickets in Freshservice where the email is "info@example.com", filter is set to "all", requester details are included, and the order type is set to "desc"? Additionally, I would like to see tickets updated since "2023-06-01T00:00:00Z", and I am interested in tasks only. I would like to see 60 tickets per page, starting from page 6, and the requester ID should be 14725. Finally, the workspace ID should be 85274.'
        ],
    },
    {
        "id": "TC_8",
        "mandatory_params": [],
        "input": "email((string)) := contact@example.com \nfilter((string)) := spam \ninclude((string)) := requester \norder_type((string)) := asc \npage((integer)) := 7 \nper_page((integer)) := 70 \nrequester_id((string)) := 96325 \ntype((string)) := incident \nupdated_since((string)) := 2023-07-01T00:00:00Z \nworkspace_id((integer)) := 74125 \n",
        "nl_utterance": [
            "Can you list all the tickets in Freshservice where the email is contact@example.com, the filter is spam, the requester is included, the order type is ascending, the page is 7, the number of tickets per page is 70, the requester ID is 96325, the type is incident, the tickets have been updated since July 1, 2023, and the workspace ID is 74125?"
        ],
    },
    {
        "id": "TC_9",
        "mandatory_params": [],
        "input": "email((string)) := feedback@example.com \nfilter((string)) := deleted \ninclude((string)) := stats \norder_type((string)) := desc \npage((integer)) := 8 \nper_page((integer)) := 80 \nrequester_id((string)) := 85214 \ntype((string)) := service_request \nupdated_since((string)) := 2023-08-01T00:00:00Z \nworkspace_id((integer)) := 96325 \n",
        "nl_utterance": [
            "Show me all service requests in Freshservice where the email is feedback@example.com, the filter is deleted, the include is stats, the order type is descending, the page is 8, the number of entries per page is 80, the requester ID is 85214, the type is service request, the updated time is after 2023-08-01T00:00:00Z, and the workspace ID is 96325."
        ],
    },
    {
        "id": "TC_10",
        "mandatory_params": [],
        "input": "email((string)) := sales@example.com \nfilter((string)) := watching \ninclude((string)) := requester,stats \norder_type((string)) := asc \npage((integer)) := 9 \nper_page((integer)) := 90 \nrequester_id((string)) := 74185 \ntype((string)) := problem \nupdated_since((string)) := 2023-09-01T00:00:00Z \nworkspace_id((integer)) := 85214 \n",
        "nl_utterance": [
            'Show me all tickets in Freshservice where the email is "sales@example.com", the filter is "watching", the include is "requester,stats", the order\\_type is "asc", the page is "9", the per\\_page is "90", the requester\\_id is "74185", the type is "problem", the updated\\_since is "2023-09-01T00:00:00Z", and the workspace\\_id is "85214".'
        ],
    },
]
agent_with_tools = get_react_agent(
    python_tool_string,
    python_tool_name,
    agent_llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
)
config = ToolValidationConfig(report_level="detailed")


# @pytest.mark.skipif(sys.version_info < (3, 12), reason="Toolops requires Python 3.12")
def test_tool_validation_with_toolkit_interface():
    tool_validation_input = ToolValidationInput(
        python_tool_name=python_tool_name,
        tool_test_cases=tool_test_cases,
        agent_with_tools=agent_with_tools,
    )

    tool_validation_middleware = PythonToolValidationComponent()
    result = tool_validation_middleware.process(
        data=tool_validation_input, config=config, phase=AgentPhase.RUNTIME
    )
    # import json
    # with open('tool_test_report','w') as trf:
    #     json.dump(result.test_report,trf,indent=2)
    # trf.close()
    assert result.test_report is not None
    assert (
        isinstance(result.test_report, dict)
        and "tool_test_cases" in result.test_report
        and len(result.test_report.get("tool_test_cases")) == 10
    )
