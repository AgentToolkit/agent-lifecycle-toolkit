# Tool Validation Component

This component validates Python tools by running test cases with a ReAct agent bound to the required tools. After execution, the tool logs are analyzed to identify error types and provide corresponding recommendations.

## Table of Contents
  - [When it is Recommended to Use This Component](#when-it-is-recommended-to-use-this-component)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Input](#input)
  - [License](#license)

## When it is Recommended to Use This Component
The component executes the test cases (can be generated using [Test Case Generation Toolkit](../test_case_generation_toolkit)) with the agent and analyses the tool execution logs to identify different types of tool errors categorized under the defined error taxonomy. Rule-based recommendations are provided different types of tool errors.
## Features
This component expects three inputs 1) tool name, 2) tool test cases, 3) ReAct agent bound to the required tools. Output from this module is the tool test report with identified errors and recommendations for each tool test case:
- Tool name - corresponds to the tool that is used of validation.
- Tool test cases - multiple tool test cases used for testing the tool with an agent. A sample test case is provided below:
    ```
        {
        "id": "TC_2",
        "mandatory_params": [],
        "input": "email((string)) := test@example.com \nfilter((string)) := new \ninclude((string)) := requester \norder_type((string)) := asc \npage((integer)) := 1 \nper_page((integer)) := 10 \nrequester_id((string)) := 12345 \ntype((string)) := incident \nupdated_since((string)) := 2023-01-01T00:00:00Z \nworkspace_id((integer)) := 67890 \n",
        "nl_utterance": [
        "Show me the new incident tickets in Freshservice with requester id 12345, email test@example.com, updated since 2023-01-01T00:00:00Z, and workspace id 67890. I want to see the requester details and display 10 tickets per page, sorted in ascending order. Start from page 1."
        ]
    }
    ```
- React agent with tools - We use langgraph type react agent bounded to the necessary tools to execute tool test cases.
- Output from this module is tool test report with identified tool error taxonomy, recommendations for each test case (additionally tool execution logs can be added). A sample output is provided below:
    ```
    {
      "test_utterance": "Show me the new incident tickets in Freshservice with requester id 12345, email test@example.com, updated since 2023-01-01T00:00:00Z, and workspace id 67890. I want to see the requester details and display 10 tickets per page, sorted in ascending order. Start from page 1.",
      "tool_mandatory_inputs": [],
      "tool_inputs": "email((string)) := test@example.com \nfilter((string)) := new \ninclude((string)) := requester \norder_type((string)) := asc \npage((integer)) := 1 \nper_page((integer)) := 10 \nrequester_id((string)) := 12345 \ntype((string)) := incident \nupdated_since((string)) := 2023-01-01T00:00:00Z \nworkspace_id((integer)) := 67890 \n",
      "test_status": "Failed",
      "test_error_taxonomy": [
        "Incorrect tool inputs - Parameter Type Mismatch"
      ],
      "test_error_recommendations": [
        "Check tool input parameter type mismatch for : [{'parameter': 'filter', 'expected_param_type': 'string', 'tool_input_param_type': None}, {'parameter': 'requester_id', 'expected_param_type': 'string', 'tool_input_param_type': 'integer'}] , *** please modify the tool input processing code accordingly ***"
      ],
      "tool_execution_events": {
        "utterance": "Show me the new incident tickets in Freshservice with requester id 12345, email test@example.com, updated since 2023-01-01T00:00:00Z, and workspace id 67890. I want to see the requester details and display 10 tickets per page, sorted in ascending order. Start from page 1.",
        "agentic_flow_events": [
          {
            "turn_id": "turn_id_1",
            "turn_event": "{\"agent\": {\"messages\": [{\"lc\": 1, \"type\": \"constructor\", \"id\": [\"langchain\", \"schema\", \"messages\", \"AIMessage\"], \"kwargs\": {\"content\": \"\", \"additional_kwargs\": {\"tool_calls\": [{\"id\": \"wKiT6MFRX\", \"type\": \"function\", \"function\": {\"name\": \"getApiV2Tickets\", \"arguments\": \"{\\\"getApiV2TicketsInput\\\": {\\\"type\\\": \\\"incident\\\", \\\"requester_id\\\": 12345, \\\"email\\\": \\\"test@example.com\\\", \\\"updated_since\\\": \\\"2023-01-01T00:00:00Z\\\", \\\"workspace_id\\\": 67890, \\\"include\\\": \\\"requester\\\", \\\"per_page\\\": 10, \\\"order_type\\\": \\\"asc\\\", \\\"page\\\": 1}}\"}}]}, \"response_metadata\": {\"token_usage\": {\"completion_tokens\": 127, \"prompt_tokens\": 164, \"total_tokens\": 291}, \"model_name\": \"mistralai/mistral-large\", \"system_fingerprint\": \"\", \"finish_reason\": \"tool_calls\"}, \"type\": \"ai\", \"id\": \"chatcmpl-a87d8d2287ef0f2e8962ccc05c5194f2\", \"tool_calls\": [{\"name\": \"getApiV2Tickets\", \"args\": {\"getApiV2TicketsInput\": {\"type\": \"incident\", \"requester_id\": 12345, \"email\": \"test@example.com\", \"updated_since\": \"2023-01-01T00:00:00Z\", \"workspace_id\": 67890, \"include\": \"requester\", \"per_page\": 10, \"order_type\": \"asc\", \"page\": 1}}, \"id\": \"wKiT6MFRX\", \"type\": \"tool_call\"}], \"usage_metadata\": {\"input_tokens\": 164, \"output_tokens\": 127, \"total_tokens\": 291}, \"invalid_tool_calls\": []}}]}}"
          },
          {
            "turn_id": "turn_id_2",
            "turn_event": "{\"tools\": {\"messages\": [{\"lc\": 1, \"type\": \"constructor\", \"id\": [\"langchain\", \"schema\", \"messages\", \"ToolMessage\"], \"kwargs\": {\"content\": \"{\\\"code\\\": \\\"access_denied\\\", \\\"message\\\": \\\"You are not authorized to perform this action.\\\"}\", \"type\": \"tool\", \"name\": \"getApiV2Tickets\", \"id\": \"91a0c3a2-d443-41c4-bc1f-a132efd8c293\", \"tool_call_id\": \"wKiT6MFRX\", \"status\": \"success\"}}]}}"
          },
          {
            "turn_id": "turn_id_3",
            "turn_event": "{\"agent\": {\"messages\": [{\"lc\": 1, \"type\": \"constructor\", \"id\": [\"langchain\", \"schema\", \"messages\", \"AIMessage\"], \"kwargs\": {\"content\": \" I'm sorry, but you don't have the necessary permissions to list all tickets. If you believe this is an error, please contact your administrator for assistance. Is there anything else I can help with?\", \"response_metadata\": {\"token_usage\": {\"completion_tokens\": 45, \"prompt_tokens\": 347, \"total_tokens\": 392}, \"model_name\": \"mistralai/mistral-large\", \"system_fingerprint\": \"\", \"finish_reason\": \"stop\"}, \"type\": \"ai\", \"id\": \"chatcmpl-88fb40176c7dde0f096017355ffe129c\", \"usage_metadata\": {\"input_tokens\": 347, \"output_tokens\": 45, \"total_tokens\": 392}, \"tool_calls\": [], \"invalid_tool_calls\": []}}]}}"
          }
        ]
      }
    }
    ```


#### Tool validation error types and corresponding recommendations
| error_type                                             | Recommendation                                                                                                                                                                                                                                   |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Incorrect tool identification                          | Agent identified incorrect tool - expected tool:<tool_name>,identified tool:<tool_name>  *** please modify or add the tool descriptions accordingly ***                                                                                          |
| Agent tool calling issue                               | Agent could not invoke tool calling for the identified tool - <tool_name> *** please check the tool definition ***                                                                                                                               |
| Agent tool calling issue - Incorrect Tool Input format | Agent could not invoke tool calling for the identified tool - <tool_name>  due to malformed tool input payload format , *** please modify the tool input processing code ***                                                                     |
| Incorrect tool input payload                           | Check tool input payload for : Missing or Incorrect parameters <parameters>  *** please modify tool input processing code accordingly,  provide parameter descriptions in the tool definition for the agent to correctly identify parameters *** |
| Incorrect tool inputs - Parameter Type Mismatch        | Check tool input parameter type mismatch for : <parameter, expected_param_type, tool_input_param_type> *** please modify the tool input processing code accordingly ***                                                                          |
| Incorrect tool inputs - Parameter Value Mismatch       | Check tool input parameter value mismatch for : <parameter, expected_param_value, tool_input_param_value> *** please provide parameter descriptions, examples in the tool definition, for the agent to correctly identify parameter values       |
| Agent Recursive Tool Calling                           | Agent invoked the tool repeatedly several times, tools invoked - <tool_name : tool_call_count> *** please modify the output returned in the tool definition accordingly, such that agent will not invoke the tool recursively ***                |
| Tool Output parsing error                              | Could not parse the tool output, *** please check output returned in tool definition ***                                                                                                                                                         |
| Tool Output exceeding LLM token limit                  | Agent LLM exceeding token limit, *** please try with a different LLM with large token limit ***                                                                                                                                                  |
| Incorrect tool autorization or access credentials      | please provide valid credentials for the tool                                                                                                                                                                                                    |
| Tool back-end server issue                             | Tool server is down, try after some time                                                                                                                                                                                                         |
| Issue with the inputs provided to the tool             | Tool input malformed, please provide tool inputs in correct format                                                                                                                                                                               |
| No results found in the tool output                    | Please provide relevant tool input values                                                                                                                                                                                                        |
## Quick Start

### Configuration
This component supports configuring the tool test report level and by default `detailed` report is generated as output.
```python
from altk.build_time.tool_validation_toolkit.core.config import ToolValidationConfig
config = ToolValidationConfig(report_level='detailed')
```

### Examples
* Tool validation requires `python_tool_name` , `tool_test_cases` , `agent_with_tools` as inputs , below code examples show sample inputs and creating a langgraph type react with necessary tools.
    ```python
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
        "id": "TC_2",
        "mandatory_params": [],
        "input": "email((string)) := test@example.com \nfilter((string)) := new \ninclude((string)) := requester \norder_type((string)) := asc \npage((integer)) := 1 \nper_page((integer)) := 10 \nrequester_id((string)) := 12345 \ntype((string)) := incident \nupdated_since((string)) := 2023-01-01T00:00:00Z \nworkspace_id((integer)) := 67890 \n",
        "nl_utterance": [
          "Show me the new incident tickets in Freshservice with requester id 12345, email test@example.com, updated since 2023-01-01T00:00:00Z, and workspace id 67890. I want to see the requester details and display 10 tickets per page, sorted in ascending order. Start from page 1."
        ]
      },
      {
        "id": "TC_3",
        "mandatory_params": [],
        "input": "email((string)) := user@example.com \nfilter((string)) := open \ninclude((string)) := stats \norder_type((string)) := desc \npage((integer)) := 2 \nper_page((integer)) := 20 \nrequester_id((string)) := 67890 \ntype((string)) := service_request \nupdated_since((string)) := 2023-02-01T00:00:00Z \nworkspace_id((integer)) := 12345 \n",
        "nl_utterance": [
          "Show me all service requests in Freshservice where the requester's email is user@example.com, the status is open, and the updated time is after 2023-02-01T00:00:00Z. I would like to see the stats and order the results by descending order. Please display 20 results per page, starting from page 2, and only show me the requests with requester\\_id 67890 and workspace\\_id 12345."
        ]
      },
      {
        "id": "TC_4",
        "mandatory_params": [],
        "input": "email((string)) := admin@example.com \nfilter((string)) := closed \ninclude((string)) := requester,stats \norder_type((string)) := asc \npage((integer)) := 3 \nper_page((integer)) := 30 \nrequester_id((string)) := 24680 \ntype((string)) := problem \nupdated_since((string)) := 2023-03-01T00:00:00Z \nworkspace_id((integer)) := 98765 \n",
        "nl_utterance": [
          "Can you list all problem tickets in the workspace with id 98765, where the requester id is 24680, the email is admin@example.com, the filter is closed, the order type is ascending, the page is 3, the number of tickets per page is 30, and the tickets were updated since 2023-03-01T00:00:00Z? Additionally, can you include the requester, stats, and type information in the results?"
        ]
      }
    ]
    ```
* Creating a langgraph react agent with tools using WATSONX LLM Provider
    ```python
    # creating react agent with python tool
    import importlib.util
    import os
    def get_python_tool(python_tool_string,python_tool_name):
        spec = importlib.util.spec_from_loader('tool_py', loader=None)
        tool_py = importlib.util.module_from_spec(spec)
        exec(python_tool_string, tool_py.__dict__)
        return tool_py.__getattribute__(python_tool_name)

    # adding react agent llm code for WATSONX
    from ibm_watsonx_ai import Credentials as wx_credentials
    from langchain_ibm import ChatWatsonx
    from langgraph.prebuilt import create_react_agent

    def get_agent_llm(agent_llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503"):
        WATSONX_URL = os.getenv("WX_URL","https://us-south.ml.cloud.ibm.com")
        WATSONX_API_KEY = os.getenv("WX_API_KEY","")
        WATSONX_PROJECT = os.getenv("WX_PROJECT_ID","")

        # set "WATSONX_API_KEY" env variable as required by ChatWatsonx Model
        os.environ["WATSONX_API_KEY"]=WATSONX_API_KEY
        credentials = wx_credentials(
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY
        )
        project_id = WATSONX_PROJECT
        try:
            llm_parameters = { "decoding_method": "greedy","max_new_tokens": 800,"min_new_tokens": 1}
            wx_chat_llm = ChatWatsonx(
                model_id=agent_llm_model_id,
                url= WATSONX_URL,
                project_id=project_id,
                credentials=credentials,

                params=llm_parameters,
            )
            return wx_chat_llm
        except Exception as e:
            print("Please check if all WatsonX related environment varaibles - WX_URL , WX_API_KEY , WX_PROJECT_ID are set")
            print("Error in react agent llm configuration - ",e)


    def get_react_agent(python_tool_string,python_tool_name,agent_llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503"):
        tool = get_python_tool(python_tool_string,python_tool_name)
        tools = [tool]
        agent_llm = get_agent_llm(agent_llm_model_id)
        react_agent_with_tools = create_react_agent(agent_llm, tools)
        print("Created react agent with tools")
        return react_agent_with_tools

    agent_with_tools = get_react_agent(python_tool_string,python_tool_name,agent_llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503")
    ```
* Running tool validation module with required inputs,configuration and obtaining the tool test report
    ```python
    from altk.core.toolkit import AgentPhase
    from altk.build_time.tool_validation_toolkit.core.toolkit import ToolValidationInput
    from altk.build_time.tool_validation_toolkit.core.config import ToolValidationConfig
    from altk.build_time.tool_validation_toolkit.utils.tool_validation import ToolValidationComponent,PythonToolValidationComponent
    tool_validation_input = ToolValidationInput(python_tool_name=python_tool_name,
                                                    tool_test_cases=tool_test_cases,
                                                    agent_with_tools=agent_with_tools)
    config = ToolValidationConfig(report_level='detailed')
    tool_validation_middleware = PythonToolValidationComponent()
    result = tool_validation_middleware.process(
        data=tool_validation_input, config=config, phase=AgentPhase.RUNTIME
    )
    print(result.test_report)
    ```

## Input
The toolkit expects `python_tool_name` , `tool_test_cases` , `agent_with_tools` as inputs and examples are shown above.

* Note : We support only langgraph type ReAct agent bounded to the tools as the agent in this component.

### Default Configurations

The component provides a pre-configured tool validation profile for tool test report level:

#### `standard`
- **report_level**: `detailed`


## License

MIT License - see LICENSE file for details.
