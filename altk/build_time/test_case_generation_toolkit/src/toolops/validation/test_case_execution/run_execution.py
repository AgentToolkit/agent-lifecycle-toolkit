"""
This module is for testing a tool in agentic environment
"""

import os
import json
import uuid
from langchain_core.load.dump import dumps as langchain_dumps
import logging

logger = logging.getLogger(__name__)
pwd = os.getcwd()


class ToolNLTestCaseExecution:
    def __init__(self, agent_with_tools):
        """
        args:
            agent_with_tools : Langgraph type react agent with tools binded to the agent
        """
        self.agent_with_tools = agent_with_tools

    def run_tool_nl_test_cases(self, tool_test_cases, tool_name):
        """
        Method to execute tool test cases with NL utterances in agentic flow
        """
        test_cases_with_tool_execution = []
        try:
            agentic_flow = self.agent_with_tools
            logger.info("Started tool test cases execution")
            for i, tc in enumerate(tool_test_cases):
                logger.info("Executing test case : " + str(i + 1))
                input_utterances = tc.get("nl_utterance")
                tool_execution_responses = []
                for j, input_utterance in enumerate(input_utterances):
                    logger.info("Testing tool nl utterance : " + str(j + 1))
                    thread_id = "thread_id_" + uuid.uuid4().hex
                    config = {"configurable": {"thread_id": thread_id}}
                    all_events = []
                    logger.info("Agent flow execution events ")
                    try:
                        for i, event in enumerate(
                            agentic_flow.stream(
                                {"messages": [("user", input_utterance)]},
                                config,
                                stream_mode="updates",
                            )
                        ):
                            turn_id = "turn_id_" + str(i + 1)
                            turn_event = {
                                "turn_id": turn_id,
                                "turn_event": langchain_dumps(event),
                            }
                            all_events.append(turn_event)
                            logger.info(
                                "Agent execution turn event - " + json.dumps(turn_event)
                            )
                    except Exception as e:
                        logger.error(
                            "Exception occured in running agentic flow , " + str(e)
                        )
                        pass
                    tool_response = {
                        "utterance": input_utterance,
                        "agentic_flow_events": all_events,
                    }
                    tool_execution_responses.append(tool_response)
                tc["tool_execution_responses"] = tool_execution_responses
                tc["tool_name"] = tool_name
                test_cases_with_tool_execution.append(tc)
            logger.info("Completed tool test cases execution")
        except Exception as e:
            logger.info("Failed to run tool test cases , error - " + str(e))
            test_cases_with_tool_execution = tool_test_cases
        return test_cases_with_tool_execution


# if __name__=="__main__":
#     # create agents and create tools modules below are not required if the react agent with binded tools is provided
#     from altk.build_time.test_case_generation_toolkit.src.toolops.validation.test_case_execution.test_case_execution_utils.agentenv.fr_langgraph.create_agents import get_react_agent,get_agent_llm
#     from altk.build_time.test_case_generation_toolkit.src.toolops.validation.test_case_execution.test_case_execution_utils.agentenv.fr_langgraph.create_tools import get_tools
#     test_data_path = os.path.join(pwd,"test_data","test_case_execution")
#     output_report_path = os.path.join(pwd,"test_data","test_case_execution")
#     test_tool_def_str = open(os.path.join(test_data_path,'getApiV2Tickets_tool.py')).read()
#     tool_details = [{'tool_name':'getApiV2Tickets','tool_def_str':test_tool_def_str}]
#     tool_name = tool_details[0].get('tool_name')
#     agent_llm_model_id="mistralai/mistral-small-3-1-24b-instruct-2503"
#     tool_names , tools = get_tools(tool_details)
#     agent_llm = get_agent_llm(agent_llm_model_id)
#     agent_with_tools = get_react_agent(agent_llm,tools)
#     tool_test_cases = json.load(open(os.path.join(test_data_path,"getApiV2Tickets_tool_test_cases.json"),'r'))


#     # running tool test cases with sample inputs
#     tool_nl_tc_execution = ToolNLTestCaseExecution(agent_with_tools)
#     test_cases_with_tool_execution = tool_nl_tc_execution.run_tool_nl_test_cases(tool_test_cases,tool_name)
#     with open(os.path.join(output_report_path,tool_name+'_test_cases_with_tool_execution.json'),'w') as ttr:
#         json.dump(test_cases_with_tool_execution,ttr,indent=2)
#     ttr.close()

#     # filtered_tool_test_cases = []
#     # for tc in tool_test_cases:
#     #     del tc["scenario_type"]
#     #     del tc["input_parameters"]
#     #     filtered_tool_test_cases.append(tc)

#     # with open(os.path.join(output_report_path,'getApiV2Tickets_tool_test_cases.json'),'w') as ttr:
#     #     json.dump(filtered_tool_test_cases,ttr,indent=2)
#     # ttr.close()
