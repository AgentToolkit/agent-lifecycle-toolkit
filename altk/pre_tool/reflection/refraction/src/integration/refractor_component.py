# from typing import List, Any, Dict
#
# # from jinja2 import Template
# from pydantic import BaseModel
# from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
# from langflow.custom import Component
# from langflow.schema import Data
# from lfx.log.logger import logger
# from langflow.io import (
#     DataInput,
#     Output,
#     MessageTextInput,
#     BoolInput,
#     FloatInput,
# )
#
# import re
# import json
#
#
# class Instruction(BaseModel):
#     tools: List[Dict[str, Any]] = []
#     tool_calls: List[Dict[str, Any]] = []
#     memory: Dict[str, Any] = {}
#
#
# class IBMRefractorComponent(Component):
#     """A component that processes tool call(s) for syntactic validation."""
#
#     display_name: str = "IBM Refractor"
#     description: str = (
#         "The refractor is a non-LLM model-agnostic domain-agnostic processor"
#         " for tool calling."
#     )
#     documentation: str = "https://github.com/IBM/refraction/wiki"
#     icon: str = "bug-off"
#     priority: int = 100
#     name: str = "refractor"
#
#     inputs = [
#         MessageTextInput(
#             name="instruction",
#             display_name="Instruction",
#             info="Prompt + LLM generated text for tool calling",
#         ),
#         DataInput(
#             name="data",
#             display_name="Tool Calling Information",
#             info="List of tool calls",
#         ),
#         BoolInput(
#             name="named_operators_only",
#             display_name="Use named operators only",
#             value=True,
#         ),
#         BoolInput(name="compression", display_name="Compress?", value=False),
#         FloatInput(
#             name="mapping_strength", display_name="Mapping Strength", value=0.65
#         ),
#     ]
#
#     outputs = [
#         Output(
#             name="refraction_response", display_name="Refract", method="refract"
#         )
#     ]
#
#     def _pre_run_setup(self) -> None:
#         self.refractor = Refractor(catalog=[])
#         self.refractor.initialize_maps(mapping_threshold=self.mapping_strength)
#
#     def refract(self) -> Data:
#         instruction_object = parse_instruction(self.instruction)
#
#         if len(self.refractor.catalog.apis) == 0:
#             self.refractor = Refractor(catalog=instruction_object.tools)
#             self.refractor.initialize_maps(
#                 mapping_threshold=self.mapping_strength
#             )
#
#         tool_calls = instruction_object.tool_calls or self.data
#
#         logger.info(instruction_object.model_dump())
#         logger.info(tool_calls)
#         logger.info(self.refractor.catalog.apis)
#
#         refraction_result = self.refractor.refract(
#             tool_calls,
#             mappings=self.refractor.mappings,
#             memory_objects=instruction_object.memory,
#             compression=self.compression,
#             use_given_operators_only=self.named_operators_only,
#         )
#
#         return Data(
#             data=refraction_result.model_dump(),
#             text=refraction_result.diff,
#         )
#
#
# def parse_instruction(instruction: str) -> Instruction:
#     instruction_dict: Dict[str, Any] = dict()
#
#     for key in Instruction().model_fields.keys():
#         parsed_item = parse_item(text=instruction, key=key)
#
#         if parsed_item is not None:
#             instruction_dict[key] = parsed_item
#
#     return Instruction(**instruction_dict)
#
#
# def parse_item(text: str, key: str) -> Any:
#     match = re.search(
#         pattern=rf".*<{key}>(?P<{key}>.*)</{key}>.*",
#         string=text,
#         flags=re.DOTALL,
#     )
#
#     if match is None:
#         return None
#
#     else:
#         match_str = match.group(key)
#
#         try:
#             match_json = json.loads(match_str)
#             return match_json
#
#         except json.JSONDecodeError:
#             return None
#
#
# # tools = [
# #     {
# #         "name": "w3",
# #         "description": "Get employee information from w3",
# #         "parameters": {
# #             "type": "object",
# #             "properties": {
# #                 "email": {
# #                     "type": "string",
# #                     "description": "Employee email",
# #                 }
# #             },
# #             "required": ["email"],
# #         },
# #         "output_parameters": {
# #             "type": "object",
# #             "properties": {
# #                 "id": {
# #                     "type": "id",
# #                     "description": "Employee id",
# #                 }
# #             },
# #         },
# #     },
# #     {
# #         "name": "author_workbench",
# #         "description": "Get employee publication information",
# #         "parameters": {
# #             "type": "object",
# #             "properties": {
# #                 "id": {
# #                     "type": "string",
# #                     "description": "Employee id",
# #                 }
# #             },
# #             "required": ["id"],
# #         },
# #         "output_parameters": {
# #             "type": "array",
# #             "items": {
# #                 "type": "object",
# #                 "properties": {
# #                     "papers": {
# #                         "type": "object",
# #                         "description": "Paper information",
# #                     }
# #                 },
# #             },
# #         },
# #     },
# #     {
# #         "name": "hr_bot",
# #         "description": "Get employee information from HR",
# #         "parameters": {
# #             "type": "object",
# #             "properties": {
# #                 "id": {
# #                     "type": "string",
# #                     "description": "Employee id",
# #                 },
# #                 "email": {
# #                     "type": "string",
# #                     "description": "Employee email",
# #                 },
# #             },
# #         },
# #         "output_parameters": {
# #             "type": "object",
# #             "properties": {
# #                 "id": {
# #                     "type": "string",
# #                     "description": "Employee id",
# #                 },
# #                 "info": {
# #                     "type": "object",
# #                     "description": "Employee information",
# #                 },
# #             },
# #         },
# #     },
# #     {
# #         "name": "concur",
# #         "description": "Apply for travel approval",
# #         "parameters": {
# #             "type": "object",
# #             "properties": {
# #                 "employee_info": {
# #                     "type": "object",
# #                     "description": "Employee information",
# #                 },
# #                 "travel_justification": {
# #                     "type": "array",
# #                     "items": {
# #                         "type": "object",
# #                         "properties": {
# #                             "paper": {
# #                                 "type": "object",
# #                                 "description": "Paper information",
# #                             }
# #                         },
# #                     },
# #                 },
# #             },
# #             "required": ["employee_info", "travel_justification"],
# #         },
# #     },
# # ]
# #
# # tool_calls = [
# #     {
# #         "name": "w3",
# #         "arguments": {"email": "tchakra2@ibm.com"},
# #     },
# #     {
# #         "name": "author_workbench",
# #         "arguments": {"id": "$var1.id$"},
# #     },
# #     {
# #         "name": "hr_bot",
# #         "arguments": {"id": "$var1.id$", "email": "tchakra2@ibm.com"},
# #     },
# #     {
# #         "name": "concur",
# #         "arguments": {
# #             "employee_info": "$var3.info$",
# #             "travel_justification": "$var2.papers$",
# #         },
# #     },
# # ]
# #
# # memory = {}
# #
# # prompt = Template("""
# # You are an expert in correcting tool calls. You are given a set of available tools, a query and an incorrect tool call that was meant to satisfy the query.
# #
# # The user said: start a conference travel approval process for me
# #
# # You have access to the following memory:
# # <tools>{{ tools | tojson }}</tools>
# #
# # <memory>{{ memory | tojson }}</memory>
# #
# # The output must strictly adhere to the following format, and NO OTHER TEXT must be included:
# #
# # <tool_calls>{{ tool_calls | tojson }}</tool_calls>
# # """)
# #
# # instruction = """
# # You are an expert in correcting tool calls. You are given a set of available tools, a query and an incorrect tool call that was meant to satisfy the query.
# #
# # The user said: start a conference travel approval process for me
# #
# # You have access to the following memory:
# # <tools>[{"description": "Get employee information from w3", "name": "w3", "output_parameters": {"properties": {"id": {"description": "Employee id", "type": "id"}}, "type": "object"}, "parameters": {"properties": {"email": {"description": "Employee email", "type": "string"}}, "required": ["email"], "type": "object"}}, {"description": "Get employee publication information", "name": "author_workbench", "output_parameters": {"items": {"properties": {"papers": {"description": "Paper information", "type": "object"}}, "type": "object"}, "type": "array"}, "parameters": {"properties": {"id": {"description": "Employee id", "type": "string"}}, "required": ["id"], "type": "object"}}, {"description": "Get employee information from HR", "name": "hr_bot", "output_parameters": {"properties": {"id": {"description": "Employee id", "type": "string"}, "info": {"description": "Employee information", "type": "object"}}, "type": "object"}, "parameters": {"properties": {"email": {"description": "Employee email", "type": "string"}, "id": {"description": "Employee id", "type": "string"}}, "type": "object"}}, {"description": "Apply for travel approval", "name": "concur", "parameters": {"properties": {"employee_info": {"description": "Employee information", "type": "object"}, "travel_justification": {"items": {"properties": {"paper": {"description": "Paper information", "type": "object"}}, "type": "object"}, "type": "array"}}, "required": ["employee_info", "travel_justification"], "type": "object"}}]</tools>
# #
# # <memory>{}</memory>
# #
# # The output must strictly adhere to the following format, and NO OTHER TEXT must be included:
# #
# # <tool_calls>[{"arguments": {"email": "tchakra2@ibm.com"}, "name": "w3"}, {"arguments": {"id": "$var1.id$"}, "name": "author_workbench"}, {"arguments": {"email": "tchakra2@ibm.com", "id": "$var1.id$"}, "name": "hr_bot"}, {"arguments": {"employee_info": "$var3.info$", "travel_justification": "$var2.papers$"}, "name": "concur"}]</tool_calls>
# # """
# #
# # print(prompt.render(tools=tools, tool_calls=tool_calls, memory=memory))
# # print(json.dumps(parse_instruction(instruction).model_dump(), indent=4))
#
# # refractor = Refractor(catalog=tools)
# # refractor.initialize_maps(mapping_threshold=0.65)
# #
# # print(refractor.mappings)
