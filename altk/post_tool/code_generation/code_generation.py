from llm_sandbox import SandboxSession
import logging
from altk.core.toolkit import AgentPhase, ComponentConfig
from typing import Set, Any
from altk.post_tool.core.toolkit import (
    PostToolReflectionComponent,
    CodeGenerationRunInput,
    CodeGenerationRunOutput,
)
from altk.post_tool.core.prompts import (
    CODE_GENERATION_ZERO_SHOT_PROMPT_WITH_RESPONSE_SCHEMA,
    CODE_GENERATION_ZERO_SHOT_PROMPT_WITH_COMPACT_RESPONSE,
    PROMPT_GET_NL_QUERY,
)
from pydantic import Field
from genson import SchemaBuilder
from smolagents.default_tools import BASE_PYTHON_TOOLS
from smolagents.local_python_executor import evaluate_python_code

logger = logging.getLogger(__name__)


class InvalidCodeException(Exception):
    pass


class CodeGenerationComponentConfig(ComponentConfig):
    """Configuration for CodeGen, includes LLMClient by default"""

    use_docker_sandbox: bool = Field(
        default=True,
        description="Set True to use a docker container as a sandbox. Set to False to use direct Python execution. Directly executed Python code is cleaned to be more safe but still carries a risk.",
    )


class CodeGenerationComponent(PostToolReflectionComponent):
    """Component responsible for generating code based on the user input and
    tool responses.
    """

    config: CodeGenerationComponentConfig = Field(
        default_factory=CodeGenerationComponentConfig
    )
    prompt_template: str = CODE_GENERATION_ZERO_SHOT_PROMPT_WITH_RESPONSE_SCHEMA
    retry: bool = True

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        return {AgentPhase.RUNTIME}

    def _parse_extraction_code(self, model_response: str) -> str:
        try:
            # TODO: simplify this using a regex
            start_idx = model_response.find("```python")
            if start_idx == -1:
                raise ValueError("Python code block not found in response.")

            model_response_first_part = model_response[start_idx + 9 :]
            end_idx = model_response_first_part.find("```")
            code = model_response_first_part[:end_idx].strip()

            if "# Example usage:" in code:
                logger.debug("Removing example usage section from code.")
                code = code.split("# Example usage:")[0]

            return code
        except Exception:
            logger.error("Error during code extraction: %s", exc_info=True)
            return ""

    def _apply_extraction_code_to_response(self, code: str, tool_response: str) -> Any:
        first_def = code.find("def")
        first_open_parenthesis = code.find("(")
        function_name = code[first_def + 3 : first_open_parenthesis].strip()

        script = f"""
{code}

result = {function_name}({tool_response})
print(result)
""".strip()
        if self.config.use_docker_sandbox:
            with SandboxSession(lang="python") as session:  # type: ignore
                execution_result = session.run(script)
                return execution_result.stdout
        else:
            state = {}
            try:
                evaluate_python_code(
                    script, BASE_PYTHON_TOOLS, state=state, authorized_imports=["json"]
                )
            except Exception:
                # TODO: better error handling?
                return None
            return state["_print_outputs"]

    def _get_relevant_data(self, prompt: str, data: CodeGenerationRunInput) -> str:
        try:
            logger.info("Calling LLM")
            model_response = self.config.llm_client.generate(prompt)
            extraction_code = self._parse_extraction_code(model_response)
            code_output = self._apply_extraction_code_to_response(
                extraction_code, data.tool_response
            )
            if not code_output:
                return "<failed_to_extract_information>"

            return str(code_output)

        except InvalidCodeException:
            if self.retry:
                # to get compact api response
                data.tool_response = self.reduce_json_by_unique_keys(data.tool_response)
                self.prompt_template = (
                    CODE_GENERATION_ZERO_SHOT_PROMPT_WITH_COMPACT_RESPONSE
                )
                self.retry = False
                self.reflect(data)
            else:
                logger.error(
                    "Exception occurred during code generation: %s", exc_info=True
                )
                return ""

    def reflect(
        self, data: CodeGenerationRunInput, **kwargs: Any
    ) -> CodeGenerationRunOutput:
        logger.info("Argument validations succeeded. Building prompt...")
        schema = self.get_schema(data.tool_response)
        prompt = self.prompt_template.replace(
            "<<task_prefix>>", data.nl_query.lower()
        ).replace(
            "<<json_obj>>",
            str(
                data.tool_response
            ).replace(  # need to provide only a sample of the response
                "<<json_schema>>", str(schema)
            ),
        )

        relevant_data = self._get_relevant_data(prompt, data)
        return CodeGenerationRunOutput(result=relevant_data)

    def _run(self, data: CodeGenerationRunInput) -> CodeGenerationRunOutput:  # type: ignore
        data.nl_query = self.get_nl_query(data)
        result = self.reflect(data)
        return result

    def get_nl_query(self, data: CodeGenerationRunInput, llm_thoughts: str = "") -> str:
        schema = self.get_schema(data.tool_response)
        prompt_template: str = PROMPT_GET_NL_QUERY
        user_query = data.nl_query
        prompt = (
            prompt_template.replace("<<user_query>>", user_query)
            .replace("<<llm_thought>>", llm_thoughts)
            .replace("<<json_obj>>", str(data.tool_response))
            .replace("<<json_schema>>", str(schema))
        )
        data.nl_query = self.config.llm_client.generate(prompt)
        return data.nl_query

    def get_all_keys(self, obj, parent_keys=None):
        """
        Recursively gather all unique keys in a JSON structure.
        """
        if parent_keys is None:
            parent_keys = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                parent_keys.add(k)
                self.get_all_keys(v, parent_keys)
        elif isinstance(obj, list):
            for item in obj:
                self.get_all_keys(item, parent_keys)
        return parent_keys

    def reduce_json_by_unique_keys(self, obj, known_keys=None):
        """
        Reduce JSON structure but keep items in lists that add new keys.
        """
        if known_keys is None:
            known_keys = set()

        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                reduced_value = self.reduce_json_by_unique_keys(v, known_keys)
                new_obj[k] = reduced_value
            return new_obj

        elif isinstance(obj, list):
            reduced_list = []
            for item in obj:
                item_keys = self.get_all_keys(item)
                if not item_keys.issubset(known_keys):
                    known_keys.update(item_keys)
                    reduced_list.append(
                        self.reduce_json_by_unique_keys(item, known_keys)
                    )
            return reduced_list

        else:
            return obj

    def get_schema(self, api_response: dict):
        builder = SchemaBuilder()
        builder.add_object(api_response)
        schema = builder.to_schema()
        return schema
