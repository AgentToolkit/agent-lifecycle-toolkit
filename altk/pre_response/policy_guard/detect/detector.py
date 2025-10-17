from abc import abstractmethod
import json
import re
from typing import Set

from pydantic import Field

from altk.core.toolkit import ComponentBase, ComponentConfig, AgentPhase
from altk.pre_response.policy_guard.core.utils import get_model_name
from altk.pre_response.policy_guard.detect.detect_prompts import batch_detect_prompt
from altk.pre_response.policy_guard.core.toolkit import (
    PolicyDetectorInput,
    PolicyDetectorSingleOutput,
    PolicyDetectorOutput,
)

DEFAULT_MODEL_NAME = "llama"
DEFAULT_DETECTOR_PROVIDER = "watsonx"


# Utility function to help with unmatched quotes in the instructions
def escape_inner_quotes_in_policy(json_str):
    # Regex to isolate the "policy" value
    pattern = r'("policy"\s*:\s*")(.+?)(")(\s*[,}])'

    def replacer(match):
        prefix, value, closing_quote, end = match.groups()
        # Escape inner quotes (not the closing quote)
        fixed_value = value.replace('"', r"\"")
        return f'{prefix}{fixed_value}"{end}'

    return re.sub(pattern, replacer, json_str, flags=re.DOTALL)


def extract_policy_chunks(text):
    chunks = []
    search_pos = 0

    while True:
        idx = text.find('"policy":', search_pos)
        if idx == -1:
            break

        # Search left for the opening '{'
        left = idx
        while left >= 0 and text[left] != "{":
            left -= 1
        if left < 0:
            search_pos = idx + 1
            continue  # malformed; skip

        # Search right for the closing '}'
        right = idx
        brace_count = 0
        while right < len(text):
            if text[right] == "{":
                brace_count += 1
            elif text[right] == "}":
                brace_count -= 1
                if brace_count == 0:
                    break
            right += 1

        if right >= len(text):
            break  # unmatched brace

        chunk = text[left : right + 1]
        chunks.append(chunk)
        search_pos = right + 1  # move forward

    return chunks


def parse_response(responses: str) -> list[dict]:
    try:
        responses = json.loads(responses)
    except Exception:
        responses = (
            responses.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        # strip of anything trailing the json list
        responses = responses[: (responses.rfind("]") + 1)]
        responses = responses.replace(responses.split("[")[0], "", 1)
        try:
            responses = json.loads(responses)
        except Exception:
            cleaned = escape_inner_quotes_in_policy(responses)
            try:
                responses = json.loads(cleaned)
            except Exception:
                # Custom parser
                chunks = extract_policy_chunks(cleaned)
                parsed = []
                for chunk in chunks:
                    try:
                        subdict = json.loads(chunk)
                        parsed.append(subdict)
                    except Exception:
                        parsed.append({"answer": "False"})
                return parsed

    if isinstance(responses, dict):
        responses = [responses]
    for r_parse in responses:
        if "answer" in r_parse:
            r_parse["answer"] = True if r_parse["answer"] == "yes" else False
        else:
            r_parse["answer"] = None
    return responses


SINGLE_DETECTOR_NAME = "single_policy_llm_detector"
BATCH_DETECTOR_NAME = "batch_policy_llm_detector"


class Detector(ComponentBase):
    # LLM config here
    config: ComponentConfig = Field(default_factory=ComponentConfig)
    name: str

    def detect_endpoint(self, prompt: str) -> str:
        response = self.config.llm_client.generate(prompt)
        return response

    @abstractmethod
    def detect(self, detector_input: PolicyDetectorInput) -> PolicyDetectorOutput:
        raise NotImplementedError("Subclass must implement `detect` method. ")

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        """
        The detector middleware is only supported at runtime
        Returns:
            Set[AgentPhase]: A set of supported AgentPhase.
        """
        return {AgentPhase.RUNTIME}

    def process(
        self, data: PolicyDetectorInput, phase: AgentPhase
    ) -> PolicyDetectorOutput:
        """
        Detect policy violations.
        Args:
            data (PolicyDetectorInput): Input data for the middleware.
            phase (AgentPhase): The phase in which the middleware is being executed.

        Returns:
            PolicyDetectorOutput: Processed output data.
        Raises:
            ValueError: If the phase is not supported by the middleware.
            NotImplementedError: If the phase is not handled in the middleware.
        """
        if phase not in self.supported_phases():
            raise ValueError(
                f"{self.__class__.__name__} does not support phase {phase}"
            )

        if phase == AgentPhase.BUILDTIME:
            return self._build(data)
        elif phase == AgentPhase.RUNTIME:
            return self._run(data)

        raise NotImplementedError(f"Unhandled phase: {phase}")

    def _run(self, data: PolicyDetectorInput) -> PolicyDetectorOutput:
        """
        This method is called during the RUNTIME phase.
        Args:
            data (PolicyDetectorInput): Input data for the run phase.
        """
        return self.detect(data)


class SinglePolicyLLMDetector(Detector):
    name: str = SINGLE_DETECTOR_NAME

    def detect(self, detector_input: PolicyDetectorInput) -> PolicyDetectorOutput:
        outputs = []
        for policy in detector_input.policies:
            policy_list = [policy]
            prompt = batch_detect_prompt(
                get_model_name(self.config.llm_client.model_id),
                detector_input.response,
                policy_list,
            )
            # response = self.detect_endpoint(prompt, self.model_id, return_counts=False)
            response = self.detect_endpoint(prompt)
            try:
                parsed_response = parse_response(response)
                res = parsed_response[0]

                output = PolicyDetectorSingleOutput(
                    policy=policy,
                    compliance=res["answer"],
                    explanation=res.get("explanation", ""),
                )

            except Exception as e:
                print("\n\n" + f"Detection exception: {e}" + "\n\n")
                output = PolicyDetectorSingleOutput(
                    policy=policy, compliance=False, explanation=""
                )

            outputs.append(output)
        return PolicyDetectorOutput(policy_outputs=outputs)


class BatchPolicyLLMDetector(Detector):
    name: str = BATCH_DETECTOR_NAME

    def detect(self, detector_input: PolicyDetectorInput) -> PolicyDetectorOutput:
        # NOTE: model_id/name is not guaranteed depending on the provider
        model_id = ""
        try:
            model_id = self.config.llm_client.model_id
        except AttributeError:
            model_id = self.config.llm_client.model_name
        prompt = batch_detect_prompt(
            get_model_name(model_id), detector_input.response, detector_input.policies
        )
        # responses = self.detect_endpoint(prompt, self.model_id, return_counts=False)
        responses = self.detect_endpoint(prompt)
        try:
            parsed_responses = parse_response(responses)
            # cut off any extra detection output generated
            max_len = len(detector_input.policies)
            parsed_responses = parsed_responses[:max_len]

            outputs = []
            for idx, res in enumerate(parsed_responses):
                output = PolicyDetectorSingleOutput(
                    policy=detector_input.policies[idx],
                    compliance=res["answer"],
                    explanation=res.get("explanation", ""),
                )
                outputs.append(output)
            if len(outputs) != max_len:
                raise Exception(
                    f"Detector output parsed to {len(outputs)} sections, but {max_len} policies were called for. "
                )
        except Exception as e:
            print("\n\n" + f"Detection exception: {e}" + "\n\n")
            outputs = []
            for idx in range(len(detector_input.instruction_text_list)):
                x = PolicyDetectorSingleOutput(
                    policy=detector_input.policies[idx],
                    compliance=False,
                    explanation="",
                )
                outputs.append(x)
        return PolicyDetectorOutput(policy_outputs=outputs)
