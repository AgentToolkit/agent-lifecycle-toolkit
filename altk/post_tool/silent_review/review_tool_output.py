from typing import Tuple
from altk.core.llm.types import GenerationMode
from altk.post_tool.silent_review.review_prompts import (
    REVIEW_JSON_PROMPT,
    REVIEW_TABULAR_PROMPT,
)
from altk.core.utils import parse_result_for_json
import json
from enum import Enum


class Outcome(Enum):
    """Used for Review"""

    NOT_ACCOMPLISHED = 0
    PARTIAL_ACCOMPLISH = 0.5
    ACCOMPLISHED = 1


class BaseReviewToolOutputUtil:
    prompt_template: str = ""

    def _build_prompt(self, query, tool_spec, tool_response) -> str:
        tool = tool_spec or ""
        return self.prompt_template.format(
            question=query,
            tool=tool,
            API_response=tool_response,
        )

    def _parse_result(self, result: str) -> Tuple[Outcome, dict]:
        try:
            result_json = json.loads(result)
            result_outcome = result_json.get("overall_assessment", "")
            outcome = {
                "Accomplished": Outcome.ACCOMPLISHED,
                "Partially Accomplished": Outcome.PARTIAL_ACCOMPLISH,
            }.get(result_outcome, Outcome.NOT_ACCOMPLISHED)
            return outcome, result_json
        except json.JSONDecodeError:
            return Outcome.NOT_ACCOMPLISHED, {"result (malformed)": result}

    def process(
        self, llm, query, tool_spec, tool_response, **kwargs
    ) -> Tuple[Outcome, dict]:
        prompt = self._build_prompt(query, tool_spec, tool_response)
        result = llm.generate(prompt)
        parsed_result = parse_result_for_json(result)
        return self._parse_result(parsed_result)

    async def aprocess(
        self, llm, query, tool_spec, tool_response, **kwargs
    ) -> Tuple[Outcome, dict]:
        prompt = self._build_prompt(query, tool_spec, tool_response)
        result = await llm.generate_async(prompt, mode=GenerationMode.CHAT_ASYNC)
        parsed_result = parse_result_for_json(result)
        return self._parse_result(parsed_result)


class ReviewJSONToolOutputUtil(BaseReviewToolOutputUtil):
    prompt_template = REVIEW_JSON_PROMPT


class ReviewTabularToolOutputUtil(BaseReviewToolOutputUtil):
    prompt_template = REVIEW_TABULAR_PROMPT
