from typing import Set, Type, ClassVar
from altk.toolkit_core.core.toolkit import AgentPhase
from altk.post_tool_reflection_toolkit.core.toolkit import (
    PostToolReflectionComponent,
    SilentReviewRunInput,
    SilentReviewRunOutput,
    Outcome,
)
from altk.post_tool_reflection_toolkit.silent_review.review_tool_output import (
    ReviewJSONToolOutputUtil,
    ReviewTabularToolOutputUtil,
)


class BaseSilentReviewComponent(PostToolReflectionComponent):
    reviewer_cls: ClassVar[Type]

    def _get_review_args(self, data: SilentReviewRunInput) -> tuple:
        return (data.messages[0].content, data.tool_spec, data.tool_response)

    def _run(self, data: SilentReviewRunInput) -> SilentReviewRunOutput:  # type: ignore
        reviewer = self.reviewer_cls()
        llm = self.config.llm_client
        review = reviewer.process(llm, *self._get_review_args(data))
        return SilentReviewRunOutput(
            outcome=Outcome(review[0].value), details=review[1]
        )

    async def _arun(self, data: SilentReviewRunInput) -> SilentReviewRunOutput:  # type: ignore
        reviewer = self.reviewer_cls()
        llm = self.config.llm_client
        review = await reviewer.aprocess(llm, *self._get_review_args(data))
        return SilentReviewRunOutput(
            outcome=Outcome(review[0].value), details=review[1]
        )

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        return {AgentPhase.RUNTIME}


class SilentReviewForJSONDataComponent(BaseSilentReviewComponent):
    reviewer_cls: ClassVar[Type] = ReviewJSONToolOutputUtil


class SilentReviewForTabularDataComponent(BaseSilentReviewComponent):
    reviewer_cls: ClassVar[Type] = ReviewTabularToolOutputUtil

    def _get_review_args(self, data: SilentReviewRunInput) -> tuple:
        return (
            data.messages[0].content,
            data.tool_spec,
            None,
            None,
            data.tool_response,
        )
