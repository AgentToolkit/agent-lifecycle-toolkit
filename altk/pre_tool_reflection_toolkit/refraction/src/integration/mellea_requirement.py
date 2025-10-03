from mellea.backends import Backend, BaseModelSubclass
from mellea.stdlib.requirement import Requirement, ValidationResult
from mellea.stdlib.base import Context, GenerateLog
from mellea.stdlib.instruction import Instruction
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
from altk.pre_tool_reflection_toolkit.refraction.src.prompt_template import (
    generate_prompt,
    PromptType,
)
from altk.pre_tool_reflection_toolkit.refraction.src.integration.utils import (
    extract_tool_calls,
)
from typing import Any, List, Tuple


def refract(context: Context, refractor_class: Refractor) -> ValidationResult:
    # TODO: We would like to extract the tool specs from the prompt here
    # TODO: Access to the prompt also allows us to extract other stuff like memory
    generated_text = context.last_output().value
    tool_calls = extract_tool_calls(generated_text)

    refraction_result = refractor_class.refract(tool_calls, memory_objects={})

    # TODO: No way to use the corrected call directly :(
    # cfc = refraction_result.corrected_function_call(catalog=refractor_class.catalog, memory={})

    reason = (
        generate_prompt(
            refraction_result,
            tool_calls,
            catalog=refractor_class.catalog,
            memory_objects={},
            prompt_type=PromptType.WITH_SUGGESTIONS,
        )
        if refraction_result.report.determination is not True
        else ""
    )

    return ValidationResult(
        result=refraction_result.report.determination is True, reason=reason
    )


def refract_repair(
    instruction: Instruction,
    constraint_scores: List[Tuple[Requirement, ValidationResult]],
    failed_instructions: List[Instruction],
) -> Instruction:
    refraction_constraint = next(
        filter(lambda x: isinstance(x[0], RefractionRequirement), constraint_scores),
        None,
    )

    if refraction_constraint is None:
        return instruction
    else:
        validation_result = refraction_constraint[1]
        instruction.description = validation_result.reason

    return instruction


class RefractionRequirement(Requirement, Refractor):
    def __init__(self, **kwargs: Any) -> None:
        Refractor.__init__(self, kwargs.get("tools", []))
        Requirement.__init__(
            self,
            description="LLM-free validation for tool calling",
            validation_fn=kwargs.get("validation_fn", refract),  # type: ignore
            check_only=True,
        )

    def validate(
        self,
        backend: Backend,
        ctx: Context,
        *,
        format: type[BaseModelSubclass] | None = None,
        model_options: dict[str, Any] | None = None,
        generate_logs: list[GenerateLog] | None = None,
    ) -> ValidationResult:
        return self.validation_fn(ctx, refractor_class=self)  # type: ignore
