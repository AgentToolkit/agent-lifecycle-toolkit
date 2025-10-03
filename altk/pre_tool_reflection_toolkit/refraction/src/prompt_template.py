from nestful import Catalog, SequencingData, SequenceStep
from nestful.utils import extract_label, get_token
from typing import Dict, List, Any, Optional, Tuple, Union
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    DebuggingResult,
    PromptType,
)
from altk.pre_tool_reflection_toolkit.refraction.src.main import (
    parse_sequence_input,
    parse_catalog_input,
)
from nl2flow.printers.verbalize import comma_separate


def generate_prompt(
    result: DebuggingResult | List[DebuggingResult],
    sequence: Union[
        SequenceStep,
        SequencingData,
        List[str],
        List[Dict[str, Any]],
        Dict[str, Any],
    ],
    catalog: Union[Catalog, List[Dict[str, Any]]],
    memory_objects: Dict[str, Any],
    prompt_type: PromptType = PromptType.NO_HELP,
) -> str:
    catalog = parse_catalog_input(catalog)
    sequence = parse_sequence_input(sequence)

    if isinstance(result, DebuggingResult):
        result = [result]

    prompt, is_multiple_suggestions = basic_prompt(sequence, result, prompt_type)

    if prompt_type == PromptType.WITH_FIX:
        prompt = (
            f"{prompt}\n\nThe following"
            f" {'are some' if is_multiple_suggestions else 'is a'} suggested"
            f" correction{'s' if is_multiple_suggestions else ''}. Use"
            f" {'these' if is_multiple_suggestions else 'this'} as a guidance"
            " to fix the above tool call."
        )

        for item in result:
            cfc = item.corrected_function_call(memory_objects, catalog)

            if cfc:
                prompt = f"{prompt}\n\n<tool_call>{cfc.objectified}</tool_call>"

    if (
        prompt_type == PromptType.SANITY_CHECK
        or prompt_type == PromptType.WITH_SUGGESTIONS
    ):
        list_of_fixes = []

        for item in result:
            new_fixes = stringify_list_of_fixes(
                sequence, item, catalog, memory_objects, prompt_type
            )

            list_of_fixes.append(new_fixes)

        if list_of_fixes:
            if prompt_type == PromptType.WITH_SUGGESTIONS:
                consolidated_string_of_fixes: List[str] = []

                for item in list_of_fixes:
                    if item:
                        tmp_string = "\n".join([f"- {fix}" for fix in item])
                        consolidated_string_of_fixes.append(tmp_string)

                tmp_string = "\n\nOR\n\n".join(consolidated_string_of_fixes)

            else:
                consolidated_list_of_fixes: List[str] = []

                for item in list_of_fixes:
                    if item:
                        consolidated_list_of_fixes.extend(item)

                deduplicated_list_of_fixes = []
                for item in consolidated_list_of_fixes:
                    if item not in deduplicated_list_of_fixes:
                        deduplicated_list_of_fixes.append(item)

                tmp_string = "\n".join(
                    [f"- {fix}" for fix in deduplicated_list_of_fixes]
                )

            if prompt_type == PromptType.WITH_SUGGESTIONS:
                tmp_string = (
                    "Each issue is accompanied by guidance on how to fix"
                    " it.\nConsider the guidance, along with the provided tool"
                    " specs, and memory, to come up with the final fixed tool"
                    f" call.\n\n{tmp_string}"
                )

            prompt = (
                f"{prompt}\n\nThe following are the identified issues:\n{tmp_string}"
            )

    return prompt


def basic_prompt(
    sequence: SequencingData,
    result: List[DebuggingResult],
    prompt_type: PromptType,
) -> Tuple[str, bool]:
    num_suggestions = len(
        [item for item in result if item.report.determination is not True]
    )

    is_multiple_suggestions = num_suggestions > 1

    if prompt_type == PromptType.NO_HELP or num_suggestions == 0:
        prompt = "Please fix the provided tool call."
    elif prompt_type == PromptType.WITH_FIX:
        prompt = (
            "Please fix the provided tool call based on the provided"
            f" suggestion{'s' if is_multiple_suggestions else ''}."
        )
    else:
        prompt = "Please fix the provided tool call based on the issues outlined."

    prompt = f"{prompt}\n\n<tool_call>{sequence}</tool_call>"

    return prompt, is_multiple_suggestions


def stringify_list_of_fixes(
    sequence: SequencingData | List[str],
    result: DebuggingResult,
    catalog: Catalog,
    memory_objects: Dict[str, Any],
    prompt_type: PromptType = PromptType.NO_HELP,
) -> List[str]:
    if not isinstance(sequence, SequencingData):
        sequence = SequencingData.parse_pretty_print(sequence)

    corrected_tool_call = result.corrected_function_call(
        memory=memory_objects, catalog=catalog
    )

    list_of_fixes = []
    cached_list_of_fixes = []

    if corrected_tool_call:
        for step in corrected_tool_call.objectified.output:
            wrong_step = get_reference_step(step, sequence)

            corrected_tool_call_object = corrected_tool_call.objectified
            corrected_tool_call_object.input = sequence.input

            if wrong_step is not None:
                new_fixes = stringify_list_of_fixes_per_step(
                    step,
                    corrected_tool_call_object,
                    wrong_step,
                    memory_objects,
                    prompt_type,
                )

                list_of_fixes.extend(new_fixes)

            else:
                # TODO: Not sure if this is a catch all for all recovery calls
                cached_list_of_fixes.append(
                    f"Possible fix: Call {step.name} with parameters:"
                    f" {comma_separate(list(step.arguments.keys()))}."
                )

                for parameter in step.arguments:
                    parameter_fix = add_parameter_string(
                        parameter,
                        step,
                        corrected_tool_call_object,
                        memory_objects,
                    )
                    cached_list_of_fixes.append(parameter_fix)

    list_of_fixes.extend(cached_list_of_fixes)
    return list_of_fixes


def stringify_list_of_fixes_per_step(
    corrected_step: SequenceStep,
    backing_sequence: SequencingData,
    wrong_step: Optional[SequenceStep],
    memory_objects: Dict[str, Any],
    prompt_type: PromptType = PromptType.NO_HELP,
) -> List[str]:
    list_of_fixes = []

    if wrong_step:
        for parameter in wrong_step.arguments:
            if parameter not in corrected_step.arguments:
                list_of_fixes.append(
                    f"Parameter {parameter} is not a recognized parameter for"
                    f" the tool {wrong_step.name}."
                )

    for parameter in corrected_step.arguments:
        if wrong_step is not None and parameter not in wrong_step.arguments:
            parameter_fix = (
                f"Parameter {parameter} is a required parameter for"
                f" {corrected_step.name}, but it is missing."
            )

            list_of_fixes.append(parameter_fix)

        if (
            prompt_type == PromptType.WITH_SUGGESTIONS
            and wrong_step.arguments.get(parameter, None)
            != corrected_step.arguments[parameter]
        ):
            if parameter in wrong_step.arguments:
                parameter_fix = f"The assignment to parameter {parameter} is wrong."

                list_of_fixes.append(parameter_fix)

            parameter_fix = add_parameter_string(
                parameter, corrected_step, backing_sequence, memory_objects
            )

            list_of_fixes.append(parameter_fix)

    return list_of_fixes


def add_parameter_string(
    parameter: str,
    corrected_step: SequenceStep,
    backing_sequence: SequencingData,
    memory_objects: Dict[str, Any],
) -> str:
    label, mapping = extract_label(corrected_step.arguments[parameter])

    if mapping is None:
        return (
            f"Possible fix: Parameter {parameter} should be assigned the value"
            f" {corrected_step.arguments[parameter]}."
        )
    else:
        if label == get_token(index=0):
            if backing_sequence.input:
                return (
                    f"Possible fix: Get value of {parameter} from the user"
                    f" input: {backing_sequence.input}"
                )
            else:
                return f"Possible fix: Get value of {parameter} from the context."
        else:
            if label in memory_objects:
                return (
                    f"Possible fix: Parameter {parameter} can be assigned to"
                    f" {mapping} in {label} from memory."
                )

            else:
                who, _ = backing_sequence.who_produced(label)
                return (
                    f"Possible fix: Get value of parameter {parameter} by"
                    f" calling {who}."
                )


def get_reference_step(
    reference_step: SequenceStep, sequence: SequencingData
) -> Optional[SequenceStep]:
    name, repeat_index = sequence.who_produced(reference_step.label)
    count = 0

    for step in sequence.output:
        if step.name == reference_step.name:
            count += 1

            if count == repeat_index:
                return step

    return None
