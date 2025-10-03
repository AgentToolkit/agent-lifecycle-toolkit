from typing import Union, List, Set, Dict, Any
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import ResultTags, Mapping
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    DebuggingResult,
)
from nl2flow.debug.schemas import Report, DiffAction, SolutionQuality
from nl2flow.compile.schemas import Step, MemoryItem
from nl2flow.compile.flow import Flow
from nestful import Catalog, SequencingData


def pprint(ip: Union[str, List[str]]) -> None:
    if isinstance(ip, List):
        ip = "\n".join(ip)

    print(f"\n\n{ip}")


def compute_number_of_edits(diff_strings: List[str]) -> int:
    edits = [
        item
        for item in diff_strings
        if item.startswith(DiffAction.ADD.value)
        or item.startswith(DiffAction.DELETE.value)
    ]

    return len(edits)


def compute_tags(
    sequence: SequencingData,
    catalog: Catalog,
    diff: List[str],
    mappings: List[Mapping] | None = None,
) -> ResultTags:
    seq_x_param = len(sequence.output) * sum(
        [len(step.arguments.keys()) for step in sequence.output]
    )

    return ResultTags(
        length_of_sequence=len(sequence.output),
        length_of_sequence_x_parameters=seq_x_param,
        number_of_maps=len(mappings) if mappings else 0,
        number_of_apis=len(catalog.apis),
        number_of_edits=compute_number_of_edits(diff),
    )


def determine(
    planner_report: Report,
    result: DebuggingResult,
    catalog: Catalog,
    sequence: SequencingData,
    memory_objects: Dict[str, Any],
) -> bool:
    if planner_report.determination is not None:
        cfc = result.corrected_function_call(memory_objects, catalog)
        is_same_as: bool = sequence.is_same_as(
            cfc.objectified,
            catalog,
            required_schema_only=True,
            check_values=True,
        )

        if planner_report.report_type == SolutionQuality.OPTIMAL:
            is_same_as = is_same_as and len(sequence.output) == len(
                cfc.objectified.output
            )

        return is_same_as

    else:
        return False


def num_references(
    name: str,
    sequence: SequencingData | None = None,
    memory_steps: SequencingData | None = None,
    goals: List[Union[Step, MemoryItem]] | None = None,
) -> int:
    pooled_names = pool_mentioned_names(sequence, memory_steps, goals)
    pooled_names = [pn for pn in pooled_names if pn == name]

    return len(pooled_names)


def pool_mentioned_names(
    sequence: SequencingData | None = None,
    memory_steps: SequencingData | None = None,
    goals: List[Union[Step, MemoryItem]] | None = None,
) -> List[str]:
    pooled_names = []

    if goals:
        goals = [goal.name for goal in goals if isinstance(goal, Step)]
        pooled_names.extend(goals)

    if memory_steps:
        pooled_names.extend([item.name for item in memory_steps.output])

    if sequence:
        pooled_names.extend([item.name for item in sequence.output])

    return pooled_names


def filter_catalog(
    catalog: Catalog,
    sequence: SequencingData | None = None,
    memory_steps: SequencingData | None = None,
    goals: List[Union[Step, MemoryItem]] | None = None,
    flow: Flow | None = None,
) -> Catalog:
    mentioned_names = pool_mentioned_names(sequence, memory_steps, goals)

    if flow is not None:
        inputs_of_interest: Set[str] = set()

        for name in mentioned_names:
            api_spec = catalog.get_api(name)
            inputs = api_spec.get_arguments()

            inputs_of_interest = {*inputs_of_interest, *inputs}

        for mapping in flow.flow_definition.list_of_mappings:
            if mapping.target_name in inputs_of_interest:
                inputs_of_interest.add(mapping.source_name)

        for api in catalog.apis:
            outputs = api.get_outputs()

            for item in api.output_parameters:
                properties = api.output_parameters[item].properties
                keys = {f"{item}.{prop}" for prop in properties.keys()}
                outputs = {*outputs, *keys}

            if any([item in inputs_of_interest for item in outputs]):
                mentioned_names.append(api.name)

    return Catalog(apis=[api for api in catalog.apis if api.name in mentioned_names])
