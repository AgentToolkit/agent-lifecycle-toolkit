from nl2flow.compile.flow import Flow
from nl2flow.compile.operators import ClassicalOperator as Operator
from nl2flow.compile.options import TypeOptions, BasicOperations, GoalOptions
from nl2flow.compile.schemas import (
    SignatureItem,
    Parameter,
    MemoryItem,
    MemoryState,
    Step,
    GoalItems,
    GoalItem,
    GoalType,
    SlotProperty,
    MappingItem,
)

from nestful import Catalog, API, SequencingData, SequenceStep
from nestful.schemas.api import QueryParameter
from nestful.utils import get_token, extract_label
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import Mapping
from altk.pre_tool_reflection_toolkit.refraction.src.utils import num_references
from altk.pre_tool_reflection_toolkit.refraction.src.printer import CustomPrint
from typing import List, Dict, Union, Any

PRINTER = CustomPrint()
EPISODE_DELIMITER = "__ep__"


def add_api_to_flow(
    api: API,
    flow: Flow,
    sequence: SequencingData | None = None,
    memory_steps: SequencingData | None = None,
    goals: List[Union[Step, MemoryItem]] | None = None,
    max_try: int = 2,
    use_optional_parameters: bool = True,
) -> Flow:
    new_agent = Operator(name=api.name)

    num_appearances = num_references(api.name, sequence, memory_steps, goals)
    new_agent.max_try = min(max_try, num_appearances or 1)

    required_inputs = api.get_arguments(required=True)
    new_agent.add_input(SignatureItem(parameters=required_inputs))

    if use_optional_parameters:
        optional_inputs = api.get_arguments(required=False)
        new_agent.add_input(
            SignatureItem(
                parameters=[
                    Parameter(item_id=item, required=False) for item in optional_inputs
                ]
            )
        )

    outputs = api.get_outputs()
    new_agent.add_output(SignatureItem(parameters=outputs))

    flow.add(new_agent)
    return flow


def add_catalog_to_flow(
    catalog: Catalog,
    flow: Flow,
    sequence: SequencingData | None = None,
    memory_steps: SequencingData | None = None,
    goals: List[Union[Step, MemoryItem]] | None = None,
    max_try: int = 2,
    use_optional_parameters: bool = True,
) -> None:
    for api in catalog.apis:
        add_api_to_flow(
            api,
            flow,
            sequence,
            memory_steps,
            goals,
            max_try,
            use_optional_parameters,
        )


def add_sequence_step_to_flow(
    sequence_step: SequenceStep,
    flow: Flow,
    catalog: Catalog,
) -> None:
    api_definition = (
        catalog.get_api(name=sequence_step.name, minified=True)
        if sequence_step.name
        else None
    )

    if api_definition is not None:
        if sequence_step.label:
            for output in api_definition.outputs:
                flow.add(MemoryItem(item_id=output))

        for arg in api_definition.inputs:
            flow.add(MemoryItem(item_id=arg))

    for arg in sequence_step.arguments:
        source = str(sequence_step.arguments[arg])
        label, mapping = extract_label(source)

        if label and mapping:
            source = mapping

            flow.add(MemoryItem(item_id=arg))
            flow.add(MemoryItem(item_id=source))
            if label:
                flow.add(
                    MemoryItem(
                        item_id=label,
                        item_type=TypeOptions.LABEL.value,
                    )
                )
        else:
            source = f'"{source}"'
            flow.add(MemoryItem(item_id=arg))
            flow.add(MemoryItem(item_id=source, item_state=MemoryState.KNOWN))

        flow.add(SlotProperty(slot_name=source, slot_desirability=0.0))
        flow.add(MappingItem(source_name=source, target_name=arg, probability=1.0))


def add_sequence_object_to_flow(
    sequence: SequencingData, flow: Flow, catalog: Catalog
) -> None:
    for output in sequence.output:
        add_sequence_step_to_flow(output, flow, catalog)


def add_mappings(flow: Flow, mappings: List[Mapping]) -> None:
    for mapping in mappings:
        flow.add(MemoryItem(item_id=mapping.source_name))
        flow.add(MemoryItem(item_id=mapping.target_name))

    flow.add(mappings)


def add_goals(flow: Flow, goals: List[Union[Step, MemoryItem]]) -> None:
    for goal in goals:
        if isinstance(goal, Step):
            step = Step(name=goal.name, parameters=goal.parameters)

            for index, declared_map in enumerate(goal.maps):
                label, mapping = extract_label(declared_map)
                transformed_map = mapping if label and mapping else declared_map

                step.maps.append(transformed_map)
                flow.add(
                    MemoryItem(item_id=transformed_map, item_state=MemoryState.KNOWN)
                )

                flow.add(
                    MappingItem(
                        source_name=transformed_map,
                        target_name=step.parameters[index],
                    )
                )

            flow.add(
                GoalItems(
                    goals=GoalItem(
                        goal_name=step, delegate_maps=len(step.parameters) > 0
                    )
                )
            )

        elif isinstance(goal, MemoryItem):
            flow.add(
                GoalItems(
                    goals=GoalItem(
                        goal_name=goal.item_id, goal_type=GoalType.OBJECT_KNOWN
                    )
                )
            )
            flow.add(SlotProperty(slot_name=goal.item_id, slot_desirability=0.0))
            flow.add(MemoryItem(item_id=goal.item_id))


def add_memory_steps(flow: Flow, catalog: Catalog, memory: SequencingData) -> None:
    tokens = memory.pretty_print(
        mapper_tag=BasicOperations.MAPPER.value, collapse_maps=False
    )

    if tokens:
        steps = PRINTER.parse_tokens(tokens.split("\n"))

        add_sequence_object_to_flow(memory, flow, catalog)
        flow.add(steps.plan)


def add_nested_memory(
    flow: Flow, memory: Dict[str, Any], label: str, prefix: str = ""
) -> None:
    # This ugly situation will be resolved with direct output assignments
    # TODO: ISS24
    if memory:
        for nested_item in memory:
            item_id = f"{prefix}.{nested_item}" if prefix else nested_item
            flow.add(
                [
                    MemoryItem(
                        item_id=item_id,
                        item_state=MemoryState.KNOWN,
                        label=label,
                    ),
                    MemoryItem(
                        item_id=nested_item,
                    ),
                    MappingItem(
                        source_name=item_id,
                        target_name=nested_item,
                        probability=1.0,
                    ),
                ]
            )

            if isinstance(memory[nested_item], Dict):
                new_prefix = f"{prefix}.{nested_item}" if prefix else nested_item
                new_memory = memory[nested_item]

                add_nested_memory(flow, new_memory, label, new_prefix)
    else:
        dummy_variable = f"dummy_item_for_{label}"
        flow.add(
            [
                MemoryItem(
                    item_id=dummy_variable,
                ),
                MemoryItem(
                    item_id=dummy_variable,
                    item_state=MemoryState.KNOWN,
                    label=label,
                ),
            ]
        )


def add_memory_objects(flow: Flow, memory: Dict[str, Any]) -> None:
    for item in memory:
        # Sort mismatch with generics and labels
        # TODO: ISS24
        if item.startswith("var"):
            if isinstance(memory[item], Dict):
                add_nested_memory(flow, memory[item], item)
        else:
            label = get_token(index=0, token="var")
            flow.add(
                MemoryItem(
                    item_id=item,
                    item_state=MemoryState.KNOWN,
                    label=label,
                )
            )

            if isinstance(memory[item], Dict):
                add_nested_memory(flow, memory[item], label, item)


def add_episodes(
    flow: Flow,
    sequence: SequencingData,
    catalog: Catalog,
    episodes: List[SequencingData],
) -> None:
    flow.goal_type = GoalOptions.AND_OR

    for step in sequence.output:
        or_goals = set()

        for _id, episode in enumerate(episodes):
            reference_steps = [
                episode_step
                for episode_step in episode.output
                if episode_step.name == step.name
            ]

            for _inner_id, reference_step in enumerate(reference_steps):
                new_operation = (
                    f"{reference_step.name}{EPISODE_DELIMITER}{_id}_{_inner_id}"
                )
                new_api = API(
                    name=new_operation,
                    id=reference_step.name,
                    description=f"Derived from call to {reference_step.name}",
                )

                or_goals.add(new_operation)

                for item in reference_step.arguments:
                    new_api.query_parameters[item] = QueryParameter(required=True)

                    assignment = reference_step.arguments[item]
                    label, mapping = extract_label(assignment)

                    if label != get_token(index=0) and mapping is not None:
                        flow.add(
                            MappingItem(
                                source_name=mapping,
                                target_name=item,
                                probability=1.0,
                            )
                        )

                catalog.apis.append(new_api)

        flow.add(GoalItems(goals=[GoalItem(goal_name=item) for item in or_goals]))
