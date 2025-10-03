from nl2flow.compile.flow import Flow
from nl2flow.debug.schemas import SolutionQuality, DebugFlag
from nl2flow.debug.debug import BasicDebugger
from nl2flow.compile.schemas import MemoryItem, Step, PartialOrder
from nl2flow.compile.options import (
    SlotOptions,
    NL2FlowOptions,
    LifeCycleOptions,
    BasicOperations,
)

from altk.pre_tool_reflection_toolkit.refraction.src.mappings.utils import filter_maps
from altk.pre_tool_reflection_toolkit.refraction.src.utils import (
    pprint,
    determine,
    filter_catalog,
    compute_tags,
)

from altk.pre_tool_reflection_toolkit.refraction.src.printer import CustomPrint
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.mappings import Mapping
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    DebuggingResult,
    BackingData,
)
from altk.pre_tool_reflection_toolkit.refraction.src.compiler import (
    EPISODE_DELIMITER,
    add_goals,
    add_memory_steps,
    add_memory_objects,
    add_mappings,
    add_sequence_object_to_flow,
    add_catalog_to_flow,
    add_episodes,
)

from nestful import Catalog, SequencingData, SequenceStep
from nestful.schemas.tools import Tool, OpenAITool, ToolCall, OpenAIToolCall
from multiprocessing import Process, Queue
from queue import Queue as TQueue
from typing import List, Dict, Optional, Union, Any
from pydantic import ValidationError
from time import time

PRINTER = CustomPrint()


def refract_sequence_object(
    sequence: SequencingData,
    catalog: Catalog,
    episodes: List[SequencingData],
    memory_objects: Dict[str, Any],
    memory_steps: SequencingData,
    mappings: List[Mapping] | None = None,
    defensive: bool = False,
    min_diff: bool = True,
    max_try: int = 3,
    compress: bool = False,
    goals: List[Union[Step, MemoryItem]] | None = None,
    partial_orders: List[PartialOrder] | None = None,
    use_given_operators_only: bool = False,
    use_cc: bool = False,
    allow_remaps: bool = False,
    report_type: Optional[SolutionQuality] = None,
) -> DebuggingResult:
    start_time = time()
    flow = Flow(name="Check Soundness", validate=False)

    if defensive:
        flow.variable_life_cycle.add(LifeCycleOptions.confirm_on_mapping)
        flow.variable_life_cycle.add(LifeCycleOptions.confirm_on_determination)

    flow.optimization_options.add(NL2FlowOptions.label_production)
    flow.optimization_options.remove(NL2FlowOptions.multi_instance)
    flow.slot_options.add(SlotOptions.last_resort)

    if compress:
        flow.optimization_options.remove(NL2FlowOptions.allow_retries)

    add_sequence_object_to_flow(sequence, flow, catalog)
    add_memory_steps(flow, catalog, memory_steps)

    new_catalog = (
        filter_catalog(
            catalog,
            sequence,
            memory_steps,
            goals,
            flow=flow if use_cc else None,
        )
        if use_given_operators_only
        else catalog
    )

    if episodes:
        add_episodes(flow, sequence, new_catalog, episodes)

    new_mappings = filter_maps(new_catalog, mappings) if mappings else []
    add_mappings(flow, new_mappings)

    add_catalog_to_flow(new_catalog, flow, sequence, memory_steps, goals, max_try)

    add_memory_objects(flow, memory_objects)

    if report_type == SolutionQuality.OPTIMAL and goals is None:
        goals = [Step(name=step.name) for step in sequence.output]

    if goals:
        add_goals(flow, goals)

    if partial_orders:
        flow.add(partial_orders)

    debugger = BasicDebugger(flow)
    tokens = sequence.pretty_print(
        mapper_tag=BasicOperations.MAPPER.value, collapse_maps=False
    )

    if report_type is None:
        report_type = (
            SolutionQuality.VALID if goals or episodes else SolutionQuality.SOUND
        )

    report = debugger.debug(
        tokens.split("\n") if tokens else [],
        report_type=report_type,
        debug_flag=DebugFlag.DIRECT,
        printer=PRINTER,
        sequence=sequence,
        catalog=new_catalog,
        memory=memory_objects,
        use_given_operators_only=use_given_operators_only,
        compress=compress,
        allow_remaps=allow_remaps,
    )

    diff_string: List[str] = []

    for plan in report.planner_response.list_of_plans:
        for step in plan.plan:
            step.name = step.name.split(EPISODE_DELIMITER)[0]

    if min_diff:
        for plan in report.planner_response.list_of_plans:
            minified_tokens = sequence.pretty_print(
                mapper_tag=BasicOperations.MAPPER.value,
                collapse_maps=True,
            )

            new_diff_string = BasicDebugger.generate_plan_diff(
                printer=PRINTER,
                plan=plan,
                list_of_tokens=minified_tokens.split("\n"),
                collapse_maps=True,
                sequence=sequence,
                catalog=new_catalog,
                memory=memory_objects,
            )

            if not diff_string or len(new_diff_string) < len(diff_string):
                diff_string = new_diff_string

    elif report.planner_response.list_of_plans:
        best_val_plan = report.planner_response.list_of_plans[0]
        minified_tokens = sequence.pretty_print(
            mapper_tag=BasicOperations.MAPPER.value,
            collapse_maps=True,
        )

        diff_string = BasicDebugger.generate_plan_diff(
            printer=PRINTER,
            plan=best_val_plan,
            list_of_tokens=(minified_tokens.split("\n") if minified_tokens else []),
            collapse_maps=True,
            sequence=sequence,
            catalog=new_catalog,
            memory=memory_objects,
        )

    end_time = time()
    elapsed_time = end_time - start_time

    result = DebuggingResult(
        diff=diff_string,
        report=report,
        time_taken=elapsed_time,
        tags=compute_tags(sequence, new_catalog, diff_string, new_mappings),
    )

    pprint(result.diff)

    result.backing_data = BackingData(data=sequence, catalog=new_catalog)
    result.report.determination = determine(
        report,
        result,
        catalog,
        sequence,
        memory_objects,
    )

    return result


def refract(
    sequence: Union[
        SequenceStep,
        SequencingData,
        List[str],
        List[Dict[str, Any]],
        Dict[str, Any],
    ],
    catalog: Union[Catalog, List[Dict[str, Any]]],
    episodes: List[SequencingData] | None = None,
    memory_objects: Optional[Dict[str, Any]] = None,
    memory_steps: Union[SequencingData, List[str], List[Dict[str, Any]]] | None = None,
    mappings: List[Mapping] | None = None,
    defensive: bool = False,
    min_diff: bool = False,
    max_try: int = 3,
    compress: bool = False,
    goals: List[Union[Step, MemoryItem]] | None = None,
    partial_orders: List[PartialOrder] | None = None,
    use_given_operators_only: bool = False,
    use_cc: bool = False,
    allow_remaps: bool = False,
    report_type: Optional[SolutionQuality] = None,
    timeout: Optional[float] = None,
    queue: Optional[TQueue[DebuggingResult]] = None,
) -> DebuggingResult:
    catalog = parse_catalog_input(catalog)
    sequence = parse_sequence_input(sequence)
    episodes = episodes or []

    memory_steps = parse_sequence_input(memory_steps)

    if not timeout:
        try:
            result = refract_sequence_object(
                sequence,
                catalog,
                episodes,
                memory_objects or {},
                memory_steps or SequencingData(),
                mappings,
                defensive,
                min_diff,
                max_try,
                compress,
                goals,
                partial_orders,
                use_given_operators_only,
                use_cc,
                allow_remaps,
                report_type,
            )

        except Exception as e:
            result = DebuggingResult(error=str(e))

        if queue:
            queue.put(result)

        return result

    else:
        q = Queue()  # type: ignore
        processes = [
            Process(
                target=refract,
                args=(
                    sequence,
                    catalog,
                    episodes,
                    memory_objects or {},
                    memory_steps or SequencingData(),
                    mappings,
                    defensive,
                    min_diff,
                    max_try,
                    compress,
                    goals,
                    partial_orders,
                    use_given_operators_only,
                    use_cc,
                    allow_remaps,
                    report_type,
                    None,
                    q,
                ),
            )
        ]

        processes[0].start()
        processes[0].join(timeout)

        try:
            result = q.get(timeout=timeout)
        except Exception as e:
            result = DebuggingResult(is_timeout=True, error=str(e))

        processes[0].kill()
        return result


def parse_catalog_input(
    catalog: Catalog | List[Dict[str, Any]],
) -> Catalog:
    if isinstance(catalog, List):
        new_catalog = Catalog()

        for item in catalog:
            try:
                new_tool = OpenAITool(**item)
            except ValidationError:
                try:
                    new_tool = Tool(**item)
                except Exception as e:
                    raise e

            new_catalog.apis.append(new_tool.convert_to_catalog_spec())

        catalog = new_catalog

    return catalog


def parse_sequence_input(
    sequence: Union[
        SequenceStep,
        SequencingData,
        List[str],
        List[Dict[str, Any]],
        Dict[str, Any],
    ],
) -> SequencingData:
    if isinstance(sequence, SequenceStep):
        sequence = SequencingData(output=[sequence])
    elif isinstance(sequence, Dict):
        sequence = SequencingData(output=[ToolCall(**sequence)])
    elif isinstance(sequence, List):
        if all([isinstance(step, str) for step in sequence]):
            sequence = SequencingData.parse_pretty_print(sequence)
        else:
            try:
                sequence = SequencingData(
                    output=OpenAIToolCall.initialize_from_list(tool_calls=sequence)
                )

            except (ValidationError, AttributeError):
                try:
                    sequence = SequencingData(
                        output=ToolCall.initialize_from_list(tool_calls=sequence)
                    )
                except Exception as e:
                    print(e)
                    pass

    return sequence
