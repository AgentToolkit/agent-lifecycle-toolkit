from nl2flow.plan.planners.kstar import Kstar
from nl2flow.plan.schemas import PlannerResponse
from nl2flow.compile.flow import Flow
from nl2flow.compile.schemas import MappingItem, MemoryItem, Step
from nl2flow.compile.options import (
    NL2FlowOptions,
    LifeCycleOptions,
    SlotOptions,
)

from nestful import Catalog
from typing import List, Dict, Union, Any
from altk.pre_tool_reflection_toolkit.refraction.src.printer import CustomPrint
from altk.pre_tool_reflection_toolkit.refraction.src.compiler import (
    add_goals,
    add_memory_objects,
    add_catalog_to_flow,
)

PRINTER = CustomPrint()
PLANNER = Kstar()


def diffract(
    catalog: Catalog,
    goals: List[Union[Step, MemoryItem]],
    memory_objects: Dict[str, Any] | None = None,
    memory_steps: List[Step] | None = None,
    mappings: List[MappingItem] | None = None,
    defensive: bool = False,
) -> PlannerResponse:
    flow = Flow(name="Diffraction", validate=False)

    if defensive:
        flow.variable_life_cycle.add(LifeCycleOptions.confirm_on_mapping)
        flow.variable_life_cycle.add(LifeCycleOptions.confirm_on_determination)

    flow.optimization_options.remove(NL2FlowOptions.multi_instance)
    flow.optimization_options.remove(NL2FlowOptions.allow_retries)

    flow.slot_options.add(SlotOptions.last_resort)

    add_catalog_to_flow(catalog, flow)

    if memory_objects:
        add_memory_objects(flow, memory_objects)

    if mappings:
        for mapping in mappings:
            flow.add(MemoryItem(item_id=mapping.source_name))
            flow.add(MemoryItem(item_id=mapping.target_name))
            flow.add(mapping)

    if memory_steps:
        flow.add(memory_steps)

    if goals:
        add_goals(flow, goals)

    planner_response = flow.plan_it(PLANNER)
    return planner_response
