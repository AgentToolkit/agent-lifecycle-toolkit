from __future__ import annotations
from pydantic import BaseModel, computed_field
from typing import List, Optional, Dict, Any
from enum import Enum, auto
from nl2flow.debug.schemas import Report
from nestful import SequencingData, SequenceStep, Catalog
from nl2flow.debug.schemas import DiffAction
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.utils import open_mean
from nl2flow.plan.schemas import ClassicalPlan as Plan, Action
from nl2flow.compile.options import BasicOperations
from altk.pre_tool_reflection_toolkit.refraction.src.printer import CustomPrint

TIMEOUT = 5.0
PRINTER = CustomPrint()
DUMMY_VALUE = "INIT"


class PromptType(str, Enum):
    NO_HELP = auto()
    WITH_FIX = auto()
    SANITY_CHECK = auto()
    WITH_SUGGESTIONS = auto()


class OperationModes(str, Enum):
    DEFAULT = auto()
    COMPRESS = auto()
    N0_MAPPINGS = auto()
    NO_MEMORY = auto()
    SEQUENCE_ONLY = auto()


class ResultTags(BaseModel):
    length_of_sequence: int
    length_of_sequence_x_parameters: int
    number_of_maps: int
    number_of_apis: int
    number_of_edits: int


class BackingData(BaseModel):
    index: int = 0
    data: SequencingData
    catalog: Catalog


class CFC(BaseModel):
    tokenized: List[str]
    objectified: SequencingData
    backing_steps: List[Action] = []

    @computed_field  # type: ignore
    @property
    def is_executable(self) -> bool:
        return len(self.backing_steps) == 0


class DebuggingResult(BaseModel):
    diff: List[str] = []
    report: Report = Report()
    time_taken: float = float("inf")
    is_timeout: bool = False
    tags: Optional[ResultTags] = None
    error: Optional[str] = None
    backing_data: Optional[BackingData] = None

    def corrected_function_call(
        self,
        memory: Dict[str, Any],
        catalog: Catalog,
        plan: Optional[Plan] = None,
    ) -> Optional[CFC]:
        def _skip_basic(name: str) -> bool:
            return name.startswith(
                BasicOperations.SLOT_FILLER.value
            ) or name.startswith(BasicOperations.CONFIRM.value)

        best_plan = plan or self.report.planner_response.best_plan

        if best_plan:
            sequence_object = SequencingData()
            tokenized_print = []

            if self.backing_data:
                sequence_object.input = self.backing_data.data.input

            printout = PRINTER.pretty_print_plan(
                best_plan,
                memory=memory,
                catalog=catalog,
                collapse_maps=True,
            ).split("\n")

            backing_steps = [
                action for action in best_plan.plan if _skip_basic(name=action.name)
            ]

            for item in printout:
                if _skip_basic(name=item):
                    continue

                sequence_step = SequenceStep.parse_pretty_print(item)

                sequence_object.output.append(sequence_step)
                tokenized_print.append(item)

            return CFC(
                tokenized=tokenized_print,
                objectified=sequence_object,
                backing_steps=backing_steps,
            )
        else:
            return None

    @computed_field  # type: ignore
    @property
    def compression_rate(self) -> Optional[float]:
        if not self.diff:
            return None

        negative_diffs = [
            item for item in self.diff if item.startswith(DiffAction.DELETE.value)
        ]

        return len(negative_diffs) / len(self.diff)


class BatchResults(BaseModel):
    results: List[DebuggingResult]

    @computed_field  # type: ignore
    @property
    def time_taken(self) -> float:
        return float(open_mean(self.results, key="time_taken"))

    @computed_field  # type: ignore
    @property
    def mean_compression(self) -> float:
        return float(open_mean(self.results, key="compression_rate"))

    def generate_time_taken_distribution(
        self, result_tag: Optional[str] = None
    ) -> List[Dict[str, float]]:
        if result_tag:
            assert result_tag in Report.model_fields
            cache = dict()

            for r in self.results:
                if r.tags:
                    key = getattr(r.tags, result_tag)

                    if key not in cache:
                        cache[key] = [r]

                    else:
                        cache[key].append(r)

            return [
                {
                    "time_taken": open_mean(cache[key], key="time_taken"),
                    "x": key,
                }
                for key in cache
            ]

        else:
            return [
                {"time_taken": r.time_taken, "x": i + 1}
                for i, r in enumerate(self.results)
            ]

    def how_many_succeeded(
        self, result_tag: Optional[str] = None
    ) -> List[Dict[str, int]] | int:
        if result_tag:
            assert result_tag in Report.model_fields
            cache: Dict[str, int] = dict()

            for r in self.results:
                if r.tags:
                    key = str(getattr(r.tags, result_tag))

                    if key not in cache:
                        cache[key] += int(r.report.determination is True)

                    else:
                        cache[key] = int(r.report.determination is True)

            return [{result_tag: cache[key], "x": int(key)} for key in cache]

        else:
            filter_by_success = list(
                filter(
                    lambda x: x.report.determination is True,
                    self.results,
                )
            )

            return len(filter_by_success)
