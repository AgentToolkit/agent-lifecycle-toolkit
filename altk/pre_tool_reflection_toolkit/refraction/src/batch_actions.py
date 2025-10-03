from typing import List, Tuple, Optional, Dict, Any, Union
from nestful import SequencingDataset, SequencingData, SequenceStep, Catalog
from nestful.utils import extract_label, get_token
from nl2flow.compile.schemas import Step, MemoryItem
from nl2flow.debug.schemas import SolutionQuality
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import (
    BatchResults,
    Mapping,
    DebuggingResult,
)
from altk.pre_tool_reflection_toolkit.refraction.src import refract, compress
from altk.pre_tool_reflection_toolkit.refraction.src.utils import pprint
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    OperationModes,
)


def run_all_modes(
    sequence: Union[
        SequenceStep,
        SequencingData,
        List[str],
        List[Dict[str, Any]],
        Dict[str, Any],
    ],
    catalog: Union[Catalog, List[Dict[str, Any]]],
    memory_objects: Optional[Dict[str, Any]] = None,
    memory_steps: SequencingData | List[str] | None = None,
    mappings: List[Mapping] | None = None,
    max_try: int = 3,
    goals: List[Union[Step, MemoryItem]] | None = None,
    timeout: Optional[float] = None,
) -> Dict[OperationModes, DebuggingResult]:
    return {
        OperationModes.DEFAULT: refract(
            sequence,
            catalog,
            memory_objects=memory_objects,
            memory_steps=memory_steps,
            mappings=mappings,
            max_try=max_try,
            goals=goals,
            timeout=timeout,
        ),
        OperationModes.COMPRESS: compress(
            sequence,
            catalog,
            memory_objects=memory_objects,
            memory_steps=memory_steps,
            mappings=mappings,
            max_try=max_try,
            timeout=timeout,
        ),
        OperationModes.N0_MAPPINGS: refract(
            sequence,
            catalog,
            memory_objects=memory_objects,
            memory_steps=memory_steps,
            max_try=max_try,
            goals=goals,
            timeout=timeout,
        ),
        OperationModes.NO_MEMORY: refract(
            sequence,
            catalog,
            max_try=max_try,
            goals=goals,
            timeout=timeout,
        ),
        OperationModes.SEQUENCE_ONLY: refract(
            sequence,
            catalog,
            memory_objects=memory_objects,
            memory_steps=memory_steps,
            max_try=max_try,
            goals=goals,
            use_given_operators_only=True,
            timeout=timeout,
        ),
    }


def run_all_batch(
    batch_data: List[Tuple[SequencingData, Union[SequencingData, None], Catalog]],
    report_type: SolutionQuality = SolutionQuality.SOUND,
    mappings: List[Mapping] | None = None,
    compression: bool = False,
    use_given_operators_only: bool = False,
    use_cc: bool = False,
    allow_remaps: bool = False,
    run_step_by_step: bool = False,
    timeout: Optional[float] = None,
) -> BatchResults:
    results = []

    for index, data in enumerate(batch_data):
        pprint(f"Validating sequence {index + 1}/{len(batch_data)}")

        new_sample = data[0]
        gt_sample = data[1]
        catalog = data[2]

        goals = []

        if isinstance(gt_sample, SequencingData):
            if gt_sample.var_result:
                for _k, v in gt_sample.var_result.items():
                    label, mapping = extract_label(v)
                    new_label = (
                        mapping
                        if label == get_token(index=0) or label is None
                        else label
                    )

                    name, _ = gt_sample.who_produced(new_label)

                    if name is not None:
                        goals.append(Step(name=name))

        if run_step_by_step:
            for step_index, step in enumerate(new_sample.output):
                memory = new_sample.get_memory(catalog, index=step_index)

                new_result = refract(
                    sequence=step,
                    catalog=catalog,
                    mappings=mappings,
                    memory_objects=memory,
                    compress=compression,
                    goals=goals or None,
                    use_given_operators_only=use_given_operators_only,
                    use_cc=use_cc,
                    allow_remaps=allow_remaps,
                    report_type=report_type,
                    timeout=timeout,
                )

                results.append(new_result)

        else:
            new_result = refract(
                sequence=new_sample,
                catalog=catalog,
                mappings=mappings,
                compress=compression,
                goals=goals or None,
                use_given_operators_only=use_given_operators_only,
                use_cc=use_cc,
                allow_remaps=allow_remaps,
                report_type=report_type,
                timeout=timeout,
            )

            results.append(new_result)

    return BatchResults(results=results)


def run_all(
    sequence_data: SequencingDataset,
    catalog: Catalog,
    report_type: SolutionQuality = SolutionQuality.SOUND,
    mappings: List[Mapping] | None = None,
    compression: bool = False,
    use_given_operators_only: bool = False,
    use_cc: bool = False,
    allow_remaps: bool = False,
    run_step_by_step: bool = False,
    timeout: Optional[float] = None,
) -> BatchResults:
    batch = [(data, None, catalog) for data in sequence_data.data]
    return run_all_batch(
        batch,
        report_type,
        mappings,
        compression,
        use_given_operators_only,
        use_cc,
        allow_remaps,
        run_step_by_step,
        timeout,
    )
