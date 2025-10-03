from altk.pre_tool_reflection_toolkit.refraction.src.main import refract
from altk.pre_tool_reflection_toolkit.refraction.src.printer import CustomPrint
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.mappings import Mapping
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    DebuggingResult,
)
from nestful import Catalog, SequencingData, SequenceStep
from typing import List, Dict, Optional, Any, Union

PRINTER = CustomPrint()


def compress(
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
    min_diff: bool = False,
    max_try: int = 3,
    timeout: Optional[float] = None,
) -> DebuggingResult:
    result: DebuggingResult = refract(
        sequence=sequence,
        catalog=catalog,
        memory_objects=memory_objects,
        memory_steps=memory_steps,
        mappings=mappings,
        min_diff=min_diff,
        compress=True,
        max_try=max_try,
        use_given_operators_only=True,
        timeout=timeout,
    )

    return result
