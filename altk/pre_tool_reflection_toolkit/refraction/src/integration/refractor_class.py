from nestful import Catalog, SequencingData, SequenceStep
from nl2flow.debug.schemas import SolutionQuality
from nl2flow.compile.schemas import MemoryItem, Step, PartialOrder
from altk.pre_tool_reflection_toolkit.refraction.src import refract, compress
from altk.pre_tool_reflection_toolkit.refraction.src.main import (
    parse_catalog_input,
    parse_sequence_input,
)
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import (
    Mapping,
    DebuggingResult,
)
from altk.pre_tool_reflection_toolkit.refraction.src.mappings.compute_maps import Mapper
from altk.pre_tool_reflection_toolkit.refraction.src.mappings.utils import (
    cache_maps,
    merge_maps,
)
from typing import List, Optional, Dict, Union, Any


class Refractor:
    def __init__(
        self,
        catalog: Catalog | List[Dict[str, Any]],
        sequence_data: List[SequencingData] | List[List[Dict[str, Any]]] | None = None,
    ) -> None:
        self._catalog = (
            catalog if isinstance(catalog, Catalog) else parse_catalog_input(catalog)
        )

        self._sequences: List[SequencingData] = (
            [parse_sequence_input(item) for item in sequence_data]
            if sequence_data
            else []
        )

        self._mappings: List[Mapping] = []

    @property
    def catalog(self) -> Catalog:
        return self._catalog

    @catalog.setter
    def catalog(self, catalog: Catalog | List[Dict[str, Any]]) -> None:
        self._catalog = (
            catalog if isinstance(catalog, Catalog) else parse_catalog_input(catalog)
        )

    @property
    def sequences(self) -> List[SequencingData]:
        return self._sequences

    @property
    def mappings(self) -> List[Mapping]:
        return self._mappings

    @mappings.setter
    def mappings(self, mappings: List[Mapping]) -> None:
        self._mappings = mappings

    def initialize_maps(
        self, mapping_threshold: float = 0.80, top_k: int = 1
    ) -> List[Mapping]:
        print("Initializing maps, this will take a few minutes...")
        mapper_object = Mapper()

        computed_maps = mapper_object.compute_maps(
            self.catalog,
            top_k=top_k,
            threshold=mapping_threshold,
        )

        cached_maps = cache_maps(self.sequences)

        computed_maps.extend(cached_maps)
        self.mappings = merge_maps(computed_maps)

        return self.mappings

    def refract(
        self,
        sequence: Union[
            SequenceStep,
            SequencingData,
            List[str],
            List[Dict[str, Any]],
            Dict[str, Any],
        ],
        catalog: Catalog | List[Dict[str, Any]] | None = None,
        episodes: List[SequencingData] | None = None,
        memory_objects: Optional[Dict[str, Any]] = None,
        memory_steps: SequencingData | List[str] | None = None,
        mappings: List[Mapping] | None = None,
        defensive: bool = False,
        min_diff: bool = False,
        max_try: int = 3,
        compression: bool = False,
        goals: List[Union[Step, MemoryItem]] | None = None,
        partial_orders: List[PartialOrder] | None = None,
        use_given_operators_only: bool = False,
        use_cc: bool = False,
        allow_remaps: bool = False,
        report_type: Optional[SolutionQuality] = None,
        timeout: Optional[float] = None,
    ) -> DebuggingResult:
        return refract(
            sequence=sequence,
            catalog=catalog or self.catalog,
            episodes=episodes,
            memory_objects=memory_objects,
            memory_steps=memory_steps,
            mappings=mappings or self.mappings,
            defensive=defensive,
            min_diff=min_diff,
            max_try=max_try,
            compress=compression,
            goals=goals,
            partial_orders=partial_orders,
            use_given_operators_only=use_given_operators_only,
            use_cc=use_cc,
            allow_remaps=allow_remaps,
            report_type=report_type,
            timeout=timeout,
        )

    def compress(
        self,
        sequence: Union[
            SequenceStep,
            SequencingData,
            List[str],
            List[Dict[str, Any]],
            Dict[str, Any],
        ],
        catalog: Catalog | List[Dict[str, Any]] | None = None,
        memory_objects: Optional[Dict[str, Any]] = None,
        memory_steps: SequencingData | List[str] | None = None,
        mappings: List[Mapping] | None = None,
        min_diff: bool = False,
        max_try: int = 3,
        timeout: Optional[float] = None,
    ) -> DebuggingResult:
        return compress(
            sequence=sequence,
            catalog=catalog or self.catalog,
            memory_objects=memory_objects,
            memory_steps=memory_steps,
            mappings=mappings or self.mappings,
            min_diff=min_diff,
            max_try=max_try,
            timeout=timeout,
        )
