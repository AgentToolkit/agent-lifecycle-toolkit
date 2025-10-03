from nestful.data_handlers import get_nestful_data_instance, get_nestful_data
from altk.pre_tool_reflection_toolkit.refraction.src.mappings.compute_maps import Mapper
from altk.pre_tool_reflection_toolkit.refraction.src.mappings.utils import (
    cache_maps,
    merge_maps,
)
from altk.pre_tool_reflection_toolkit.refraction.src.utils import filter_catalog

import json
import pytest


class TestMappings:
    def setup_method(self) -> None:
        self.sequence_data, self.catalog = get_nestful_data(executable=True)
        self.mapper = Mapper()

    @pytest.mark.skip(reason="Takes too long")
    def test_computed_maps(self) -> None:
        sequence, catalog = get_nestful_data_instance(executable=True, index=0)
        filtered_catalog = filter_catalog(catalog, sequence=sequence)

        maps = self.mapper.compute_maps(filtered_catalog, top_k=1, threshold=0.80)

        model_dump = [item.model_dump() for item in maps]

        with open("cached_mappings.json", "w") as f:
            f.write(json.dumps(model_dump, indent=4))

        assert len(maps) == 7

    def test_cached_maps(self) -> None:
        maps = cache_maps(self.sequence_data)
        assert len(maps) == 111

    def test_merge_maps(self) -> None:
        maps = cache_maps(self.sequence_data)
        maps = merge_maps(maps)

        assert len(maps) == 20
