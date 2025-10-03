from altk.pre_tool_reflection_toolkit.refraction.src import refract
from nl2flow.compile.schemas import MemoryItem
from nestful.data_handlers import get_nestful_catalog
from tests.utils.refraction.utils import get_cached_maps

import pytest

pytestmark = pytest.mark.refract_extra
# These tests take a while, going to set as extra


class TestObjectGoal:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)
        self.cached_mappings = get_cached_maps()

    def test_object_goal_false(self) -> None:
        result = refract(
            sequence=[
                {
                    "name": "SkyScrapperSearchAirport",
                    "arguments": {"query": "New York"},
                },
                {
                    "name": "SkyScrapperSearchAirport",
                    "arguments": {"query": "San Juan"},
                },
            ],
            catalog=self.catalog,
            mappings=self.cached_mappings,
            goals=[MemoryItem(item_id="flightId")],
        )

        assert result.report.determination is False
        assert (
            result.report.planner_response.best_plan.plan[-1].name
            == "SkyScrapperFlightSearch"
        )

    def test_object_goal_true(self) -> None:
        result = refract(
            sequence=[
                {
                    "name": "SkyScrapperSearchAirport",
                    "arguments": {"query": "New York"},
                },
                {
                    "name": "SkyScrapperSearchAirport",
                    "arguments": {"query": "San Juan"},
                },
                {
                    "name": "SkyScrapperFlightSearch",
                    "arguments": {
                        "originSkyId": "...",
                        "destinationSkyId": "...",
                        "originEntityId": "...",
                        "destinationEntityId": "...",
                        "date": "...",
                    },
                },
            ],
            catalog=self.catalog,
            mappings=self.cached_mappings,
            goals=[MemoryItem(item_id="flightId")],
        )

        assert result.report.determination
