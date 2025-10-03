from typing import Any, Dict
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import DebuggingResult
from nestful.data_handlers import get_nestful_catalog
from tests.utils.refraction.tools.custom_tools import (
    SkyScrapperFlightSearch,
    search_hotels,
)

import pytest

pytestmark = pytest.mark.refract_extra


class TestDecoration:
    def setup_method(self) -> None:
        catalog = get_nestful_catalog(executable=True)
        self.refractor = Refractor(catalog=catalog)
        self.memory: Dict[str, Any] = {}

    def test_success(self) -> None:
        flight_details = SkyScrapperFlightSearch(
            originSkyId="BOS",
            destinationSkyId="JFK",
            originEntityId="123",
            destinationEntityId="456",
            date="2024-05-09",
            # NOTE: Extra parameters
            refractor=self.refractor,
        )  # type: ignore[call-arg]

        assert flight_details.get("flightId") == 12345

    def test_failure_basic(self) -> None:
        flight_details = SkyScrapperFlightSearch(
            # missing parameter
            # originSkyId="BOS",
            destinationSkyId="JFK",
            originEntityId="123",
            destinationEntityId="456",
            date="2024-05-09",
            # NOTE: Extra parameters
            refractor=self.refractor,
        )  # type: ignore[call-arg]

        assert isinstance(flight_details, DebuggingResult)
        assert flight_details.report.determination is False

    def test_failure_with_reference_and_memory(self) -> None:
        memory = {"query": "London", "var1": {"geoId": "foo"}}

        payload = {
            "checkIn": "2024-09-05",
            "checkOut": "2024-09-15",
        }

        hotel_details: Dict[str, Any] = search_hotels(
            **payload,
            # NOTE: Extra parameters
            refractor=self.refractor,
            memory=memory,
        )

        assert hotel_details.get("id") == "hotel123"

    def test_failure_recovery_call(self) -> None:
        memory = {
            "query": "London",
        }

        payload = {
            "geoId": "$var1.geoId$",
            "checkIn": "2024-09-05",
            "checkOut": "2024-09-15",
        }

        hotel_details = search_hotels(
            refractor=self.refractor, memory=memory, param="foo", **payload
        )

        assert hotel_details.get("id") == "hotel123"
