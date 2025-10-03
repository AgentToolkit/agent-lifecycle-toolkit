from altk.pre_tool_reflection_toolkit.refraction.src import refract
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import Mapping
from nestful.data_handlers import get_nestful_catalog
from nl2flow.compile.options import BasicOperations
from typing import Dict, Any

import pytest

pytestmark = pytest.mark.refract_extra
# These tests take a while, going to set as extra


class TestCFC:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)

        self.mappings = [
            Mapping(
                source_name="skyId",
                target_name="originSkyId",
            ),
            Mapping(
                source_name="skyId",
                target_name="destinationSkyId",
            ),
            Mapping(
                source_name="entityId",
                target_name="originEntityId",
            ),
            Mapping(
                source_name="entityId",
                target_name="destinationEntityId",
            ),
        ]

    def test_cfc(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
            "var3": {"datetime": {"date": "2024-08-15"}},
        }

        sequence = [
            (
                'var4 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$")'
            ),
            'var5 = TripadvisorSearchLocation(query="London")',
            (
                'var6 = TripadvisorSearchHotels(checkIn="2024-08-15",'
                ' checkOut="2024-08-18")'
            ),
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            memory_objects=memory_items,
            mappings=self.mappings,
            use_given_operators_only=True,
        )

        cfc = result.corrected_function_call(memory=memory_items, catalog=self.catalog)

        reference_sequence = [
            (
                'var4 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$",'
                ' date="$var3.datetime.date$")'
            ),
            'var5 = TripadvisorSearchLocation(query="London")',
            (
                'var6 = TripadvisorSearchHotels(geoId="$var5.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        assert result.report.determination is False
        assert cfc is not None
        assert cfc.tokenized == reference_sequence
        assert cfc.is_executable

    def test_backing_steps(self) -> None:
        sequence = [
            'var6 = TripadvisorSearchHotels(checkIn="2024-08-15",'
            ' checkOut="2024-08-18")'
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            mappings=self.mappings,
        )

        cfc = result.corrected_function_call(memory={}, catalog=self.catalog)

        reference_sequence = [
            'var1 = TripadvisorSearchLocation(query="$query$")',
            (
                'var6 = TripadvisorSearchHotels(geoId="$var1.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        assert result.report.determination is False
        assert cfc is not None
        assert cfc.tokenized == reference_sequence
        assert not cfc.is_executable

        assert cfc.backing_steps[0].name == BasicOperations.SLOT_FILLER.value
        assert cfc.backing_steps[0].inputs == ["query"]

    def test_backing_steps_with_recovery_call(self) -> None:
        memory: Dict[str, Any] = {"var3": {"date": {}}, "query": "London"}
        sequence = [
            'var6 = TripadvisorSearchHotels(checkIn="$var6.date$",'
            ' checkOut="2024-08-18")'
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            mappings=self.mappings,
            memory_objects=memory,
        )

        cfc = result.corrected_function_call(memory={}, catalog=self.catalog)

        reference_sequence = [
            'var1 = TripadvisorSearchLocation(query="$query$")',
            (
                'var6 = TripadvisorSearchHotels(geoId="$var1.geoId$",'
                ' checkIn="$var6.date$", checkOut="2024-08-18")'
            ),
        ]

        assert result.report.determination is False
        assert cfc is not None
        assert cfc.tokenized == reference_sequence

        assert cfc.is_executable
        assert cfc.backing_steps == []
