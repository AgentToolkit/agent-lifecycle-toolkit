from altk.pre_tool_reflection_toolkit.refraction.src import refract
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import Mapping
from nestful.data_handlers import get_nestful_catalog
from nl2flow.compile.options import BasicOperations

import pytest

pytestmark = pytest.mark.refract_extra
# These tests take a while, going to set as extra


class TestMemory:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)

    def test_everything_is_fine(self) -> None:
        memory_items = {
            "date_from_user": "2024-08-15",
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        sequence = [
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$",'
                ' date="$date_from_user$")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(sequence, catalog=self.catalog, memory_objects=memory_items)

        assert result.report.determination

    def test_everything_is_fine_direct_map(self) -> None:
        memory_items = {
            "date": "2024-08-15",
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
            "var3": {},
            "var4": {
                "geoId": "2572131",
            },
        }

        sequence = [
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(sequence, catalog=self.catalog, memory_objects=memory_items)

        assert result.report.determination

    def test_everything_is_fine_nested(self) -> None:
        memory_items = {
            "var1": {
                "location": {
                    "name": "Kolkata",
                    "region": "West Bengal",
                    "country": "India",
                    "lat": "22.5744N",
                    "lon": "88.3629E",
                    "tz_id": "IST",
                    "localtime_epoch": 1731625064,
                    "localtime": "Thursday, November 14, 2024 10:57:44 PM",
                }
            },
            "test_1": {"key": "value"},
            "test_2": {"key": {"nested:": "value"}},
        }

        sequence = [
            'var2 = WeatherAPI.com_Forecast_Weather_API(q="Paris")',
            ('var3 = WeatherAPI.com_Realtime_Weather_Api(q="$var1.location.name$")'),
        ]

        result = refract(sequence, catalog=self.catalog, memory_objects=memory_items)

        assert result.report.determination

    def test_everything_is_not_fine(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        sequence = [
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$",'
                ' date="$date$")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            memory_objects=memory_items,
            defensive=True,
        )

        num_confirms = 0

        for step in result.report.planner_response.best_plan.plan:
            if step.name.startswith(BasicOperations.CONFIRM.value):
                num_confirms += 1
                assert step.inputs[0] in ["date", "query", "originEntityId"]

        assert num_confirms == 3

    def test_memory_missing_parameter(self) -> None:
        memory_items = {
            "date_from_user": "2024-08-15",
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        mappings = [
            Mapping(
                source_name="skyId",
                target_name="originSkyId",
            ),
            Mapping(
                source_name="entityId",
                target_name="originEntityId",
            ),
        ]

        sequence = [
            'var3 = SkyScrapperFlightSearch(destinationSkyId="$var2.skyId$",'
            ' destinationEntityId="$var2.entityId$", date="$date_from_user$")'
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            mappings=mappings,
            memory_objects=memory_items,
        )

        assert result.report.determination is False
        assert (
            result.diff[1]
            == '+ var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
            ' destinationSkyId="$var2.skyId$",'
            ' originEntityId="$var1.entityId$",'
            ' destinationEntityId="$var2.entityId$",'
            ' date="$date_from_user$")'
        )

    def test_memory_missing_parameter_same_name(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
            "var3": {"date": "2024-08-15"},
        }

        sequence = [
            'var4 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
            ' destinationSkyId="$var2.skyId$",'
            ' originEntityId="$var1.entityId$",'
            ' destinationEntityId="$var2.entityId$",'
            ' date="$var3.date$")'
        ]

        result = refract(sequence, catalog=self.catalog, memory_objects=memory_items)

        assert result.report.determination

    def test_memory_missing_parameter_same_name_nested(self) -> None:
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
            'var4 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
            ' destinationSkyId="$var2.skyId$",'
            ' originEntityId="$var1.entityId$",'
            ' destinationEntityId="$var2.entityId$")'
        ]

        result = refract(sequence, catalog=self.catalog, memory_objects=memory_items)

        assert result.report.determination is False
        assert (
            result.diff[1]
            == '+ var4 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
            ' destinationSkyId="$var2.skyId$",'
            ' originEntityId="$var1.entityId$",'
            ' destinationEntityId="$var2.entityId$",'
            ' date="$var3.datetime.date$")'
        )

    @pytest.mark.skip(reason="ISS54")
    def test_empty_dictionary_in_memory(self) -> None:
        memory = {
            "var1": {"skyId": "foo", "entityId": "bar"},
            "var2": {},
        }

        sequence = [
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        result = refract(
            sequence,
            self.catalog,
            memory_objects=memory,
            use_given_operators_only=True,
        )

        assert result.report.determination is False
        assert result.report.planner_response.best_plan is None
        assert result.report.planner_response.is_no_solution is True

        result = refract(
            sequence,
            self.catalog,
            memory_objects=memory,
        )

        assert result.report.determination is False
        assert result.report.planner_response.best_plan

        assert "SkyScrapperSearchAirport" in [
            step.name for step in result.report.planner_response.best_plan.plan
        ]
