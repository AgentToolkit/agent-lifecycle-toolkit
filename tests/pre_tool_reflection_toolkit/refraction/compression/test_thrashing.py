from altk.pre_tool_reflection_toolkit.refraction.src import compress
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import Mapping
from nestful.data_handlers import get_nestful_catalog
from nl2flow.compile.options import BasicOperations


class TestThrashing:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)

    def test_thrashing(self) -> None:
        sequence = [
            'var1 = TripadvisorSearchLocation(query="Costa Rica")',
            (
                'var2 = TripadvisorSearchHotels(geoId="$var1.geoId$",'
                ' checkIn="2024-12-01", checkOut="2024-12-15")'
            ),
            'var3 = SkyScrapperSearchAirport(query="New York")',
            'var4 = SkyScrapperSearchAirport(query="Costa Rica")',
            # This should be there -> multiple correction invocation with different instantiation
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var3.skyId$",'
                ' destinationSkyId="$var4.skyId$", date="2024-12-01")'
            ),
            # This should go -> wrong parameters
            'var6 = SkyScrapperSearchAirport(query="New York")',
            # This should go -> already done -> double invocation where one part is (probably) not necessary
            (
                "var7 ="
                ' SkyScrapperFlightSearch(originEntityId="$var6.entityId$",'
                ' destinationSkyId="$var4.skyId$", date="2024-12-01")'
            ),
            # This should go -> wrong parameters again
            'var8 = SkyScrapperSearchAirport(query="New York")',
            # This should go -> already done -> example of thrashing
            (
                'var9 = SkyScrapperFlightSearch(originSkyId="$var8.skyId$",'
                ' destinationSkyId="$var4.skyId$",'
                ' originEntityId="$var8.entityId$", date="2024-12-01")'
            ),
            # This should go -> but either this one or the previous one should be replaced by the correct call
        ]

        mappings = [
            Mapping(
                source_name="entityId",
                target_name="destinationEntityId",
            ),
        ]

        result = compress(sequence=sequence, catalog=self.catalog, mappings=mappings)

        assert result.report.determination is False
        assert result.report.planner_response.best_plan is not None

        actions_in_plan = {
            step.name for step in result.report.planner_response.best_plan.plan
        }

        assert BasicOperations.SLOT_FILLER.value not in actions_in_plan

        non_basic_actions_in_plan = {
            item for item in actions_in_plan if not BasicOperations.is_basic(item)
        }

        assert len(non_basic_actions_in_plan) == 4
        assert {
            "TripadvisorSearchLocation",
            "TripadvisorSearchHotels",
            "SkyScrapperSearchAirport",
            "SkyScrapperFlightSearch",
        } == non_basic_actions_in_plan

    def test_thrashing_with_memory(self) -> None:
        memory = {
            "var1": {"foo": "bar"},
            "var2": {},
            "var3": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
            "var4": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        executed_sequence = [
            'var1 = TripadvisorSearchLocation(query="Costa Rica")',
            (
                'var2 = TripadvisorSearchHotels(geoId="$var1.geoId$",'
                ' checkIn="2024-12-01", checkOut="2024-12-15")'
            ),
            'var3 = SkyScrapperSearchAirport(query="New York")',
            'var4 = SkyScrapperSearchAirport(query="Costa Rica")',
        ]

        sequence = [
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var3.skyId$",'
                ' destinationSkyId="$var4.skyId$", date="2024-12-01")'
            ),
            # This should go -> wrong parameters + items not in memory
            'var6 = SkyScrapperSearchAirport(query="New York")',
            # TODO: To be fixed with mapping issue
            # This should remain -> already done but items not in memory
            (
                "var7 ="
                ' SkyScrapperFlightSearch(originEntityId="$var6.entityId$",'
                ' destinationSkyId="$var4.skyId$", date="2024-12-01")'
            ),
            # This should go -> wrong parameters again
            'var8 = SkyScrapperSearchAirport(query="New York")',
            # This should go -> already done -> example of thrashing
            (
                'var9 = SkyScrapperFlightSearch(originSkyId="$var8.skyId$",'
                ' destinationSkyId="$var4.skyId$",'
                ' originEntityId="$var8.entityId$", date="2024-12-01")'
            ),
            # This should go -> but either this one or the previous one should be replaced by the correct call
        ]

        result = compress(
            sequence=sequence,
            catalog=self.catalog,
            memory_objects=memory,
            memory_steps=executed_sequence,
        )

        assert result.report.determination is False

        actions_in_plan = [
            step.name
            for step in result.report.planner_response.best_plan.plan
            if not BasicOperations.is_basic(step.name)
        ]

        assert actions_in_plan == ["SkyScrapperFlightSearch"]
        assert (
            result.report.planner_response.best_plan.plan[-1].name
            == "SkyScrapperFlightSearch"
        )

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name == BasicOperations.MAPPER.value
                and step.inputs[0] == "originSkyId"
            ):
                assert step.inputs[2] == "var3"

    def test_thrashing_with_memory_not_there(self) -> None:
        memory = {
            "var1": {"foo": "bar"},
            "var2": {},
            "var3": {},
            "var4": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        executed_sequence = [
            'var1 = TripadvisorSearchLocation(query="Costa Rica")',
            (
                'var2 = TripadvisorSearchHotels(geoId="$var1.geoId$",'
                ' checkIn="2024-12-01", checkOut="2024-12-15")'
            ),
            'var3 = SkyScrapperSearchAirport(query="New York")',
            'var4 = SkyScrapperSearchAirport(query="Costa Rica")',
        ]

        sequence = [
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var3.skyId$",'
                ' destinationSkyId="$var4.skyId$", date="2024-12-01")'
            ),
            # This should go -> wrong parameters + items not in memory
            'var6 = SkyScrapperSearchAirport(query="New York")',
            # TODO: To be fixed with mapping issue
            # This should remain -> already done but items not in memory, retry on failed execution
            (
                "var7 ="
                ' SkyScrapperFlightSearch(originEntityId="$var6.entityId$",'
                ' destinationSkyId="$var4.skyId$", date="2024-12-01")'
            ),
            # This should go -> wrong parameters again
            'var8 = SkyScrapperSearchAirport(query="New York")',
            # This should go -> already done -> example of thrashing
            (
                'var9 = SkyScrapperFlightSearch(originSkyId="$var8.skyId$",'
                ' destinationSkyId="$var4.skyId$",'
                ' originEntityId="$var8.entityId$", date="2024-12-01")'
            ),
            # This should go -> but either this one or the previous one should be replaced by the correct call
        ]

        result = compress(
            sequence=sequence,
            catalog=self.catalog,
            memory_objects=memory,
            memory_steps=executed_sequence,
        )

        assert result.report.determination is False

        actions_in_plan = [
            step.name
            for step in result.report.planner_response.best_plan.plan
            if not BasicOperations.is_basic(step.name)
        ]

        assert actions_in_plan == [
            "SkyScrapperSearchAirport",
            "SkyScrapperFlightSearch",
        ]

        assert (
            result.report.planner_response.best_plan.plan[-1].name
            == "SkyScrapperFlightSearch"
        )

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name == BasicOperations.MAPPER.value
                and step.inputs[0] == "originSkyId"
            ):
                assert step.inputs[2] == "var4"
