from altk.pre_tool_reflection_toolkit.refraction.src import refract
from nestful.data_handlers import get_nestful_catalog
from nl2flow.compile.options import BasicOperations
from nl2flow.plan.schemas import Action

import pytest


class TestRepetition:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)

    def test_keep_both(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        steps_already_done = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="Costa Rica")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        sequence = [
            'var4 = SkyScrapperSearchAirport(query="New York")',
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            memory_objects=memory_items,
            memory_steps=steps_already_done,
            use_given_operators_only=True,
        )

        assert result.report.determination

    def test_keep_both_with_mistakes(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        steps_already_done = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="Costa Rica")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        sequence = [
            'var4 = SkyScrapperSearchAirport(query="New York")',
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            memory_objects=memory_items,
            memory_steps=steps_already_done,
            use_given_operators_only=True,
        )

        assert result.report.determination is False
        assert len(result.report.planner_response.best_plan.plan) == 8

        for step in result.report.planner_response.best_plan.plan:
            assert isinstance(step, Action)

            if step.name == BasicOperations.MAPPER.value:
                if step.inputs[1] == "originEntityId":
                    assert step.inputs[0] == "entityId" and step.inputs[2] in [
                        "var1",
                        "var4",
                    ]

            else:
                assert step.name in [
                    "SkyScrapperSearchAirport",
                    "SkyScrapperFlightSearch",
                ]

    def test_cant_do(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        steps_already_done = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="Costa Rica")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = SkyScrapperSearchAirport(query="New York")',
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        sequence = [
            'var6 = SkyScrapperSearchAirport(query="New York")',
            (
                'var7 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            memory_objects=memory_items,
            memory_steps=steps_already_done,
            max_try=3,
            use_given_operators_only=True,
        )

        assert result.report.determination is False
        assert result.report.planner_response.best_plan is None
        assert result.report.planner_response.is_no_solution is True

    @pytest.mark.skip(reason="We have enforced goals now")
    def test_remove_and_keep(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        steps_already_done = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="Costa Rica")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = SkyScrapperSearchAirport(query="New York")',
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        sequence = [
            'var6 = SkyScrapperSearchAirport(query="New York")',
            (
                'var7 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            memory_objects=memory_items,
            memory_steps=steps_already_done,
            max_try=3,
            use_given_operators_only=True,
        )

        assert result.report.determination is False
        assert len(result.report.planner_response.best_plan.plan) == 7

        for step in result.report.planner_response.best_plan.plan:
            if step.name != BasicOperations.MAPPER.value:
                assert step.name == "SkyScrapperFlightSearch"

    @pytest.mark.skip(reason="We have enforced goals now")
    def test_remove_and_remove(self) -> None:
        memory_items = {
            "var1": {
                "skyId": "NYCA",
                "entityId": "27537542",
            },
            "var2": {
                "skyId": "SJO",
                "entityId": "2572131",
            },
        }

        steps_already_done = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="Costa Rica")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = SkyScrapperSearchAirport(query="New York")',
            (
                'var5 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var6 = SkyScrapperSearchAirport(query="New York")',
            (
                'var7 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        sequence = [
            'var8 = SkyScrapperSearchAirport(query="New York")',
            (
                'var9 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
        ]

        result = refract(
            sequence,
            catalog=self.catalog,
            memory_objects=memory_items,
            memory_steps=steps_already_done,
            use_given_operators_only=True,
        )

        assert result.report.determination is False
        assert [
            action
            for action in result.report.planner_response.best_plan.plan
            if not BasicOperations.is_basic(action.name)
        ] == []
