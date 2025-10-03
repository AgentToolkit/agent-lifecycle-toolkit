from altk.pre_tool_reflection_toolkit.refraction.src import refract
from altk.pre_tool_reflection_toolkit.refraction.src.printer import CustomPrint
from nl2flow.compile.schemas import Step
from nestful import SequencingData
from nestful.data_handlers import get_nestful_catalog

PRINTER = CustomPrint()


class TestOperatorGoal:
    def setup_method(self) -> None:
        self.memory_items = {
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

        self.sequence = [
            'var3 = SkyScrapperSearchAirport(query="New York")',
            (
                'SkyScrapperFlightSearch(originSkyId="$var3.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var3.entityId$",'
                ' destinationEntityId="$var2.entityId$",'
                ' date="$date_from_user$")'
            ),
        ]

        self.catalog = get_nestful_catalog(executable=True)

    def test_operator_goal(self) -> None:
        result = refract(
            self.sequence,
            catalog=self.catalog,
            memory_objects=self.memory_items,
            goals=[Step(name="SkyScrapperFlightSearch")],
        )

        assert result.report.determination
        assert (
            result.report.planner_response.best_plan.plan[-1].name
            == "SkyScrapperFlightSearch"
        )

    def test_operator_goal_with_parameters(self) -> None:
        result = refract(
            self.sequence,
            catalog=self.catalog,
            memory_objects=self.memory_items,
            goals=[
                Step(
                    name="SkyScrapperFlightSearch",
                    parameters=[
                        "destinationSkyId",
                        "destinationEntityId",
                        "date",
                    ],
                    maps=[
                        "$var2.skyId$",
                        "$var2.entityId$",
                        "$date_from_user$",
                    ],
                )
            ],
        )

        assert result.report.determination
        assert (
            result.report.planner_response.best_plan.plan[-1].name
            == "SkyScrapperFlightSearch"
        )

        pretty_print = PRINTER.pretty_print_plan(
            result.report.planner_response.best_plan,
            collapse_maps=True,
            catalog=self.catalog,
            sequence=SequencingData.parse_pretty_print(self.sequence),
        ).split("\n")
        assert pretty_print[-1] == self.sequence[-1]
