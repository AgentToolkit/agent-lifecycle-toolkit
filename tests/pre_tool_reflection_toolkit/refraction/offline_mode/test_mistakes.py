from altk.pre_tool_reflection_toolkit.refraction.src import refract
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import Mapping
from altk.pre_tool_reflection_toolkit.refraction.src.printer import CustomPrint
from nl2flow.compile.options import BasicOperations
from nestful.data_handlers import get_nestful_catalog

import pytest

pytestmark = pytest.mark.refract_extra

PRINTER = CustomPrint()


class TestMistakes:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)
        self.mappings = [
            Mapping(source_name="skyId", target_name="originSkyId"),
            Mapping(source_name="skyId", target_name="destinationSkyId"),
        ]

        # var1 = SkyScrapperSearchAirport(query="New York")
        # var2 = SkyScrapperSearchAirport(query="London")
        # var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$", destinationSkyId="$var2.skyId$", originEntityId="$var1.entityId$", destinationEntityId="$var2.entityId$", date="2024-08-15")
        # var4 = TripadvisorSearchLocation(query="London")
        # var5 = TripadvisorSearchHotels(geoId="$var4.geoId$", checkIn="2024-08-15", checkOut="2024-08-18")

    @pytest.mark.skip(reason="Taking too long! FML")
    def test_made_up_api(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (  # Messed up API name
                'var3 = SkyCrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog)

        assert result.report.determination is False
        assert "SkyScrapperFlightSearch" in [
            step.name for step in result.report.planner_response.best_plan.plan
        ]

        result = refract(tokens, self.catalog, timeout=5)
        assert result.is_timeout

    def test_wrong_label(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var20 = SkyScrapperSearchAirport(query="London")',  # Messed up output
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog, use_given_operators_only=True)
        assert result.report.determination is True

        check_flag = False

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name.startswith(BasicOperations.MAPPER.value)
                and step.inputs[1] == "destinationSkyId"
            ):
                check_flag = True
                assert step.inputs[0] == "skyId" and step.inputs[2] == "var2"

        assert check_flag

    def test_missing_label(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'SkyScrapperSearchAirport(query="London")',  # Missing output
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog, use_given_operators_only=True)
        assert result.report.determination is True

        check_flag = False

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name.startswith(BasicOperations.MAPPER.value)
                and step.inputs[1] == "destinationSkyId"
            ):
                check_flag = True
                assert step.inputs[0] == "skyId" and step.inputs[2] == "var2"

        assert check_flag

    def test_missing_input_parameter(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (  # Missing originSkyId parameter
                "var3 ="
                ' SkyScrapperFlightSearch(destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(
            tokens,
            self.catalog,
            mappings=self.mappings,
            use_given_operators_only=True,
        )

        assert result.report.determination is False
        check_flag = False

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name.startswith(BasicOperations.MAPPER.value)
                and step.inputs[1] == "originSkyId"
            ):
                check_flag = True
                assert step.inputs[0] == "skyId" and step.inputs[2] == "var1"

            if step.name.startswith(BasicOperations.CONFIRM.value):
                assert step.inputs == ["originSkyId"]

        assert check_flag

    def test_made_up_input_parameter(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (  # Messed up originSkyId parameter
                'var3 = SkyScrapperFlightSearch(originalSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(
            tokens,
            self.catalog,
            mappings=self.mappings,
            use_given_operators_only=True,
            allow_remaps=True,
        )
        assert result.report.determination is False

        check_flag = False

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name.startswith(BasicOperations.MAPPER.value)
                and step.inputs[1] == "originSkyId"
            ):
                check_flag = True
                assert step.inputs[0] == "skyId" and step.inputs[2] in [
                    "var1",
                    "var2",
                ]

            if step.name == "SkyScrapperFlightSearch":
                assert step.inputs[0] == "originSkyId"

        assert check_flag

    @pytest.mark.skip(reason="Not sure if we actually need this!")
    def test_made_up_input_parameter_assignment_to_variable(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (  # Assignment to made up var20
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var20.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog, use_given_operators_only=True)
        assert result.report.determination is False

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name.startswith(BasicOperations.MAPPER.value)
                and step.inputs[1] == "destinationSkyId"
            ):
                assert step.inputs[0] == "skyId" and step.inputs[2] == "var2"

    def test_made_up_input_parameter_assignment_to_property(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (  # Assignment to made up parameter of var2
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyayeId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(
            tokens,
            self.catalog,
            mappings=self.mappings,
            use_given_operators_only=True,
        )
        assert result.report.determination is False

        check_flag = False

        for step in result.report.planner_response.best_plan.plan:
            assert not (
                step.name.startswith(BasicOperations.SLOT_FILLER.value)
                and step.inputs[0] == "skyayeId"
            )

            if (
                step.name.startswith(BasicOperations.MAPPER.value)
                and step.inputs[1] == "destinationSkyId"
            ):
                check_flag = True
                assert step.inputs[0] == "skyId" and step.inputs[2] == "var2"

        assert check_flag

    def test_made_up_assignment_recovery_function_call(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (  # Missing parameter date
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$")'
            ),
            'var4 = TripadvisorSearchLocation(query="London")',
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog)

        assert result.report.determination is False
        assert "NewsAPISearchByKeyWord" in [
            step.name for step in result.report.planner_response.best_plan.plan
        ]

        for step in result.report.planner_response.best_plan.plan:
            if (
                step.name.startswith(BasicOperations.MAPPER.value)
                and step.inputs[1] == "date"
            ):
                assert step.inputs[0] == "date" and step.inputs[2] == "var6"

            if step.name.startswith(BasicOperations.CONFIRM.value):
                assert step.inputs == ["date"]

    def test_made_up_input_parameter_missing_step(self) -> None:
        tokens = [
            'var1 = SkyScrapperSearchAirport(query="New York")',
            'var2 = SkyScrapperSearchAirport(query="London")',
            (
                'var3 = SkyScrapperFlightSearch(originSkyId="$var1.skyId$",'
                ' destinationSkyId="$var2.skyId$",'
                ' originEntityId="$var1.entityId$",'
                ' destinationEntityId="$var2.entityId$", date="2024-08-15")'
            ),
            # 'var4 = TripadvisorSearchLocation(query="London")',
            # Missing step
            (
                'var5 = TripadvisorSearchHotels(geoId="$var4.geoId$",'
                ' checkIn="2024-08-15", checkOut="2024-08-18")'
            ),
        ]

        result = refract(tokens, self.catalog)
        assert result.report.determination is False

        prettified_plan = PRINTER.pretty_print_plan(
            result.report.planner_response.best_plan,
            catalog=self.catalog,
            collapse_maps=True,
        ).split("\n")

        assert prettified_plan[3] == 'var4 = TripadvisorSearchLocation(query="London")'
