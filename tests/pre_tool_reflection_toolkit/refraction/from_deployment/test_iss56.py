from nestful.data_handlers import get_nestful_catalog
from nestful import SequenceStep
from altk.pre_tool_reflection_toolkit.refraction.src import refract
from typing import Dict, Any


class TestISS56:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)

    def test_unseen_q(self) -> None:
        call = {
            "name": "Coronavirus_Smartable_GetStats",
            "arguments": {"q": "Australia"},
            "label": "var1",
        }

        sequence_object = SequenceStep(**call)
        result = refract(
            sequence=sequence_object,
            catalog=self.catalog,
            use_cc=True,
        )

        assert result.report.determination is False

        corrected_call = result.corrected_function_call(catalog=self.catalog, memory={})

        assert corrected_call is not None

        assert corrected_call.is_executable
        assert corrected_call.tokenized == [
            'var1 = WeatherAPI.com_Forecast_Weather_API(q="Australia")',
            'var1 = Coronavirus_Smartable_GetStats(location="$var1.location$")',
        ]

    def test_recovery_call_on_empty_memory(self) -> None:
        memory: Dict[str, Any] = {"var2": {}}
        call = {
            "name": "Tripadvisor_Get_Restaurant_Details",
            "arguments": {"restaurantsId": "$var2.restaurantsId$"},
            "label": "var3",
        }

        sequence_object = SequenceStep(**call)
        result = refract(
            sequence=sequence_object,
            catalog=self.catalog,
            memory_objects=memory,
        )

        assert result.report.determination is False
        assert result.report.planner_response.best_plan

        action_names = [
            step.name for step in result.report.planner_response.best_plan.plan
        ]

        assert (
            "TripadvisorSearchRestaurants" in action_names
            or "Tripadvisor_Search_Restaurants" in action_names
        )

        assert "Tripadvisor_Get_Restaurant_Details" in action_names
