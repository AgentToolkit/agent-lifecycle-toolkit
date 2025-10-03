from altk.pre_tool_reflection_toolkit.refraction.src import refract
from nl2flow.debug.schemas import SolutionQuality
from nl2flow.compile.schemas import Step, MemoryItem
from tests.utils.refraction.tools.sample_tool_specs import tool_calls, tools


class TestPlanning:
    def setup_method(self) -> None:
        self.tools = tools
        self.tool_calls = tool_calls

    def test_soundness(self) -> None:
        result = refract(self.tool_calls, self.tools)
        assert result.report.determination

    def test_validity(self) -> None:
        result = refract(self.tool_calls, self.tools, report_type=SolutionQuality.VALID)
        assert result.report.determination

    def test_optimality(self) -> None:
        result = refract(
            self.tool_calls, self.tools, report_type=SolutionQuality.OPTIMAL
        )

        assert result.report.determination

    def test_optimality_alternative(self) -> None:
        result = refract(
            self.tool_calls,
            self.tools,
            goals=[Step(name="concur")],
            report_type=SolutionQuality.OPTIMAL,
        )

        assert result.report.determination is False

    def test_optimality_memory(self) -> None:
        result = refract(
            self.tool_calls,
            self.tools,
            goals=[MemoryItem(item_id="employee_info")],
            report_type=SolutionQuality.OPTIMAL,
        )

        assert result.report.determination is False

    def test_optimality_serendipity(self) -> None:
        result = refract(
            sequence=self.tool_calls,
            catalog=self.tools,
            memory_objects={
                "id": 213213,
                "var1": {
                    "info": "...",
                },
            },
            goals=[Step(name="concur")],
            report_type=SolutionQuality.OPTIMAL,
        )

        assert result.report.determination is False
