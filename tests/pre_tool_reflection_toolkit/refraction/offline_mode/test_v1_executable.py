from typing import Any
from altk.pre_tool_reflection_toolkit.refraction.src import refract
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import (
    DebuggingResult,
)
from nestful.data_handlers import get_nestful_data_instance

import pytest

pytestmark = pytest.mark.refract_extra


class TestV1Executable:
    @staticmethod
    def check_sequence(data_id: int, **kwargs: Any) -> DebuggingResult:
        sequence, catalog = get_nestful_data_instance(executable=True, index=data_id)

        result: DebuggingResult = refract(
            sequence, catalog, use_given_operators_only=True, **kwargs
        )
        return result

    def test_sequence_0_direct(self) -> None:
        result = self.check_sequence(data_id=0)
        assert result.report.determination

    def test_sequence_4_direct(self) -> None:
        result = self.check_sequence(data_id=3)
        assert result.report.determination

    def test_sequence_82(self) -> None:
        # this sample contains a nested response
        result = self.check_sequence(data_id=81)
        assert result.report.determination

    def test_sequence_55(self) -> None:
        # this sample gets a map overwritten
        result = self.check_sequence(data_id=54)
        assert result.report.determination is True

    def test_sequence_21(self) -> None:
        # this sample has parameter reordering
        result = self.check_sequence(data_id=20)
        assert result.report.determination is True

    @pytest.mark.skip(reason="Regression, needs investigation")
    def test_sequence_84(self) -> None:
        # this sample has a reassignment
        result = self.check_sequence(data_id=83)
        assert result.report.determination is False

        result = self.check_sequence(data_id=83, allow_remaps=True)
        assert result.report.determination is True

    def test_sgd_11(self) -> None:
        # this sample gets stuck
        sequence, catalog = get_nestful_data_instance(
            name="sgd",
            executable=False,
            index=10,
        )

        result = refract(sequence, catalog)
        assert result.report.determination is True

    def test_sgd_32(self) -> None:
        # this sample needs to let optionals pass
        sequence, catalog = get_nestful_data_instance(
            name="sgd",
            executable=False,
            index=31,
        )

        result = refract(sequence, catalog)
        assert result.report.determination is True

    @pytest.mark.skip(reason="Planner throwing error, needs investigation")
    def test_glaive_16(self) -> None:
        # this sample is throwing an error in the planner
        sequence, catalog = get_nestful_data_instance(
            name="glaive",
            executable=False,
            index=15,
        )

        result = refract(sequence, catalog)
        assert result.report.determination is True

    def test_glaive_137(self) -> None:
        # this sample wrangles optional parameters
        sequence, catalog = get_nestful_data_instance(
            name="glaive",
            executable=False,
            index=136,
        )

        result = refract(sequence, catalog)
        assert result.report.determination is True

    def test_glaive_144(self) -> None:
        # this sample wrangles optional parameters together
        sequence, catalog = get_nestful_data_instance(
            name="glaive",
            executable=False,
            index=143,
        )

        result = refract(sequence, catalog)
        assert result.report.determination is True
