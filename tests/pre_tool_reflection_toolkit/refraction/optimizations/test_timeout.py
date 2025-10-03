from altk.pre_tool_reflection_toolkit.refraction.src import refract
from nestful import SequencingData
from nestful.data_handlers import get_nestful_data_instance

import pytest

pytestmark = pytest.mark.refract_extra


class TestTimeout:
    def setup_method(self) -> None:
        self.sequence, self.catalog = get_nestful_data_instance(
            executable=True, index=0
        )

    def test_timeout_fine(self) -> None:
        result = refract(sequence=self.sequence, catalog=self.catalog, timeout=2.0)
        assert result.report.determination is True

    def test_long_gone_wrong(self) -> None:
        sequence, catalog = get_nestful_data_instance(executable=True, index=0)
        incomplete_sequence = SequencingData(output=self.sequence.output[2:])

        result = refract(sequence=incomplete_sequence, catalog=catalog, timeout=1.0)

        assert result.report.determination is None
        assert result.is_timeout is True
