from nestful.data_handlers import get_nestful_data
from nestful.errors import induce_error_in_sequence
from altk.pre_tool_reflection_toolkit.refraction.src.integration import Refractor

import pytest

pytestmark = pytest.mark.refract_extra


class TestRefractorClass:
    def setup_method(self) -> None:
        self.sequence_data, self.catalog = get_nestful_data(executable=True)
        self.refractor = Refractor(
            catalog=self.catalog, sequence_data=self.sequence_data.data
        )

    def test_refractor_class_without_maps(self) -> None:
        result = self.refractor.refract(self.sequence_data.data[0])
        assert result.report.determination

    @pytest.mark.skip(reason="Takes too long, offline test only.")
    def test_refractor_class_with_maps(self) -> None:
        self.refractor.initialize_maps()

        error_sequence = induce_error_in_sequence(
            self.sequence_data.data[0],
            self.catalog,
            memory={},
            num_errors=3,
            referred_only=True,
        )

        result = self.refractor.refract(error_sequence)
        assert result.report.determination is False
