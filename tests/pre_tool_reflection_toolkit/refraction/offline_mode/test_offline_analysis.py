from nestful import SequencingData, SequencingDataset
from nestful.data_handlers import get_nestful_catalog, get_nestful_data
from altk.pre_tool_reflection_toolkit.refraction.src.batch_actions import (
    run_all,
    run_all_batch,
)
from altk.pre_tool_reflection_toolkit.refraction.src.schemas.results import BatchResults
from tests.utils.refraction.utils import load_data, print_results, timeit
from typing import List, Any, Optional
from inspect import stack

import json
import pytest

pytestmark = pytest.mark.refract_extra


class TestOfflineAnalysis:
    def setup_method(self) -> None:
        self.catalog = get_nestful_catalog(executable=True)

    @staticmethod
    def load_data(file_name: Optional[str] = None) -> Any:
        file_name = file_name or stack()[1][3]
        return load_data(file_name)

    @timeit
    @print_results
    def test_ground_truth_exe(self) -> BatchResults:
        sequence_data, catalog = get_nestful_data(executable=True)
        results: BatchResults = run_all(
            sequence_data,
            catalog,
            use_given_operators_only=True,
            allow_remaps=True,
        )

        assert results.how_many_succeeded() == len(sequence_data.data) - 8
        return results

    @timeit
    @print_results
    def test_ground_truth_sgd(self) -> BatchResults:
        sequence_data, catalog = get_nestful_data(name="sgd", executable=False)
        results: BatchResults = run_all(
            sequence_data, catalog, use_given_operators_only=True
        )

        assert results.how_many_succeeded() == len(sequence_data.data)
        return results

    @timeit
    @print_results
    def test_ground_truth_glaive(self) -> BatchResults:
        sequence_data, catalog = get_nestful_data(name="glaive", executable=False)

        results: BatchResults = run_all(
            sequence_data, catalog, use_given_operators_only=True
        )

        assert results.how_many_succeeded() == len(sequence_data.data) - 14
        return results

    @pytest.mark.skip(reason="This will be enabled with the new offline API.")
    def test_one_shot(self) -> BatchResults:
        data = self.load_data()
        sequence_data = SequencingDataset(
            data=[
                SequencingData(output=json.loads(str(item.get("output", []))))
                for item in data
            ]
        )

        results: BatchResults = run_all(sequence_data, catalog=self.catalog)
        return results

    @staticmethod
    def get_gt_sample(
        sample: SequencingData, gt_samples: List[SequencingData]
    ) -> Optional[SequencingData]:
        filtered_samples = [gt for gt in gt_samples if gt.input == sample.input]

        if len(filtered_samples) == 1:
            return filtered_samples[0]

        return None

    @timeit
    @print_results
    def test_one_shot_with_gt(self) -> BatchResults:
        gt_sequence_data, catalog = get_nestful_data(executable=True)

        data = self.load_data(file_name="test_step_by_step_no_memory.json")
        parsed_data = [SequencingData.model_validate(item) for item in data]

        test_data = []
        for item in parsed_data:
            gt = self.get_gt_sample(item, gt_sequence_data.data)

            if gt is not None:
                test_data.append((item, gt, catalog))

        test_data = test_data[:10]

        results: BatchResults = run_all_batch(test_data)
        return results

    @timeit
    @print_results
    def test_step_by_step_with_memory(self) -> BatchResults:
        catalog = get_nestful_catalog(executable=True)

        data = self.load_data(file_name="test_step_by_step_no_memory.json")
        test_data = [
            (SequencingData.model_validate(item), None, catalog) for item in data
        ]

        test_data = test_data[:5]

        results: BatchResults = run_all_batch(test_data, run_step_by_step=True)
        return results
