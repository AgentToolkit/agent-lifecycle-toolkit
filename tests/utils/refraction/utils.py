from pathlib import Path
from os import listdir
from os.path import isfile, join
from time import time
from typing import Callable, TypeVar, Any, Optional, List, cast
from altk.pre_tool_reflection_toolkit.refraction.src.schemas import Mapping

import json


F = TypeVar("F", bound=Callable[..., Any])


def get_cached_maps() -> List[Mapping]:
    path_to_file = Path(__file__).parent.resolve()
    relative_path_to_data = "./data/cached_nestful_maps.json"

    abs_path_to_data = Path.joinpath(path_to_file, relative_path_to_data).resolve()

    return [Mapping(**item) for item in json.loads(open(abs_path_to_data).read())]


def load_data(file_name: str) -> Any:
    path_to_file = Path(__file__).parent.resolve()
    relative_path_to_data = "./data"

    abs_path_to_data = Path.joinpath(path_to_file, relative_path_to_data).resolve()

    data_file: Optional[str] = next(
        (
            item
            for item in listdir(abs_path_to_data)
            if isfile(join(abs_path_to_data, item)) and file_name in item
        ),
        None,
    )

    assert data_file, f"File {file_name} not found!"

    if data_file.endswith(".json"):
        with open(join(abs_path_to_data, data_file), "r") as file:
            try:
                json_object = json.loads(file.read())
                return json_object

            except json.JSONDecodeError as e:
                print(e)
                return []

    elif data_file.endswith(".jsonl"):
        list_of_jsons = []
        with open(join(abs_path_to_data, data_file), "r") as file:
            for line in file:
                try:
                    json_object = json.loads(line)
                    list_of_jsons.append(json_object)

                except json.JSONDecodeError as e:
                    print(e)
                    return []

        return list_of_jsons
    else:
        raise ValueError(f"Unknown file extension for {data_file}!")


def timeit(func: F) -> F:
    def inner(*args: Any, **kwargs: Any) -> Any:
        start_time = time()
        result = func(*args, **kwargs)

        end_time = time()
        elapsed_time = end_time - start_time

        print(f"\nTime taken: {round(elapsed_time, 2)} secs")
        return result

    return cast(F, inner)


def print_results(func: F) -> F:
    def inner(*args: Any, **kwargs: Any) -> Any:
        batch_results = func(*args, **kwargs)
        how_many_succeeded = batch_results.how_many_succeeded()

        print(f"\n\nAverage time taken: {batch_results.time_taken} sec")
        print(f"Success Rate: {how_many_succeeded}/{len(batch_results.results)}")

        print(f"Compression Rate: {batch_results.mean_compression}")

        troubled_indices = []
        for index, result in enumerate(batch_results.results):
            if not result.report.determination:
                troubled_indices.append(str(index))

        print(f"Troubled Indices: {', '.join(troubled_indices)}")
        return batch_results

    return cast(F, inner)
