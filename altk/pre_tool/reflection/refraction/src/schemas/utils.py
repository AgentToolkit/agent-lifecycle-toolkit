from typing import List, Optional, Any
from functools import reduce


def open_mean(
    list_of_number_like_items: List[Any],
    key: Optional[str] = None,
) -> float:
    if not list_of_number_like_items:
        return 0.0

    filtered_items = (
        [
            getattr(item, key, None)
            for item in list_of_number_like_items
            if item is not None
        ]
        if key
        else list_of_number_like_items
    )

    filtered_float_items = [float(item) for item in filtered_items if item is not None]

    filtered_float_items = [
        item for item in filtered_float_items if item < float("inf")
    ]

    mean = float(reduce(lambda x, y: x + y, filtered_float_items)) / len(
        filtered_float_items
    )

    return round(mean, 2)
