from typing import Any, Callable, Dict

from pandas import DataFrame


def cleaner(function: Callable, arguments: Dict) -> Dict[str, Dict[str, Any]]:
    """Define a function to clean a dataset."""
    return {function.__name__: {"function": function, "arguments": arguments}}


def cleaners(*cleaner: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Combine multiple cleaners in a single object to pass to `process()`"""
    return {k: v for d in cleaner for k, v in d.items()}


def process(data: DataFrame, pipeline: Dict[str, Dict[str, Any]]) -> DataFrame:
    """Apply cleaners to input dataframe."""
    curr = data
    for func in pipeline.keys():
        curr = pipeline[func]["function"](curr, **pipeline[func]["arguments"])
    return curr
