from functools import wraps
from os import path
from pathlib import Path

from matchbox.client._handler import get_resolution_graph
from vcr import use_cassette


def vcr_cassette(func):
    test_dir = Path(__file__).resolve().parents[1]
    cassette_name = f"{func.__name__}.yaml"
    path.join(test_dir, "fixtures", cassette_name)

    @wraps(func)
    @use_cassette(f"test/fixtures/vcr_cassettes/{func.__name__}.yaml")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@vcr_cassette
def test_get_resolution_graph():
    assert get_resolution_graph()
