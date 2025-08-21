"""Re-export of names used to build pipelines."""

from enum import StrEnum


class SourceConfig:
    """."""

    def fetch():
        """."""
        ...


class Source(SourceConfig):
    """Adds client to Source Config."""


class RelationalDBLocation:
    """."""


class DAG:
    """."""


class ModelConfig:
    """."""


class Model(ModelConfig):
    """Adds behaviours to ModelConfig DTO."""

    def fetch_left():
        """."""
        ...

    def clean_left():
        """."""
        ...

    def run():
        """."""
        ...


class QueryConfig:
    """."""


class Query(QueryConfig):
    """Adds behaviours to QueryConfig DTO."""


class DeduperType(StrEnum):
    """."""

    DETERMINISTIC = "DETERMINISTIC"


class LinkerType(StrEnum):
    """."""

    DETERMINISTIC = "DETERMINISTIC"


class Collection:
    """."""

    def add_dag():
        """."""
        ...

    def new_version():
        """."""
        ...

    def get_dag():
        """."""
        ...

    def add_source():
        """."""
        ...

    def add_model():
        """."""
        ...

    def get_source():
        """."""
        ...

    def get_model():
        """."""
        ...

    def map_key():
        """."""
        ...

    ...


class Cleaner:
    """."""

    ...


def run_model():
    """."""
    pass


def run_query():
    """."""
    pass
