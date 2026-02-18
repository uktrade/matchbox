"""Resolver methodology registry."""

from matchbox.common.logging import logger
from matchbox.common.resolvers.base import ResolverMethod

_RESOLVER_CLASSES: dict[str, type[ResolverMethod]] = {}


def add_resolver_class(resolver_class: type[ResolverMethod]) -> None:
    """Register a resolver methodology class."""
    if not issubclass(resolver_class, ResolverMethod):
        raise ValueError("The argument is not a proper subclass of ResolverMethod.")
    _RESOLVER_CLASSES[resolver_class.__name__] = resolver_class
    logger.debug(f"Registered resolver class: {resolver_class.__name__}")


def get_resolver_class(class_name: str) -> type[ResolverMethod]:
    """Retrieve a resolver methodology class by name."""
    try:
        return _RESOLVER_CLASSES[class_name]
    except KeyError as e:
        raise ValueError(f"Unknown resolver class: {class_name}") from e
