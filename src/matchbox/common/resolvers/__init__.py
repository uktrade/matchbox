"""Resolver methodologies shared by client and server."""

from matchbox.common.resolvers.base import ResolverMethod, ResolverSettings
from matchbox.common.resolvers.components import Components, ComponentsSettings
from matchbox.common.resolvers.registry import add_resolver_class, get_resolver_class

add_resolver_class(Components)

__all__ = (
    "ResolverMethod",
    "ResolverSettings",
    "Components",
    "ComponentsSettings",
    "add_resolver_class",
    "get_resolver_class",
)
