"""Resolver methodologies shared by client and server."""

from matchbox.common.resolvers.base import (
    ResolverMethod,
    ResolverSettings,
)
from matchbox.common.resolvers.components import Components, ComponentsSettings
from matchbox.common.resolvers.registry import add_resolver_class, get_resolver_class
from matchbox.common.resolvers.runtime import (
    build_override_lookup,
    collect_used_ids,
    compute_override_assignments,
    project_baseline_rows,
    run_resolver_method,
)

add_resolver_class(Components)

__all__ = (
    "ResolverMethod",
    "ResolverSettings",
    "Components",
    "ComponentsSettings",
    "build_override_lookup",
    "collect_used_ids",
    "compute_override_assignments",
    "project_baseline_rows",
    "run_resolver_method",
    "add_resolver_class",
    "get_resolver_class",
)
