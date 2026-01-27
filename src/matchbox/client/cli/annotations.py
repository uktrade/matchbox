"""Common annotations for the CLI."""

from collections.abc import Callable
from typing import Annotated, Any, TypeVar

import typer
from pydantic import TypeAdapter, ValidationError

from matchbox.common.dtos import CollectionName, GroupName, PermissionType

T = TypeVar("T")


def typer_validator(pydantic_type: type[T]) -> Callable[[Any], T]:
    """Create a Typer callback that validates using a Pydantic type.

    Args:
        pydantic_type: Any Pydantic-compatible type (TypeAlias, BaseModel, etc.)

    Returns:
        A callback function suitable for Typer Option/Argument
    """
    adapter = TypeAdapter(pydantic_type)

    def callback(value: str | None) -> T | None:
        if value is None:
            return None
        try:
            return adapter.validate_python(value)
        except ValidationError as e:
            msg = e.errors()[0].get("msg", str(e))
            raise typer.BadParameter(msg) from e

    return callback


CollectionOpt = Annotated[
    CollectionName,
    typer.Option(
        "--collection",
        "-c",
        help="Collection name",
        callback=typer_validator(CollectionName),
    ),
]
GroupOpt = Annotated[
    GroupName,
    typer.Option(
        "--group", "-g", help="Group name", callback=typer_validator(GroupName)
    ),
]
PermissionOpt = Annotated[
    PermissionType,
    typer.Option("--permission", "-p", help="Permission level"),
]
DeletionOpt = Annotated[
    bool,
    typer.Option(
        "--certain",
        "-y",
        help="Confirm deletion",
        prompt="Are you sure you want to delete this resource?",
    ),
]
UsernameOpt = Annotated[
    str,
    typer.Option("--username", "-u", help="User name"),
]
