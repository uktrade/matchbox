"""Common annotations for the CLI."""

from typing import Annotated

import typer

from matchbox.common.dtos import PermissionType

CollectionOpt = Annotated[
    str,
    typer.Option("--collection", "-c", help="Collection name"),
]
GroupOpt = Annotated[
    str,
    typer.Option("--group", "-g", help="Group name"),
]
PermissionOpt = Annotated[
    PermissionType,
    typer.Option("--permission", "-p", help="Permission level"),
]
DeletionOpt = Annotated[
    bool,
    typer.Option("--certain", "-y", help="Confirm deletion"),
]
UsernameOpt = Annotated[
    str,
    typer.Option("--username", "-u", help="User name"),
]
