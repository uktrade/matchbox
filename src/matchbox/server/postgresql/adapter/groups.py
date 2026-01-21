"""Groups PostgreSQL mixin for Matchbox server."""

from sqlalchemy import CursorResult, and_, delete, select
from sqlalchemy.dialects.postgresql import insert

from matchbox.common.dtos import (
    Group,
    GroupName,
    User,
)
from matchbox.common.exceptions import (
    MatchboxDeletionNotConfirmed,
    MatchboxGroupAlreadyExistsError,
    MatchboxGroupNotFoundError,
    MatchboxSystemGroupError,
    MatchboxUserNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.server.postgresql.db import MBDB
from matchbox.server.postgresql.orm import (
    Groups,
    UserGroups,
    Users,
)


class MatchboxPostgresGroupsMixin:
    """Groups mixin for the PostgreSQL adapter for Matchbox."""

    # Group management

    def get_user_groups(self, user_name: str) -> list[GroupName]:  # noqa: D102
        with MBDB.get_session() as session:
            user = session.scalar(select(Users).where(Users.name == user_name))

            if not user:
                raise MatchboxUserNotFoundError(f"User '{user_name}' not found")

            group_names = [GroupName(group.name) for group in user.groups]

            return group_names

    def list_groups(self) -> list[Group]:  # noqa: D102
        with MBDB.get_session() as session:
            groups_orm = session.scalars(select(Groups)).all()

            groups = [
                Group(
                    name=GroupName(g.name),
                    description=g.description,
                    is_system=g.is_system,
                    members=[
                        User(
                            user_id=member.user_id,
                            user_name=member.name,
                            email=member.email,
                        )
                        for member in g.members
                    ],
                )
                for g in groups_orm
            ]

            return groups

    def get_group(self, name: GroupName) -> Group:  # noqa: D102
        with MBDB.get_session() as session:
            group = session.scalar(select(Groups).where(Groups.name == name))

            if not group:
                raise MatchboxGroupNotFoundError(f"Group '{name}' not found")

            return Group(
                name=GroupName(group.name),
                description=group.description,
                is_system=group.is_system,
                members=[
                    User(
                        user_id=member.user_id,
                        user_name=member.name,
                        email=member.email,
                    )
                    for member in group.members
                ],
            )

    def create_group(self, group: Group) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            # Check if group already exists
            existing = session.scalar(select(Groups).where(Groups.name == group.name))

            if existing:
                raise MatchboxGroupAlreadyExistsError(
                    f"Group '{group.name}' already exists"
                )

            # Create the group
            new_group = Groups(
                name=group.name,
                description=group.description,
                is_system=group.is_system,
            )
            session.add(new_group)
            session.commit()

            logger.info(f"Created group '{group.name}'", prefix="Create group")

    def delete_group(self, name: GroupName, certain: bool = False) -> None:  # noqa: D102
        if not certain:
            raise MatchboxDeletionNotConfirmed(
                f"This operation will delete the group '{name}' and all its "
                "permission grants. If you're sure you want to continue, rerun "
                "with certain=True"
            )

        with MBDB.get_session() as session:
            group = session.scalar(select(Groups).where(Groups.name == name))

            if not group:
                raise MatchboxGroupNotFoundError(f"Group '{name}' not found")

            if group.is_system:
                raise MatchboxSystemGroupError(f"Cannot delete system group '{name}'")

            session.delete(group)
            session.commit()

            logger.info(f"Deleted group '{name}'", prefix="Delete group")

    # User-group membership

    def add_user_to_group(self, user_name: str, group_name: GroupName) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            # Get user
            user = session.scalar(select(Users).where(Users.name == user_name))
            if not user:
                raise MatchboxUserNotFoundError(f"User '{user_name}' not found")

            # Get group
            group = session.scalar(select(Groups).where(Groups.name == group_name))
            if not group:
                raise MatchboxGroupNotFoundError(f"Group '{group_name}' not found")

            # Upsert membership
            result: CursorResult = session.execute(
                insert(UserGroups)
                .values(
                    user_id=user.user_id,
                    group_id=group.group_id,
                )
                .on_conflict_do_nothing()
            )

            session.commit()

            if result.rowcount > 0:
                logger.info(
                    f"Added user '{user_name}' to group '{group_name}'",
                    prefix="Add user to group",
                )
            else:
                logger.info(
                    f"User '{user_name}' already belongs to '{group_name}'",
                    prefix="Add user to group",
                )

    def remove_user_from_group(self, user_name: str, group_name: GroupName) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            # Get user
            user = session.scalar(select(Users).where(Users.name == user_name))
            if not user:
                raise MatchboxUserNotFoundError(f"User '{user_name}' not found")

            # Get group
            group = session.scalar(select(Groups).where(Groups.name == group_name))
            if not group:
                raise MatchboxGroupNotFoundError(f"Group '{group_name}' not found")

            # Delete membership
            result = session.execute(
                delete(UserGroups).where(
                    and_(
                        UserGroups.user_id == user.user_id,
                        UserGroups.group_id == group.group_id,
                    )
                )
            )

            session.commit()

            if result.rowcount > 0:
                logger.info(
                    f"Removed user '{user_name}' from group '{group_name}'",
                    prefix="Remove user from group",
                )
            else:
                logger.info(
                    f"User '{user_name}' not present in group '{group_name}'",
                    prefix="Remove user from group",
                )
