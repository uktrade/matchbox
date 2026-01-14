"""Admin PostgreSQL mixin for Matchbox server."""

from typing import Literal

from sqlalchemy import and_, bindparam, delete, select, union_all

from matchbox.common.dtos import (
    BackendResourceType,
    CollectionName,
    Group,
    GroupName,
    PermissionGrant,
    PermissionType,
    User,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxGroupAlreadyExistsError,
    MatchboxGroupNotFoundError,
    MatchboxSystemGroupError,
    MatchboxUserNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.server.base import PERMISSION_GRANTS, MatchboxSnapshot
from matchbox.server.postgresql.db import MBDB, MatchboxBackends
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceKey,
    Collections,
    EvalJudgements,
    Groups,
    Permissions,
    Probabilities,
    Results,
    UserGroups,
    Users,
)
from matchbox.server.postgresql.utils.db import dump, restore


class MatchboxPostgresAdminMixin:
    """Admin mixin for the PostgreSQL adapter for Matchbox."""

    # User management

    def login(self, user: User) -> User:  # noqa: D102
        with MBDB.get_session() as session:
            # Try to find existing user
            existing_user = session.scalar(
                select(Users).where(Users.name == user.user_name)
            )

            if existing_user:
                # Update email if provided
                if user.email and existing_user.email != user.email:
                    existing_user.email = user.email
                    session.commit()

                return User(
                    user_name=existing_user.name,
                    email=existing_user.email,
                )

            # Create new user
            new_user = Users(name=user.user_name, email=user.email)
            session.add(new_user)
            session.commit()

            return User(
                user_name=new_user.name,
                email=new_user.email,
            )

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

            # Check if already a member
            existing_membership = session.scalar(
                select(UserGroups).where(
                    and_(
                        UserGroups.user_id == user.user_id,
                        UserGroups.group_id == group.group_id,
                    )
                )
            )

            if not existing_membership:
                membership = UserGroups(
                    user_id=user.user_id,
                    group_id=group.group_id,
                )
                session.add(membership)

                session.commit()

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

    # Permissions management

    def check_permission(  # noqa: D102
        self,
        user_name: str,
        permission: PermissionType,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> bool:
        with MBDB.get_session() as session:
            # Get user
            user = session.scalar(select(Users).where(Users.name == user_name))
            if not user:
                return False

            # Get user's group IDs
            user_group_ids = session.scalars(
                select(UserGroups.group_id).where(UserGroups.user_id == user.user_id)
            ).all()

            if not user_group_ids:
                return False

            # Get permissions that would satisfy this check
            sufficient_permissions = PERMISSION_GRANTS[permission]

            # Check permissions based on resource type
            if resource == BackendResourceType.SYSTEM:
                # Check system permissions
                grant = session.scalar(
                    select(Permissions).where(
                        and_(
                            Permissions.is_system == True,  # noqa: E712
                            Permissions.group_id.in_(user_group_ids),
                            Permissions.permission.in_(sufficient_permissions),
                        )
                    )
                )
            else:
                # Check collection permissions
                collection = session.scalar(
                    select(Collections).where(Collections.name == resource)
                )
                if not collection:
                    raise MatchboxCollectionNotFoundError(name=resource)

                grant = session.scalar(
                    select(Permissions).where(
                        and_(
                            Permissions.collection_id == collection.collection_id,
                            Permissions.group_id.in_(user_group_ids),
                            Permissions.permission.in_(sufficient_permissions),
                        )
                    )
                )

            return grant is not None

    def get_permissions(  # noqa: D102
        self,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> list[PermissionGrant]:
        with MBDB.get_session() as session:
            if resource == BackendResourceType.SYSTEM:
                # Get system permissions
                permissions_query = (
                    select(Permissions)
                    .where(Permissions.is_system == True)  # noqa: E712
                    .join(Groups, Permissions.group_id == Groups.group_id)
                )
            else:
                # Get collection permissions
                collection = session.scalar(
                    select(Collections).where(Collections.name == resource)
                )
                if not collection:
                    raise MatchboxCollectionNotFoundError(name=resource)

                permissions_query = (
                    select(Permissions)
                    .where(Permissions.collection_id == collection.collection_id)
                    .join(Groups, Permissions.group_id == Groups.group_id)
                )

            permissions_orm = session.scalars(permissions_query).all()

            grants = [
                PermissionGrant(
                    group_name=GroupName(perm.group.name),
                    permission=PermissionType(perm.permission),
                )
                for perm in permissions_orm
            ]

            return grants

    def grant_permission(  # noqa: D102
        self,
        group_name: GroupName,
        permission: PermissionType,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> None:
        with MBDB.get_session() as session:
            # Get group
            group = session.scalar(select(Groups).where(Groups.name == group_name))
            if not group:
                raise MatchboxGroupNotFoundError(f"Group '{group_name}' not found")

            if resource == BackendResourceType.SYSTEM:
                # Grant system permission
                # Check if already exists
                existing = session.scalar(
                    select(Permissions).where(
                        and_(
                            Permissions.group_id == group.group_id,
                            Permissions.permission == permission,
                            Permissions.is_system == True,  # noqa: E712
                        )
                    )
                )

                if not existing:
                    new_permission = Permissions(
                        group_id=group.group_id,
                        permission=permission,
                        is_system=True,
                    )
                    session.add(new_permission)
            else:
                # Grant collection permission
                collection = session.scalar(
                    select(Collections).where(Collections.name == resource)
                )
                if not collection:
                    raise MatchboxCollectionNotFoundError(name=resource)

                # Check if already exists
                existing = session.scalar(
                    select(Permissions).where(
                        and_(
                            Permissions.group_id == group.group_id,
                            Permissions.permission == permission,
                            Permissions.collection_id == collection.collection_id,
                        )
                    )
                )

                if not existing:
                    new_permission = Permissions(
                        group_id=group.group_id,
                        permission=permission,
                        collection_id=collection.collection_id,
                    )
                    session.add(new_permission)

            session.commit()

            logger.info(
                f"Granted {permission} permission on '{resource}' "
                f"to group '{group_name}'",
                prefix="Grant permission",
            )

    def revoke_permission(  # noqa: D102
        self,
        group_name: GroupName,
        permission: PermissionType,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
    ) -> None:
        with MBDB.get_session() as session:
            # Get group
            group = session.scalar(select(Groups).where(Groups.name == group_name))
            if not group:
                raise MatchboxGroupNotFoundError(f"Group '{group_name}' not found")

            if resource == BackendResourceType.SYSTEM:
                # Revoke system permission
                result = session.execute(
                    delete(Permissions).where(
                        and_(
                            Permissions.group_id == group.group_id,
                            Permissions.permission == permission,
                            Permissions.is_system == True,  # noqa: E712
                        )
                    )
                )
            else:
                # Revoke collection permission
                collection = session.scalar(
                    select(Collections).where(Collections.name == resource)
                )
                if not collection:
                    raise MatchboxCollectionNotFoundError(name=resource)

                result = session.execute(
                    delete(Permissions).where(
                        and_(
                            Permissions.group_id == group.group_id,
                            Permissions.permission == permission,
                            Permissions.collection_id == collection.collection_id,
                        )
                    )
                )

            session.commit()

            if result.rowcount > 0:
                logger.info(
                    f"Revoked {permission} permission on '{resource}' "
                    f"from group '{group_name}'",
                    prefix="Revoke permission",
                )
            else:
                logger.info(
                    f"Permission {permission} on '{resource}' "
                    f"not present for group '{group_name}'",
                    prefix="Revoke permission",
                )

    # Data management

    def validate_ids(self, ids: list[int]) -> bool:  # noqa: D102
        with MBDB.get_session() as session:
            data_inner_join = (
                session.query(Clusters)
                .filter(
                    Clusters.cluster_id.in_(
                        bindparam(
                            "ins_ids",
                            ids,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        existing_ids = {item.cluster_id for item in data_inner_join}
        missing_ids = set(ids) - existing_ids

        if missing_ids:
            raise MatchboxDataNotFound(
                message="Some items don't exist in Clusters table.",
                table=Clusters.__tablename__,
                data=missing_ids,
            )

        return True

    def dump(self) -> MatchboxSnapshot:  # noqa: D102
        return dump()

    def drop(self, certain: bool) -> None:  # noqa: D102
        if certain:
            MBDB.drop_database()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database and recreate it."
                "It's not expected to be used as part normal operations."
                "If you're sure you want to continue, rerun with certain=True"
            )

    def clear(self, certain: bool) -> None:  # noqa: D102
        if certain:
            MBDB.clear_database()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop all rows in the database but not the "
                "tables themselves. It's primarily used to reset following tests."
                "If you're sure you want to continue, rerun with certain=True"
            )

    def restore(self, snapshot: MatchboxSnapshot) -> None:  # noqa: D102
        if snapshot.backend_type != MatchboxBackends.POSTGRES:
            raise TypeError(
                f"Cannot restore {snapshot.backend_type} snapshot to PostgreSQL backend"
            )

        MBDB.clear_database()

        restore(
            snapshot=snapshot,
            batch_size=self.settings.batch_size,
        )

    def delete_orphans(self) -> int:  # noqa: D102
        with MBDB.get_session() as session:
            # Get all cluster ids in related tables
            union_all_cte = union_all(
                select(EvalJudgements.endorsed_cluster_id.label("cluster_id")),
                select(EvalJudgements.shown_cluster_id.label("cluster_id")),
                select(ClusterSourceKey.cluster_id),
                select(Probabilities.cluster_id),
                select(Results.left_id.label("cluster_id")),
                select(Results.right_id.label("cluster_id")),
            ).cte("union_all_cte")

            # Deduplicate only once
            not_orphans = (
                select(union_all_cte.c.cluster_id).distinct().cte("not_orphans")
            )

            # Return clusters not in related tables
            stmt = delete(Clusters).where(
                ~select(not_orphans.c.cluster_id)
                .where(not_orphans.c.cluster_id == Clusters.cluster_id)
                .exists()
            )
            # Delete orphans
            deletion = session.execute(stmt)

            session.commit()

            logger.info(f"Deleted {deletion.rowcount} orphans", prefix="Delete orphans")
            return deletion.rowcount
