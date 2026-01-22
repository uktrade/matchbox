"""Admin PostgreSQL mixin for Matchbox server."""

from typing import Literal

from sqlalchemy import CursorResult, and_, bindparam, delete, select, union_all

from matchbox.common.dtos import (
    BackendResourceType,
    CollectionName,
    DefaultGroup,
    DefaultUser,
    GroupName,
    LoginResponse,
    PermissionGrant,
    PermissionType,
    User,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxGroupNotFoundError,
)
from matchbox.common.logging import logger
from matchbox.server.base import (
    PERMISSION_GRANTS,
    MatchboxSnapshot,
)
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
    insert,
)
from matchbox.server.postgresql.utils.db import dump, grant_permission, restore


class MatchboxPostgresAdminMixin:
    """Admin mixin for the PostgreSQL adapter for Matchbox."""

    # User management

    def login(self, user: User) -> LoginResponse:  # noqa: D102
        with MBDB.get_session() as session:
            # Get public and admins groups
            public_group = session.scalar(
                select(Groups).where(Groups.name == DefaultGroup.PUBLIC)
            )
            admins_group = session.scalar(
                select(Groups).where(Groups.name == GroupName(DefaultGroup.ADMINS))
            )

            # Upsert user
            if user.email:
                session.execute(
                    insert(Users)
                    .values(
                        name=user.user_name,
                        email=user.email,
                    )
                    .on_conflict_do_update(
                        index_elements=["name"],
                        set_={"email": user.email},
                    )
                )
            else:
                session.execute(
                    insert(Users)
                    .values(name=user.user_name)
                    .on_conflict_do_nothing(index_elements=["name"])
                )

            # Get the user object
            user_obj = session.scalar(select(Users).where(Users.name == user.user_name))

            # Ensure user is in public group
            session.execute(
                insert(UserGroups)
                .values(
                    user_id=user_obj.user_id,
                    group_id=public_group.group_id,
                )
                .on_conflict_do_nothing()
            )

            # Check if any other non-public user exists
            other_user_exists = (
                session.scalar(
                    select(Users.user_id)
                    .where(Users.name != DefaultUser.PUBLIC)
                    .where(Users.user_id != user_obj.user_id)
                    .limit(1)
                )
                is not None
            )

            setup_mode_admin = not other_user_exists

            if setup_mode_admin:
                # Try to add to admins group
                admin_result: CursorResult = session.execute(
                    insert(UserGroups)
                    .values(
                        user_id=user_obj.user_id,
                        group_id=admins_group.group_id,
                    )
                    .on_conflict_do_nothing()
                )

                if admin_result.rowcount > 0:
                    logger.info(
                        f"Added first user '{user.user_name}' to {DefaultGroup.ADMINS} "
                        "group",
                        prefix="Login",
                    )

            session.commit()

            return LoginResponse(
                user=User(
                    user_name=user_obj.name,
                    email=user_obj.email,
                ),
                setup_mode_admin=setup_mode_admin,
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
            grant_permission(
                session=session,
                group_name=group_name,
                permission=permission,
                resource=resource,
            )
            session.commit()

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
            Groups.initialise()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database and recreate it."
                "It's not expected to be used as part normal operations."
                "If you're sure you want to continue, rerun with certain=True"
            )

    def clear(self, certain: bool) -> None:  # noqa: D102
        if certain:
            MBDB.clear_database()
            Groups.initialise()
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
