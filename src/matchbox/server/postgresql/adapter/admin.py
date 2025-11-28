"""Admin PostgreSQL mixin for Matchbox server."""

from sqlalchemy import bindparam, select

from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
)
from matchbox.server.base import MatchboxSnapshot
from matchbox.server.postgresql.db import MBDB, MatchboxBackends
from matchbox.server.postgresql.orm import Clusters, PKSpace, Users
from matchbox.server.postgresql.utils.db import dump, restore


class MatchboxPostgresAdminMixin:
    """Admin mixin for the PostgreSQL adapter for Matchbox."""

    # User management

    def login(self, user_name: str) -> int:  # noqa: D102
        with MBDB.get_session() as session:
            if user_id := session.scalar(
                select(Users.user_id).where(Users.name == user_name)
            ):
                return user_id

            user = Users(name=user_name)
            session.add(user)
            session.commit()

            return user.user_id

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
            PKSpace.initialise()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database and recreate it."
                "It's not expected to be used as part normal operations."
                "If you're sure you want to continue, rerun with certain=True"
            )

    def clear(self, certain: bool) -> None:  # noqa: D102
        if certain:
            MBDB.clear_database()
            PKSpace.initialise()
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
