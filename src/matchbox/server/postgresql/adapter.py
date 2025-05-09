"""PostgreSQL adapter for Matchbox server."""

from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pyarrow import Table
from pydantic import BaseModel
from sqlalchemy import and_, bindparam, delete, func, or_, select

from matchbox.common.dtos import (
    ModelAncestor,
    ModelMetadata,
    ModelResolutionName,
    ModelType,
    ResolutionName,
    SourceResolutionName,
)
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph, ResolutionNodeType
from matchbox.common.sources import Match, SourceConfig
from matchbox.server.base import MatchboxDBAdapter, MatchboxSnapshot
from matchbox.server.postgresql.db import (
    MBDB,
    MatchboxBackends,
    MatchboxPostgresSettings,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    ClusterSourceIdentifiers,
    Contains,
    PKSpace,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    SourceConfigs,
)
from matchbox.server.postgresql.utils.db import (
    dump,
    get_resolution_graph,
    restore,
)
from matchbox.server.postgresql.utils.insert import (
    insert_dataset,
    insert_model,
    insert_results,
)
from matchbox.server.postgresql.utils.query import match, query
from matchbox.server.postgresql.utils.results import (
    get_model_metadata,
    get_model_results,
)

T = TypeVar("T")
P = ParamSpec("P")

if TYPE_CHECKING:
    from pyarrow import Table as ArrowTable
else:
    ArrowTable = Any


class FilteredClusters(BaseModel):
    """Wrapper class for filtered cluster queries."""

    has_dataset: bool | None = None

    def count(self) -> int:
        """Counts the number of clusters in the database."""
        with MBDB.get_session() as session:
            query = session.query(
                func.count(func.distinct(Clusters.cluster_id))
            ).select_from(Clusters)

            if self.has_dataset is not None:
                if self.has_dataset:
                    query = query.join(
                        ClusterSourceIdentifiers,
                        ClusterSourceIdentifiers.cluster_id == Clusters.cluster_id,
                    )
                else:
                    query = query.join(
                        Probabilities,
                        Probabilities.cluster == Clusters.cluster_id,
                    )

            return query.scalar()


class FilteredProbabilities(BaseModel):
    """Wrapper class for filtered probability queries."""

    over_truth: bool = False

    def count(self) -> int:
        """Counts the number of probabilities in the database."""
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Probabilities)

            if self.over_truth:
                query = query.join(
                    Resolutions, Probabilities.resolution == Resolutions.resolution_id
                ).filter(
                    and_(
                        Resolutions.truth.isnot(None),
                        Probabilities.probability >= Resolutions.truth,
                    )
                )
            return query.scalar()


class FilteredResolutions(BaseModel):
    """Wrapper class for filtered resolution queries."""

    datasets: bool = False
    humans: bool = False
    models: bool = False

    def count(self) -> int:
        """Counts the number of resolutions in the database."""
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Resolutions)

            filter_list = []
            if self.datasets:
                filter_list.append(Resolutions.type == ResolutionNodeType.DATASET)
            if self.humans:
                filter_list.append(Resolutions.type == ResolutionNodeType.HUMAN)
            if self.models:
                filter_list.append(Resolutions.type == ResolutionNodeType.MODEL)

            if filter_list:
                query = query.filter(or_(*filter_list))

            return query.scalar()


class MatchboxPostgres(MatchboxDBAdapter):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxPostgresSettings):
        """Initialise the PostgreSQL adapter."""
        self.settings = settings
        MBDB.settings = settings
        MBDB.run_migrations()

        PKSpace.initialise()

        self.datasets = SourceConfigs
        self.models = FilteredResolutions(datasets=False, humans=False, models=True)
        self.source_resolutions = FilteredResolutions(
            datasets=True, humans=False, models=False
        )
        self.data = FilteredClusters(has_dataset=True)
        self.clusters = FilteredClusters(has_dataset=False)
        self.merges = Contains
        self.creates = FilteredProbabilities(over_truth=True)
        self.proposes = FilteredProbabilities()

    # Retrieval

    def query(  # noqa: D102
        self,
        source: SourceResolutionName,
        resolution: ResolutionName | None = None,
        threshold: int | None = None,
        limit: int | None = None,
    ) -> ArrowTable:
        return query(
            source=source,
            resolution=resolution,
            threshold=threshold,
            limit=limit,
        )

    def match(  # noqa: D102
        self,
        source_identifier: str,
        source: SourceResolutionName,
        targets: list[SourceResolutionName],
        resolution: ModelResolutionName,
        threshold: int | None = None,
    ) -> list[Match]:
        return match(
            source_identifier=source_identifier,
            source=source,
            targets=targets,
            resolution=resolution,
            threshold=threshold,
        )

    # Data management

    def index(self, source: SourceConfig, data_hashes: Table) -> None:  # noqa: D102
        insert_dataset(
            source=source, data_hashes=data_hashes, batch_size=self.settings.batch_size
        )

    def get_source(self, name: SourceResolutionName) -> SourceConfig:  # noqa: D102
        with MBDB.get_session() as session:
            source: SourceConfigs = (
                session.query(SourceConfigs)
                .join(
                    Resolutions,
                    SourceConfigs.resolution_id == Resolutions.resolution_id,
                )
                .where(Resolutions.name == name)
                .first()
            )
            if source:
                return source.to_common_source()
            else:
                raise MatchboxSourceNotFoundError(name=name)

    def get_resolution_sources(  # noqa: D102
        self,
        name: ModelResolutionName,
    ) -> list[SourceConfig]:
        with MBDB.get_session() as session:
            # Find resolution by name
            resolution: Resolutions | None = (
                session.query(Resolutions).filter(Resolutions.name == name).first()
            )
            if not resolution:
                raise MatchboxResolutionNotFoundError(name=name)
            # Find all resolutions in scope (selected + ancestors)
            relevant_resolutions = (
                session.query(Resolutions)
                .filter(
                    Resolutions.resolution_id.in_(
                        [
                            res.resolution_id
                            for res in resolution.ancestors.union([resolution])
                        ]
                    )
                )
                .subquery()
            )

            # Find all sources matching a resolution in scope
            res_sources: list[SourceConfigs] = (
                session.query(SourceConfigs)
                .join(
                    relevant_resolutions,
                    SourceConfigs.resolution_id == relevant_resolutions.c.resolution_id,
                )
                .all()
            )

            return [s.to_common_source() for s in res_sources]

    def validate_ids(self, ids: list[int]) -> None:  # noqa: D102
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

    def validate_hashes(self, hashes: list[bytes]) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            data_inner_join = (
                session.query(Clusters)
                .filter(
                    Clusters.cluster_hash.in_(
                        bindparam(
                            "ins_hashs",
                            hashes,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        existing_hashes = {item.cluster_hash for item in data_inner_join}
        missing_hashes = set(hashes) - existing_hashes

        if missing_hashes:
            raise MatchboxDataNotFound(
                message="Some items don't exist in Clusters table.",
                table=Clusters.__tablename__,
                data=missing_hashes,
            )

    def cluster_id_to_hash(self, ids: list[int]) -> dict[int, bytes | None]:  # noqa: D102
        initial_dict = {id: None for id in ids}

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

        return initial_dict | {
            item.cluster_id: item.cluster_hash for item in data_inner_join
        }

    def get_resolution_graph(self) -> ResolutionGraph:  # noqa: D102
        return get_resolution_graph()

    def dump(self) -> MatchboxSnapshot:  # noqa: D102
        return dump()

    def drop(self, certain: bool = False) -> None:  # noqa: D102
        if certain:
            MBDB.drop_database()
            PKSpace.initialise()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database and recreate it."
                "It's not expected to be used as part normal operations."
                "If you're sure you want to continue, rerun with certain=True"
            )

    def clear(self, certain: bool = False) -> None:  # noqa: D102
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

    # Model management

    def insert_model(self, model: ModelMetadata) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            left_resolution = (
                session.query(Resolutions)
                .filter(Resolutions.name == model.left_resolution)
                .first()
            )
            if not left_resolution:
                raise MatchboxResolutionNotFoundError(name=model.left_resolution)

            # Overwritten with actual right model if in a link job
            right_resolution = left_resolution
            if model.type == ModelType.LINKER:
                right_resolution = (
                    session.query(Resolutions)
                    .filter(Resolutions.name == model.right_resolution)
                    .first()
                )
                if not right_resolution:
                    raise MatchboxResolutionNotFoundError(name=model.right_resolution)

        insert_model(
            model=model.name,
            left=left_resolution,
            right=right_resolution,
            description=model.description,
        )

    def get_model(self, name: ModelResolutionName) -> ModelMetadata:  # noqa: D102
        resolution = Resolutions.from_name(name=name, res_type="model")
        return get_model_metadata(resolution=resolution)

    def set_model_results(self, name: ModelResolutionName, results: Table) -> None:  # noqa: D102
        resolution = Resolutions.from_name(name=name, res_type="model")
        insert_results(
            results=results,
            resolution=resolution,
            batch_size=self.settings.batch_size,
        )

    def get_model_results(self, name: ModelResolutionName) -> Table:  # noqa: D102
        resolution = Resolutions.from_name(name=name, res_type="model")
        return get_model_results(resolution=resolution)

    def set_model_truth(self, name: ModelResolutionName, truth: int) -> None:  # noqa: D102
        with MBDB.get_session() as session:
            resolution = Resolutions.from_name(
                name=name, res_type="model", session=session
            )
            resolution.truth = truth
            session.commit()

    def get_model_truth(self, name: ModelResolutionName) -> int:  # noqa: D102
        resolution = Resolutions.from_name(name=name, res_type="model")
        return resolution.truth

    def get_model_ancestors(self, name: ModelResolutionName) -> list[ModelAncestor]:  # noqa: D102
        resolution = Resolutions.from_name(name=name, res_type="model")
        return [
            ModelAncestor(name=resolution.name, truth=resolution.truth)
            for resolution in resolution.ancestors
        ]

    def set_model_ancestors_cache(  # noqa: D102
        self,
        name: ModelResolutionName,
        ancestors_cache: list[ModelAncestor],
    ) -> None:
        resolution = Resolutions.from_name(name=name, res_type="model")
        with MBDB.get_session() as session:
            ancestor_names = [ancestor.name for ancestor in ancestors_cache]
            name_to_id = dict(
                session.query(Resolutions.name, Resolutions.resolution_id)
                .filter(Resolutions.name.in_(ancestor_names))
                .all()
            )

            for ancestor in ancestors_cache:
                parent_id = name_to_id.get(ancestor.name)
                if parent_id is None:
                    raise ValueError(f"Model '{ancestor.name}' not found in database")

                session.execute(
                    ResolutionFrom.__table__.update()
                    .where(ResolutionFrom.parent == parent_id)
                    .where(ResolutionFrom.child == resolution.resolution_id)
                    .values(truth_cache=ancestor.truth)
                )

            session.commit()

    def get_model_ancestors_cache(  # noqa: D102
        self, name: ModelResolutionName
    ) -> list[ModelAncestor]:
        resolution = Resolutions.from_name(name=name, res_type="model")
        with MBDB.get_session() as session:
            query = (
                select(Resolutions.name, ResolutionFrom.truth_cache)
                .join(Resolutions, Resolutions.resolution_id == ResolutionFrom.parent)
                .where(ResolutionFrom.child == resolution.resolution_id)
                .where(ResolutionFrom.truth_cache.isnot(None))
            )

            return [
                ModelAncestor(name=name, truth=truth)
                for name, truth in session.execute(query).all()
            ]

    def delete_resolution(  # noqa: D102
        self, resolution: ResolutionName, certain: bool = False
    ) -> None:
        resolution = Resolutions.from_name(name=resolution)
        with MBDB.get_session() as session:
            if certain:
                delete_stmt = delete(Resolutions).where(
                    Resolutions.resolution_id.in_(
                        [
                            resolution.resolution_id,
                            *(d.resolution_id for d in resolution.descendants),
                        ]
                    )
                )
                session.execute(delete_stmt)
                session.commit()
            else:
                children = [r.name for r in resolution.descendants]
                raise MatchboxDeletionNotConfirmed(childen=children)
