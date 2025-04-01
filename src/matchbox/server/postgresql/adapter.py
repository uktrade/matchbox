"""PostgreSQL adapter for Matchbox server."""

from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pyarrow import Table
from pydantic import BaseModel
from sqlalchemy import and_, bindparam, delete, func, or_, select
from sqlalchemy.orm import Session

from matchbox.common.dtos import ModelAncestor, ModelMetadata, ModelType
from matchbox.common.exceptions import (
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxResolutionNotFoundError,
    MatchboxSourceNotFoundError,
)
from matchbox.common.graph import ResolutionGraph, ResolutionNodeType
from matchbox.common.sources import Match, Source, SourceAddress, SourceColumn
from matchbox.server.base import MatchboxBackends, MatchboxDBAdapter, MatchboxSnapshot
from matchbox.server.postgresql import MatchboxPostgresSettings
from matchbox.server.postgresql.db import (
    MBDB,
)
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Sources,
)
from matchbox.server.postgresql.utils.db import (
    dump,
    get_resolution_graph,
    resolve_model_name,
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
            query = session.query(func.count()).select_from(Clusters)
            if self.has_dataset is not None:
                if self.has_dataset:
                    query = query.filter(Clusters.dataset.isnot(None))
                else:
                    query = query.filter(Clusters.dataset.is_(None))
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
        MBDB.create_database()

        self.datasets = Sources
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
        source_address: SourceAddress,
        resolution_name: str | None = None,
        threshold: int | None = None,
        limit: int | None = None,
    ) -> ArrowTable:
        return query(
            engine=MBDB.get_engine(),
            source_address=source_address,
            resolution_name=resolution_name,
            threshold=threshold,
            limit=limit,
        )

    def match(  # noqa: D102
        self,
        source_pk: str,
        source: SourceAddress,
        targets: list[SourceAddress],
        resolution_name: str,
        threshold: int | None = None,
    ) -> list[Match]:
        return match(
            engine=MBDB.get_engine(),
            source_pk=source_pk,
            source=source,
            targets=targets,
            resolution_name=resolution_name,
            threshold=threshold,
        )

    # Data management

    def index(self, source: Source, data_hashes: Table) -> None:  # noqa: D102
        insert_dataset(
            source=source,
            data_hashes=data_hashes,
            engine=MBDB.get_engine(),
            batch_size=self.settings.batch_size,
        )

    def get_source(self, address: SourceAddress) -> Source:  # noqa: D102
        with Session(MBDB.get_engine()) as session:
            source = (
                session.query(Sources)
                .where(
                    and_(
                        Sources.full_name == address.full_name,
                        Sources.warehouse_hash == address.warehouse_hash,
                    )
                )
                .first()
            )
            if source:
                return Source(
                    resolution_name=source.resolution_name,
                    address=address,
                    db_pk=source.id,
                    columns=(
                        SourceColumn(name=name, alias=alias, type=type_)
                        for name, alias, type_ in zip(
                            source.column_names,
                            source.column_aliases,
                            source.column_types,
                            strict=True,
                        )
                    ),
                )
            else:
                raise MatchboxSourceNotFoundError(address=str(address))

    def validate_ids(self, ids: list[int]) -> None:  # noqa: D102
        with Session(MBDB.get_engine()) as session:
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
        with Session(MBDB.get_engine()) as session:
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

        with Session(MBDB.get_engine()) as session:
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
        return get_resolution_graph(engine=MBDB.get_engine())

    def dump(self) -> MatchboxSnapshot:  # noqa: D102
        return dump(engine=MBDB.get_engine())

    def clear(self, certain: bool = False) -> None:  # noqa: D102
        if certain:
            MBDB.clear_database()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database. "
                "It's principally used for testing. \n\n"
                "If you're sure you want to continue, rerun with certain=True"
            )

    def restore(self, snapshot: MatchboxSnapshot, clear: bool = False) -> None:  # noqa: D102
        if snapshot.backend_type != MatchboxBackends.POSTGRES:
            raise TypeError(
                f"Cannot restore {snapshot.backend_type} snapshot to PostgreSQL backend"
            )

        if clear:
            MBDB.clear_database()

        restore(
            engine=MBDB.get_engine(),
            snapshot=snapshot,
            batch_size=self.settings.batch_size,
        )

    # Model management

    def insert_model(self, model: ModelMetadata) -> None:  # noqa: D102
        with Session(MBDB.get_engine()) as session:
            left_resolution = (
                session.query(Resolutions)
                .filter(Resolutions.name == model.left_resolution)
                .first()
            )
            if not left_resolution:
                raise MatchboxResolutionNotFoundError(
                    resolution_name=model.left_resolution
                )

            # Overwritten with actual right model if in a link job
            right_resolution = left_resolution
            if model.type == ModelType.LINKER:
                right_resolution = (
                    session.query(Resolutions)
                    .filter(Resolutions.name == model.right_resolution)
                    .first()
                )
                if not right_resolution:
                    raise MatchboxResolutionNotFoundError(
                        resolution_name=model.right_resolution
                    )

        insert_model(
            model=model.name,
            left=left_resolution,
            right=right_resolution,
            description=model.description,
            engine=MBDB.get_engine(),
        )

    def get_model(self, model: str) -> ModelMetadata:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return get_model_metadata(engine=MBDB.get_engine(), resolution=resolution)

    def set_model_results(self, model: str, results: Table) -> None:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        insert_results(
            results=results,
            resolution=resolution,
            engine=MBDB.get_engine(),
            batch_size=self.settings.batch_size,
        )

    def get_model_results(self, model: str) -> Table:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return get_model_results(engine=MBDB.get_engine(), resolution=resolution)

    def set_model_truth(self, model: str, truth: int) -> None:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        with Session(MBDB.get_engine()) as session:
            session.add(resolution)
            resolution.truth = truth
            session.commit()

    def get_model_truth(self, model: str) -> int:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return resolution.truth

    def get_model_ancestors(self, model: str) -> list[ModelAncestor]:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return [
            ModelAncestor(name=resolution.name, truth=resolution.truth)
            for resolution in resolution.ancestors
        ]

    def set_model_ancestors_cache(  # noqa: D102
        self,
        model: str,
        ancestors_cache: list[ModelAncestor],
    ) -> None:
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        with Session(MBDB.get_engine()) as session:
            session.add(resolution)
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

    def get_model_ancestors_cache(self, model: str) -> list[ModelAncestor]:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        with Session(MBDB.get_engine()) as session:
            session.add(resolution)
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

    def delete_model(self, model: str, certain: bool = False) -> None:  # noqa: D102
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        with Session(MBDB.get_engine()) as session:
            session.add(resolution)
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
