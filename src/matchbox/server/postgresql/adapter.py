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
from matchbox.server.base import MatchboxDBAdapter
from matchbox.server.postgresql.db import MBDB, MatchboxPostgresSettings
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Probabilities,
    ResolutionFrom,
    Resolutions,
    Sources,
)
from matchbox.server.postgresql.utils.db import get_resolution_graph, resolve_model_name
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
    """Wrapper class for filtered cluster queries"""

    has_dataset: bool | None = None

    def count(self) -> int:
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Clusters)
            if self.has_dataset is not None:
                if self.has_dataset:
                    query = query.filter(Clusters.dataset.isnot(None))
                else:
                    query = query.filter(Clusters.dataset.is_(None))
            return query.scalar()


class FilteredProbabilities(BaseModel):
    """Wrapper class for filtered probability queries"""

    over_truth: bool = False

    def count(self) -> int:
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Probabilities)

            if self.over_truth:
                query = query.join(
                    Resolutions, Probabilities.resolution == Resolutions.resolution_id
                ).filter(
                    and_(
                        Resolutions.truth.isnot(None),
                        Probabilities.probability > Resolutions.truth,
                    )
                )
            return query.scalar()


class FilteredResolutions(BaseModel):
    """Wrapper class for filtered resolution queries"""

    datasets: bool = False
    humans: bool = False
    models: bool = False

    def count(self) -> int:
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

    def query(
        self,
        source_address: SourceAddress,
        resolution_name: str | None = None,
        threshold: float | None = None,
        limit: int | None = None,
    ) -> ArrowTable:
        """Queries the database from an optional point of truth.

        Args:
            source_address: the `SourceAddress` object identifying the source to query
            resolution_name (optional): the resolution to use for filtering results
                If not specified, will use the dataset resolution for the queried source
            threshold (optional): the threshold to use for creating clusters
                If None, uses the models' default threshold
                If an integer, uses that threshold for the specified model, and the
                model's cached thresholds for its ancestors
            limit (optional): the number to use in a limit clause. Useful for testing

        Returns:
            The resulting matchbox IDs in Arrow format
        """
        return query(
            engine=MBDB.get_engine(),
            source_address=source_address,
            resolution_name=resolution_name,
            threshold=threshold,
            limit=limit,
        )

    def match(
        self,
        source_pk: str,
        source: SourceAddress,
        targets: list[SourceAddress],
        resolution_name: str,
        threshold: float | None = None,
    ) -> list[Match]:
        """Matches an ID in a source dataset and returns the keys in the targets.

        Args:
            source_pk: The primary key to match from the source.
            source: The address of the source dataset.
            targets: The addresses of the target datasets.
            resolution_name: The name of the resolution to use for matching.
            threshold (optional): the threshold to use for creating clusters
                If None, uses the resolutions' default threshold
                If an integer, uses that threshold for the specified resolution, and the
                resolution's cached thresholds for its ancestors
                Will use these threshold values instead of the cached thresholds
        """
        return match(
            engine=MBDB.get_engine(),
            source_pk=source_pk,
            source=source,
            targets=targets,
            resolution_name=resolution_name,
            threshold=threshold,
        )

    # Data management

    def index(self, source: Source, data_hashes: Table) -> None:
        """Indexes to Matchbox a source dataset in your warehouse.

        Args:
            source: The source dataset to index.
            data_hashes: The Arrow table with the hash of each data row
        """
        insert_dataset(
            source=source,
            data_hashes=data_hashes,
            engine=MBDB.get_engine(),
            batch_size=self.settings.batch_size,
        )

    def get_source(self, address: SourceAddress) -> Source:
        """Get a source from its name address.

        Args:
            address: The name address for the source

        Returns:
            A Source object
        """
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
                    alias=source.alias,
                    address=address,
                    db_pk=source.id,
                    columns=[
                        SourceColumn(name=name, alias=alias, type=type_)
                        for name, alias, type_ in zip(
                            source.column_names,
                            source.column_aliases,
                            source.column_types,
                            strict=True,
                        )
                    ],
                )
            else:
                raise MatchboxSourceNotFoundError(address=str(address))

    def validate_ids(self, ids: list[int]) -> None:
        """Validates a list of IDs exist in the database.

        Args:
            ids: A list of IDs to validate.

        Raises:
            MatchboxDataNotFound: If some items don't exist in the target table.
        """
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

    def validate_hashes(self, hashes: list[bytes]) -> None:
        """Validates a list of hashes exist in the database.

        Args:
            hashes: A list of hashes to validate.

        Raises:
            MatchboxDataNotFound: If some items don't exist in the target table.
        """
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

    def cluster_id_to_hash(self, ids: list[int]) -> dict[int, bytes | None]:
        """Get a lookup of Cluster hashes from a list of IDs.

        Args:
            ids: A list of IDs to get hashes for.

        Returns:
            A dictionary mapping IDs to hashes.
        """
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

    def get_resolution_graph(self) -> ResolutionGraph:
        """Get the full resolution graph."""
        return get_resolution_graph(engine=MBDB.get_engine())

    def clear(self, certain: bool = False) -> None:
        """Clears all data from the database.

        Args:
            certain: Whether to clear the database without confirmation.
        """
        if certain:
            MBDB.clear_database()
        else:
            raise MatchboxDeletionNotConfirmed(
                "This operation will drop the entire database. "
                "It's principally used for testing. \n\n"
                "If you're sure you want to continue, rerun with certain=True"
            )

    # Model management

    def insert_model(self, model: ModelMetadata) -> None:
        """Writes a model to Matchbox.

        Args:
            model: ModelMetadata object with the model's metadata

        Raises:
            MatchboxDataNotFound: If, for a linker, the source models weren't found in
                the database
        """
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

    def get_model(self, model: str) -> ModelMetadata:
        """Get a model from the database."""
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return get_model_metadata(engine=MBDB.get_engine(), resolution=resolution)

    def set_model_results(self, model: str, results: Table) -> None:
        """Set the results for a model."""
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        insert_results(
            results=results,
            resolution=resolution,
            engine=MBDB.get_engine(),
            batch_size=self.settings.batch_size,
        )

    def get_model_results(self, model: str) -> Table:
        """Get the results for a model."""
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return get_model_results(engine=MBDB.get_engine(), resolution=resolution)

    def set_model_truth(self, model: str, truth: float) -> None:
        """Sets the truth threshold for this model, changing the default clusters."""
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        with Session(MBDB.get_engine()) as session:
            session.add(resolution)
            resolution.truth = truth
            session.commit()

    def get_model_truth(self, model: str) -> float:
        """Gets the current truth threshold for this model."""
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return resolution.truth

    def get_model_ancestors(self, model: str) -> list[ModelAncestor]:
        """Gets the current truth values of all ancestors.

        Returns a list of ModelAncestor objects mapping model names to their current
        truth thresholds.

        Unlike ancestors_cache which returns cached values, this property returns
        the current truth values of all ancestor models.
        """
        resolution = resolve_model_name(model=model, engine=MBDB.get_engine())
        return [
            ModelAncestor(name=resolution.name, truth=resolution.truth)
            for resolution in resolution.ancestors
        ]

    def set_model_ancestors_cache(
        self,
        model: str,
        ancestors_cache: list[ModelAncestor],
    ) -> None:
        """Updates the cached ancestor thresholds.

        Args:
            ancestors_cache: List of ModelAncestor objects mapping model names to
                their truth thresholds
        """
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

    def get_model_ancestors_cache(self, model: str) -> list[ModelAncestor]:
        """Gets the cached ancestor thresholds, converting hashes to model names.

        Returns a list of ModelAncestor objects mapping model names to their cached
        truth thresholds.

        This is required because each point of truth needs to be stable, so we choose
        when to update it, caching the ancestor's values in the model itself.
        """
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

    def delete_model(self, model: str, certain: bool = False) -> None:
        """Delete a model from the database.

        Args:
            certain: Whether to delete the model without confirmation.
        """
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
                session.delete(resolution)
                session.commit()
            else:
                children = [r.name for r in resolution.descendants]
                raise MatchboxDeletionNotConfirmed(childen=children)
