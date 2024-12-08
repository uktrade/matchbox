from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from rustworkx import PyDiGraph
from sqlalchemy import Engine, and_, bindparam, delete, func, or_, select
from sqlalchemy.orm import Session

from matchbox.common.db import Match, Source, SourceWarehouse
from matchbox.common.exceptions import (
    MatchboxDataError,
    MatchboxDatasetError,
    MatchboxModelError,
)
from matchbox.common.results import ClusterResults, ProbabilityResults, Results
from matchbox.server.base import MatchboxDBAdapter, MatchboxModelAdapter
from matchbox.server.postgresql.db import MBDB, MatchboxPostgresSettings
from matchbox.server.postgresql.orm import (
    Clusters,
    Contains,
    Models,
    ModelsFrom,
    Probabilities,
    Sources,
)
from matchbox.server.postgresql.utils.db import get_model_subgraph
from matchbox.server.postgresql.utils.insert import (
    insert_dataset,
    insert_model,
    insert_results,
)
from matchbox.server.postgresql.utils.query import match, query
from matchbox.server.postgresql.utils.results import (
    get_model_clusters,
    get_model_probabilities,
)

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDataFrame
    from polars import DataFrame as PolarsDataFrame
    from pyarrow import Table as ArrowTable
else:
    PandasDataFrame = Any
    PolarsDataFrame = Any
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
                query = query.join(Models, Probabilities.model == Models.hash).filter(
                    and_(
                        Models.truth.isnot(None),
                        Probabilities.probability > Models.truth,
                    )
                )
            return query.scalar()


class FilteredModels(BaseModel):
    """Wrapper class for filtered model queries"""

    datasets: bool = False
    humans: bool = False
    models: bool = False

    def count(self) -> int:
        with MBDB.get_session() as session:
            query = session.query(func.count()).select_from(Models)

            filter_list = []
            if self.datasets:
                filter_list.append(Models.type == "dataset")
            if self.humans:
                filter_list.append(Models.type == "human")
            if self.models:
                filter_list.append(Models.type == "model")

            if filter_list:
                query = query.filter(or_(*filter_list))

            return query.scalar()


class MatchboxPostgresModel(MatchboxModelAdapter):
    """An adapter for Matchbox models in PostgreSQL."""

    def __init__(self, model: Models, backend: "MatchboxPostgres"):
        self.model: Models = model
        self.backend: "MatchboxPostgres" = backend

    @property
    def hash(self) -> bytes:
        with Session(MBDB.get_engine()) as session:
            session.add(self.model)
            return self.model.hash

    @property
    def name(self) -> str:
        with Session(MBDB.get_engine()) as session:
            session.add(self.model)
            return self.model.name

    @property
    def probabilities(self) -> ProbabilityResults:
        """Retrieve probabilities for this model."""
        return get_model_probabilities(engine=MBDB.get_engine(), model=self.model)

    @property
    def clusters(self) -> ClusterResults:
        """Retrieve clusters for this model."""
        return get_model_clusters(engine=MBDB.get_engine(), model=self.model)

    @property
    def results(self) -> Results:
        """Retrieve results for this model."""
        return Results(probabilities=self.probabilities, clusters=self.clusters)

    @results.setter
    def results(self, results: Results) -> None:
        """Inserts results for this model."""
        insert_results(
            results=results,
            model=self.model,
            engine=MBDB.get_engine(),
            batch_size=self.backend.settings.batch_size,
        )

    @property
    def truth(self) -> float:
        """Gets the current truth threshold for this model."""
        with Session(MBDB.get_engine()) as session:
            session.add(self.model)
            return self.model.truth

    @truth.setter
    def truth(self, truth: float) -> None:
        """Sets the truth threshold for this model, changing the default clusters."""
        with Session(MBDB.get_engine()) as session:
            self.model.truth = truth
            session.add(self.model)
            session.commit()

    @property
    def ancestors(self) -> dict[str, float | None]:
        """
        Gets the current truth values of all ancestors.
        Returns a dict mapping model names to their current truth thresholds.

        Unlike ancestors_cache which returns cached values, this property returns
        the current truth values of all ancestor models.
        """
        with Session(MBDB.get_engine()) as session:
            session.add(self.model)
            return {model.name: model.truth for model in self.model.ancestors}

    @property
    def ancestors_cache(self) -> dict[str, float]:
        """
        Gets the cached ancestor thresholds, converting hashes to model names.

        Returns a dictionary mapping model names to their truth thresholds.

        This is required because each point of truth needs to be stable, so we choose
        when to update it, caching the ancestor's values in the model itself.
        """
        with Session(MBDB.get_engine()) as session:
            query = (
                select(Models.name, ModelsFrom.truth_cache)
                .join(Models, Models.hash == ModelsFrom.parent)
                .where(ModelsFrom.child == self.model.hash)
                .where(ModelsFrom.truth_cache.isnot(None))
            )

            return {
                name: truth_cache for name, truth_cache in session.execute(query).all()
            }

    @ancestors_cache.setter
    def ancestors_cache(self, new_values: dict[str, float]) -> None:
        """
        Updates the cached ancestor thresholds.

        Args:
            new_values: Dictionary mapping model names to their truth thresholds
        """

        with Session(MBDB.get_engine()) as session:
            model_names = list(new_values.keys())
            name_to_hash = dict(
                session.query(Models.name, Models.hash)
                .filter(Models.name.in_(model_names))
                .all()
            )

            for model_name, truth_value in new_values.items():
                parent_hash = name_to_hash.get(model_name)
                if parent_hash is None:
                    raise ValueError(f"Model '{model_name}' not found in database")

                session.execute(
                    ModelsFrom.__table__.update()
                    .where(ModelsFrom.parent == parent_hash)
                    .where(ModelsFrom.child == self.model.hash)
                    .values(truth_cache=truth_value)
                )

            session.commit()

    @classmethod
    def get_model(
        cls, model_name: str, backend: "MatchboxPostgres"
    ) -> "MatchboxPostgresModel":
        with Session(MBDB.get_engine()) as session:
            if model := session.query(Models).filter_by(name=model_name).first():
                return cls(model, backend=backend)
            else:
                raise MatchboxModelError(model_name=model_name)


class MatchboxPostgres(MatchboxDBAdapter):
    """A PostgreSQL adapter for Matchbox."""

    def __init__(self, settings: MatchboxPostgresSettings):
        self.settings = settings
        MBDB.settings = settings
        MBDB.create_database()

        self.datasets = Sources
        self.models = FilteredModels(datasets=False, humans=False, models=True)
        self.data = FilteredClusters(has_dataset=True)
        self.clusters = FilteredClusters(has_dataset=False)
        self.merges = Contains
        self.creates = FilteredProbabilities(over_truth=True)
        self.proposes = FilteredProbabilities()

    def query(
        self,
        selector: dict[str, list[str]],
        model: str | None = None,
        threshold: float | dict[str, float] | None = None,
        return_type: Literal["pandas", "arrow", "polars"] | None = None,
        limit: int = None,
    ) -> PandasDataFrame | ArrowTable | PolarsDataFrame:
        """Queries the database from an optional model of truth.

        Args:
            selector: the tables and fields to query
            return_type: the form to return data in, one of "pandas" or "arrow"
                Defaults to pandas for ease of use
            model (optional): the model to use for filtering results
            threshold (optional): the threshold to use for creating clusters
                If None, uses the models' default threshold
                If a float, uses that threshold for the specified model, and the
                model's cached thresholds for its ancestors
                If a dictionary, expects a shape similar to model.ancestors, keyed
                by model name and valued by the threshold to use for that model. Will
                use these threshold values instead of the cached thresholds
            limit (optional): the number to use in a limit clause. Useful for testing

        Returns:
            Data in the requested return type
        """
        return query(
            selector=selector,
            engine=MBDB.get_engine(),
            return_type=return_type if return_type else "pandas",
            model=model,
            threshold=threshold,
            limit=limit,
        )

    def match(
        self,
        source_id: str,
        source: str,
        target: str | list[str],
        model: str,
        threshold: float | dict[str, float] | None = None,
    ) -> Match | list[Match]:
        """Matches an ID in a source dataset and returns the keys in the targets.

        Args:
            source_id: The ID of the source to match.
            source: The name of the source dataset.
            target: The name of the target dataset(s).
            model: The name of the model to use for matching.
            threshold (optional): the threshold to use for creating clusters
                If None, uses the models' default threshold
                If a float, uses that threshold for the specified model, and the
                model's cached thresholds for its ancestors
                If a dictionary, expects a shape similar to model.ancestors, keyed
                by model name and valued by the threshold to use for that model. Will
                use these threshold values instead of the cached thresholds
        """
        return match(
            source_id=source_id,
            source=source,
            target=target,
            model=model,
            engine=MBDB.get_engine(),
            threshold=threshold,
        )

    def index(self, dataset: Source) -> None:
        """Indexes a data from your data warehouse within Matchbox.

        Args:
            dataset: The dataset to index.
            engine: The SQLAlchemy engine of your data warehouse.
        """
        insert_dataset(
            dataset=dataset,
            engine=MBDB.get_engine(),
            batch_size=self.settings.batch_size,
        )

    def validate_hashes(self, hashes: list[bytes]) -> None:
        """Validates a list of hashes exist in the database.

        Args:
            hashes: A list of hashes to validate.

        Raises:
            MatchboxDataError: If some items don't exist in the target table.
        """
        with Session(MBDB.get_engine()) as session:
            data_inner_join = (
                session.query(Clusters)
                .filter(
                    Clusters.hash.in_(
                        bindparam(
                            "ins_hashs",
                            hashes,
                            expanding=True,
                        )
                    )
                )
                .all()
            )

        existing_hashes = {item.hash for item in data_inner_join}
        missing_hashes = set(hashes) - existing_hashes

        if missing_hashes:
            raise MatchboxDataError(
                message="Some items don't exist in Clusters table.",
                table=Clusters.__tablename__,
                data=missing_hashes,
            )

    def get_dataset(self, db_schema: str, db_table: str, engine: Engine) -> Source:
        """Get a source dataset from the database.

        Args:
            db_schema: The schema of the dataset.
            db_table: The table of the dataset.
            engine: The engine to use to connect to your data warehouse.
        """
        with Session(MBDB.get_engine()) as session:
            dataset = (
                session.query(Sources)
                .filter_by(schema=db_schema, table=db_table)
                .first()
            )
            if dataset:
                return Source(
                    db_schema=dataset.schema,
                    db_table=dataset.table,
                    db_pk=dataset.id,
                    database=SourceWarehouse.from_engine(engine),
                )
            else:
                raise MatchboxDatasetError(db_schema=db_schema, db_table=db_table)

    def get_model_subgraph(self) -> PyDiGraph:
        """Get the full subgraph of a model."""
        return get_model_subgraph(engine=MBDB.get_engine())

    def get_model(self, model: str) -> MatchboxPostgresModel:
        """Get a model from the database.

        Args:
            model: The model to get.
        """
        with Session(MBDB.get_engine()) as session:
            if model := session.query(Models).filter_by(name=model).first():
                return MatchboxPostgresModel(model=model, backend=self)
            else:
                raise MatchboxModelError(model_name=model)

    def delete_model(self, model: str, certain: bool = False) -> None:
        """Delete a model from the database.

        Args:
            model: The model to delete.
            certain: Whether to delete the model without confirmation.
        """
        with Session(MBDB.get_engine()) as session:
            if model := session.query(Models).filter(Models.name == model).first():
                if certain:
                    delete_stmt = delete(Models).where(
                        Models.hash.in_(
                            [model.hash, *(d.hash for d in model.descendants)]
                        )
                    )
                    session.execute(delete_stmt)
                    session.delete(model)
                    session.commit()
                else:
                    childen = model.descendants
                    children_names = ", ".join([m.name for m in childen])
                    raise ValueError(
                        f"This operation will delete the models {children_names}, "
                        "as well as all probabilities they have created. \n\n"
                        "It won't delete validation associated with these "
                        "probabilities. \n\n"
                        "If you're sure you want to continue, rerun with certain=True"
                    )
            else:
                raise MatchboxModelError(model_name=model)

    def insert_model(
        self, model: str, left: str, description: str, right: str | None = None
    ) -> None:
        """Writes a model to Matchbox.

        Args:
            model: The name of the model
            left: The name of the model on the left side of this model's join, or the
                name of the dataset it deduplicates
            right: The name of the model on the right side of this model's join
                When deduplicating, this is None
            description: A description of the model

        Raises
            MatchboxDataError if, for a linker, the source models weren't found in
                the database
        """
        with Session(MBDB.get_engine()) as session:
            left_model = session.query(Models).filter(Models.name == left).first()
            if not left_model:
                raise MatchboxModelError(model_name=left)

            # Overwritten with actual right model if in a link job
            right_model = left_model
            if right:
                right_model = session.query(Models).filter(Models.name == right).first()
                if not right_model:
                    raise MatchboxModelError(model_name=right)

        insert_model(
            model=model,
            left=left_model,
            right=right_model,
            description=description,
            engine=MBDB.get_engine(),
        )

    def clear(self, certain: bool = False) -> None:
        """Clears all data from the database.

        Args:
            certain: Whether to clear the database without confirmation.
        """
        if certain:
            MBDB.clear_database()
        else:
            raise ValueError(
                "This operation will drop the entire database. "
                "It's principally used for testing. \n\n"
                "If you're sure you want to continue, rerun with certain=True"
            )
