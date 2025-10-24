"""Client-side helpers for retrieving and preparing evaluation samples."""

from typing import cast

import polars as pl
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

from matchbox.client import _handler
from matchbox.client.dags import DAG
from matchbox.client.results import Results
from matchbox.common.dtos import ModelResolutionName, ModelResolutionPath, SourceConfig
from matchbox.common.eval import Judgement, ModelComparison, precision_recall
from matchbox.common.exceptions import MatchboxSourceTableError


class EvaluationItem(BaseModel):
    """A cluster awaiting evaluation, with pre-computed display data."""

    model_config = {"arbitrary_types_allowed": True}

    cluster_id: int
    dataframe: pl.DataFrame  # Original raw data (needed for judgement leaf IDs)
    display_data: dict[str, list[str]]  # field_name -> [val1, val2, val3]
    duplicate_groups: list[list[int]]  # Groups of leaf_ids with identical data
    display_columns: list[int]  # Representative leaf_id for each displayed column
    assignments: dict[int, str] = {}  # display_column_index -> group_letter


def create_judgement(item: EvaluationItem, user_id: int) -> Judgement:
    """Convert item assignments to Judgement - no default group assignment.

    Args:
        item: Evaluation item with assignments
        user_id: User ID for the judgement

    Returns:
        Judgement with endorsed groups based on assignments
    """
    groups: dict[str, list[int]] = {}

    for col_idx, group in item.assignments.items():
        leaf_ids = item.duplicate_groups[col_idx]
        groups.setdefault(group, []).extend(leaf_ids)

    endorsed = [sorted(set(leaf_ids)) for leaf_ids in groups.values()]
    return Judgement(user_id=user_id, shown=item.cluster_id, endorsed=endorsed)


def create_evaluation_item(
    df: pl.DataFrame, source_configs: list[tuple[str, SourceConfig]], cluster_id: int
) -> EvaluationItem:
    """Create EvaluationItem with pre-computed display data."""
    # Get all data columns (exclude metadata columns)
    data_cols = [c for c in df.columns if c not in ["root", "leaf", "key"]]

    # Extract field names from source configs
    field_names = []
    for _, config in source_configs:
        for field in config.index_fields:
            if field.name not in field_names:
                field_names.append(field.name)

    if not data_cols:
        # No data columns found - return empty item
        return EvaluationItem(
            cluster_id=cluster_id,
            dataframe=df,
            display_data={},
            duplicate_groups=[],
            display_columns=[],
            assignments={},
        )

    # Deduplicate using polars group_by
    df_dedup = df.select(["leaf"] + data_cols)
    grouped = df_dedup.group_by(data_cols, maintain_order=True).agg(pl.col("leaf"))

    duplicate_groups = [group for group in grouped["leaf"]]
    display_columns = [group[0] for group in duplicate_groups]

    # Build display data
    display_data = {}
    for field_name in field_names:
        qualified_cols = [c for c in data_cols if c.endswith(f"_{field_name}")]

        values = []
        for leaf_id in display_columns:
            row = df.filter(pl.col("leaf") == leaf_id).row(0, named=True)
            val = next(
                (str(row.get(c, "")).strip() for c in qualified_cols if row.get(c)), ""
            )
            values.append(val)

        if any(values):  # Only include fields with data
            display_data[field_name] = values

    return EvaluationItem(
        cluster_id=cluster_id,
        dataframe=df,
        display_data=display_data,
        duplicate_groups=duplicate_groups,
        display_columns=display_columns,
        assignments={},
    )


def get_samples(
    n: int,
    dag: DAG,
    user_id: int,
    resolution: ModelResolutionName | None = None,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data as EvaluationItems.

    Args:
        n: Number of clusters to sample
        dag: DAG for which to retrieve samples
        user_id: ID of the user requesting the samples
        resolution: The optional resolution from which to sample. If not provided,
            the final step in the DAG is used

    Returns:
        Dictionary of cluster ID to EvaluationItems describing the cluster

    Raises:
        MatchboxSourceTableError: If a source cannot be queried from a location using
            provided or default clients.
    """
    if resolution:
        resolution_path: ModelResolutionPath = dag.get_model(resolution).resolution_path
    else:
        resolution_path: ModelResolutionPath = dag.final_step.resolution_path

    samples: pl.DataFrame = cast(
        pl.DataFrame,
        pl.from_arrow(
            _handler.sample_for_eval(n=n, resolution=resolution_path, user_id=user_id)
        ),
    )

    if not len(samples):
        return {}

    results_by_source: list[pl.DataFrame] = []
    source_configs: list[tuple[str, SourceConfig]] = []

    for source_resolution in samples["source"].unique():
        try:
            source = dag.get_source(source_resolution)
        except ValueError as exc:  # pragma: no cover - defensive path
            raise MatchboxSourceTableError(
                f"Source '{source_resolution}' not found in DAG. "
                "Ensure DAG was loaded with all resolutions."
            ) from exc

        source_configs.append((source_resolution, source.config))

        samples_by_source = samples.filter(pl.col("source") == source_resolution)
        keys_by_source = samples_by_source["key"].to_list()

        try:
            source_data = pl.concat(
                source.fetch(batch_size=10_000, qualify_names=True, keys=keys_by_source)
            )
        except OperationalError as exc:
            raise MatchboxSourceTableError(
                f"Could not query source '{source_resolution}' from warehouse. "
                "Check warehouse connection and ensure source table exists."
            ) from exc

        samples_and_source = samples_by_source.join(
            source_data, left_on="key", right_on=source.qualified_key
        )
        desired_columns = ["root", "leaf", "key"] + source.qualified_index_fields
        results_by_source.append(samples_and_source[desired_columns])

    if not results_by_source:
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    results_by_root: dict[int, EvaluationItem] = {}
    for root in all_results["root"].unique():
        cluster_df = all_results.filter(pl.col("root") == root).drop("root")
        evaluation_item = create_evaluation_item(cluster_df, source_configs, root)
        results_by_root[root] = evaluation_item

    return results_by_root


class EvalData:
    """Object which caches evaluation data to measure model performance."""

    def __init__(self) -> None:
        """Download judgement and expansion data used to compute evaluation metrics."""
        self.judgements, self.expansion = _handler.download_eval_data()

    def precision_recall(
        self, results: Results, threshold: float
    ) -> tuple[float, float]:
        """Compute precision and recall for a given Results object."""
        if not len(results.clusters):
            raise ValueError("No clusters suggested by these results.")

        threshold = int(threshold * 100)
        root_leaf = results.root_leaf().rename({"root_id": "root", "leaf_id": "leaf"})
        values = precision_recall([root_leaf], self.judgements, self.expansion)[0]
        return values[0], values[1]


def compare_models(resolutions: list[ModelResolutionPath]) -> ModelComparison:
    """Compare metrics of models based on cached evaluation data."""
    return _handler.compare_models(resolutions)
