"""Client-side helpers for retrieving and preparing evaluation samples."""

import polars as pl
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

from matchbox.client import _handler
from matchbox.client.dags import DAG
from matchbox.client.results import ModelResults
from matchbox.common.dtos import ModelResolutionName, ModelResolutionPath, SourceConfig
from matchbox.common.eval import Judgement, precision_recall
from matchbox.common.exceptions import MatchboxSourceTableError


class EvaluationFieldMetadata(BaseModel):
    """Metadata for a field in evaluation."""

    display_name: str
    source_columns: list[str]


class EvaluationItem(BaseModel):
    """A cluster ready for evaluation.

    The records dataframe contains the leaf IDs and the qualified index fields
    associated with it. For example:

    | leaf_id | src_a_first | src_a_last | src_b_first | src_b_last |
    |---------|-------------|------------|-------------|------------|
    | 1       | Thomas      | Bayes      |             |            |
    | 2       | Tommy       | B          |             |            |
    | 12      |             |            | Tom         | Bayes      |

    The fields attribute allows any evaluation system to map between a display
    version of the source columns, and the actual columns contained in the
    records dataframe. For example:

    ```text
    {
        "display_name": "first",
        "source_columns": "src_a_first", "src_b_first"
    }
    ```
    """

    model_config = {"arbitrary_types_allowed": True}

    leaves: list[int]
    records: pl.DataFrame
    fields: list[EvaluationFieldMetadata]

    def get_unique_record_groups(self) -> list[list[int]]:
        """Group identical records by leaf ID.

        Returns:
            List of groups, where each group is a list of leaf IDs
            that have identical values across all data fields.
            Example: [[1, 3], [2], [4, 5, 6]] means records 1 & 3 are identical.
        """
        # Get all data column names (not "leaf")
        # Flatten the source_columns lists from all fields
        data_cols = [col for field in self.fields for col in field.source_columns]

        # Group by all data columns to find duplicates
        grouped = self.records.group_by(data_cols, maintain_order=True).agg(
            pl.col("leaf")
        )

        # Extract list of leaf ID lists
        return [group for group in grouped["leaf"]]


def create_judgement(
    item: EvaluationItem,
    assignments: dict[int, str],
    user_name: str,
    tag: str | None = None,
) -> Judgement:
    """Convert item assignments to Judgement - no default group assignment.

    Args:
        item: evaluation item
        assignments: column assignments (group_idx -> group_letter)
        user_name: user name for the judgement
        tag: string by which to tag the judgement

    Returns:
        Judgement with endorsed groups based on assignments
    """
    groups: dict[str, list[int]] = {}
    unique_record_groups = item.get_unique_record_groups()

    for col_idx, group in assignments.items():
        leaf_ids = unique_record_groups[col_idx]
        groups.setdefault(group, []).extend(leaf_ids)

    endorsed = [sorted(set(leaf_ids)) for leaf_ids in groups.values()]
    return Judgement(user_name=user_name, shown=item.leaves, endorsed=endorsed, tag=tag)


def create_evaluation_item(
    df: pl.DataFrame, source_configs: list[tuple[str, SourceConfig]], leaves: list[int]
) -> EvaluationItem:
    """Create EvaluationItem with structured metadata."""
    # Get all data columns (exclude metadata columns)
    data_cols = [c for c in df.columns if c not in ["root", "leaf", "key"]]

    # Build mapping of field_name -> list of qualified column names
    field_to_columns: dict[str, list[str]] = {}

    for source_id, config in source_configs:
        for field in config.index_fields:
            # Build qualified column name for this source+field
            column_name = config.qualify_field(source_id, field.name)

            # Only add if this column exists in DataFrame
            if column_name in data_cols:
                # Add to list for this field name
                if field.name not in field_to_columns:
                    field_to_columns[field.name] = []
                field_to_columns[field.name].append(column_name)

    # Create EvaluationFieldMetadata objects (one per unique field name)
    fields: list[EvaluationFieldMetadata] = []
    for field_name, source_columns in field_to_columns.items():
        fields.append(
            EvaluationFieldMetadata(
                display_name=field_name, source_columns=source_columns
            )
        )

    # Keep ALL data columns in records
    records = df.select(["leaf"] + data_cols)

    return EvaluationItem(leaves=leaves, records=records, fields=fields)


def _read_sample_file(sample_file: str, n: int) -> pl.DataFrame:
    resolved_matches_dump = pl.read_parquet(sample_file)
    clusters = resolved_matches_dump["id"].unique()
    select_clusters = clusters.sample(min(n, len(clusters)), shuffle=True).to_list()
    select_rows = resolved_matches_dump.filter(pl.col("id").is_in(select_clusters))
    return select_rows.rename({"id": "root", "leaf_id": "leaf"})


def _get_samples_from_server(
    dag: DAG, user_name: str, n: int, resolution: ModelResolutionName | None = None
) -> pl.DataFrame:
    if resolution:
        resolution_path: ModelResolutionPath = dag.get_model(resolution).resolution_path
    else:
        resolution_path: ModelResolutionPath = dag.final_step.resolution_path
    return pl.from_arrow(
        _handler.sample_for_eval(n=n, resolution=resolution_path, user_name=user_name)
    )


def get_samples(
    n: int,
    dag: DAG,
    user_name: str,
    resolution: ModelResolutionName | None = None,
    sample_file: str | None = None,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data as EvaluationItems.

    Args:
        n: Number of clusters to sample
        dag: DAG for which to retrieve samples
        user_name: Name of the user requesting the samples
        resolution: The optional resolution from which to sample. If not set, the final
            step in the DAG is used. If sample_file is set, resolution is ignored
        sample_file: path to parquet file output by ResolvedMatches. If specified,
            won't sample from server, ignoring the user_name and resolution arguments

    Returns:
        Dictionary of cluster ID to EvaluationItems describing the cluster

    Raises:
        MatchboxSourceTableError: If a source cannot be queried from a location using
            provided or default clients.
    """
    if sample_file:
        samples = _read_sample_file(sample_file=sample_file, n=n)
    else:
        samples = _get_samples_from_server(
            dag=dag, user_name=user_name, n=n, resolution=resolution
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
        leaves = cluster_df.select("leaf").to_series().unique().to_list()
        evaluation_item = create_evaluation_item(cluster_df, source_configs, leaves)
        results_by_root[root] = evaluation_item

    return results_by_root


class EvalData:
    """Object which caches evaluation data to measure model performance."""

    def __init__(self, tag: str | None = None) -> None:
        """Download judgement and expansion data used to compute evaluation metrics."""
        self.tag = tag
        self.judgements, self.expansion = _handler.download_eval_data(tag)

    def precision_recall(
        self, results: ModelResults, threshold: float
    ) -> tuple[float, float]:
        """Compute precision and recall for a given Results object."""
        if not len(results.clusters):
            raise ValueError("No clusters suggested by these results.")

        threshold = int(threshold * 100)
        root_leaf = results.root_leaf().rename({"root_id": "root", "leaf_id": "leaf"})
        values = precision_recall([root_leaf], self.judgements, self.expansion)[0]
        return values[0], values[1]
