"""Collection of client-side functions in aid of model evaluation."""

import warnings
from typing import Any

import polars as pl
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from matchbox.client import _handler
from matchbox.client._settings import settings
from matchbox.client.results import Results
from matchbox.common.eval import (
    Judgement,
    ModelComparison,
    PrecisionRecall,
    precision_recall,
)
from matchbox.common.exceptions import MatchboxSourceTableError
from matchbox.common.graph import DEFAULT_RESOLUTION, ModelResolutionName
from matchbox.common.logging import logger
from matchbox.common.sources import SourceConfig

try:
    from pydantic import BaseModel, computed_field
except ImportError:
    BaseModel = object

    def computed_field():
        """Fallback decorator when Pydantic is not available."""

        def decorator(func):
            return func

        return decorator


class EvaluationItem(BaseModel):
    """A cluster awaiting evaluation, with deduplicated column data."""

    model_config = {"arbitrary_types_allowed": True}

    cluster_id: int
    dataframe: pl.DataFrame  # Original raw data
    display_dataframe: pl.DataFrame  # Enhanced DataFrame for flexible rendering
    duplicate_groups: list[list[int]]  # Groups of leaf_ids with identical data
    display_columns: list[int]  # Representative leaf_id for each displayed column
    leaf_to_display_mapping: dict[int, int]  # actual leaf_id -> display_column_index
    assignments: dict[int, str] = {}  # display_column_index -> group_letter

    @computed_field
    @property
    def total_columns(self) -> int:
        """Total number of display columns in this cluster."""
        return len(self.display_columns)

    @computed_field
    @property
    def is_painted(self) -> bool:
        """True only if ALL columns have been assigned to groups."""
        return len(self.assignments) == self.total_columns

    def to_judgement(self, user_id: int) -> Judgement:
        """Convert assignments to Judgement format, expanding duplicate groups."""
        groups = {}

        # Expand display column assignments to all underlying leaf IDs
        for display_col_index, group in self.assignments.items():
            if group not in groups:
                groups[group] = []

            # Get all leaf IDs for this display column (including duplicates)
            if display_col_index < len(self.duplicate_groups):
                duplicate_group = self.duplicate_groups[display_col_index]
                groups[group].extend(duplicate_group)

        # Handle unassigned display columns - expand to all their leaf IDs
        assigned_display_cols = set(self.assignments.keys())
        unassigned_leaf_ids = []

        for display_col_index in range(len(self.duplicate_groups)):
            if display_col_index not in assigned_display_cols:
                unassigned_leaf_ids.extend(self.duplicate_groups[display_col_index])

        if unassigned_leaf_ids:
            groups.setdefault("a", []).extend(unassigned_leaf_ids)

        # Convert to endorsed format, ensuring no duplicate groups
        endorsed = []
        seen_groups = set()
        for group_items in groups.values():
            # Remove duplicates within the group and sort for consistent comparison
            unique_items = sorted(set(group_items))
            group_tuple = tuple(unique_items)
            # Only add if we haven't seen this exact group before
            if group_tuple not in seen_groups:
                endorsed.append(unique_items)
                seen_groups.add(group_tuple)

        return Judgement(
            user_id=user_id,
            shown=self.cluster_id,
            endorsed=endorsed,
        )


def create_display_dataframe(
    df: pl.DataFrame, source_configs: list[SourceConfig]
) -> pl.DataFrame:
    """Create enhanced display DataFrame for flexible view rendering.

    Args:
        df: DataFrame with records as rows and qualified fields as columns
        source_configs: List of SourceConfig objects that generated the qualified fields

    Returns:
        DataFrame with columns: field_name, source_name, record_index, value, leaf_id
    """
    # Get leaf IDs for records
    leaf_ids = (
        df.select("leaf").to_series().to_list()
        if "leaf" in df.columns
        else list(range(len(df)))
    )

    rows = []

    # Iterate through each record in the DataFrame
    for record_idx, record in enumerate(df.iter_rows(named=True)):
        leaf_id = leaf_ids[record_idx]

        # For each source config, extract its fields
        for source_config in source_configs:
            source_name = source_config.name

            for field in source_config.index_fields:
                qualified_field = source_config.f(field.name)
                unqualified_field = field.name

                # Only include fields that exist in the DataFrame
                if qualified_field in record:
                    value = record[qualified_field]
                    # Only include non-null, non-empty values
                    if value is not None:
                        str_value = str(value).strip()
                        if str_value:  # Only add non-empty strings
                            rows.append(
                                {
                                    "field_name": unqualified_field,
                                    "source_name": source_name,
                                    "record_index": record_idx,
                                    "value": str_value,
                                    "leaf_id": leaf_id,
                                }
                            )

    # Create DataFrame from collected rows
    if rows:
        return pl.DataFrame(rows)
    else:
        # Return empty DataFrame with correct schema
        return pl.DataFrame(
            {
                "field_name": [],
                "source_name": [],
                "record_index": [],
                "value": [],
                "leaf_id": [],
            },
            schema={
                "field_name": pl.String,
                "source_name": pl.String,
                "record_index": pl.Int32,
                "value": pl.String,
                "leaf_id": pl.Int32,
            },
        )


class DeduplicationResult:
    """Result of column deduplication analysis."""

    def __init__(
        self,
        duplicate_groups: list[list[int]],
        display_columns: list[int],
        leaf_to_display_mapping: dict[int, int],
    ):
        """Initialize deduplication result."""
        self.duplicate_groups = duplicate_groups
        self.display_columns = display_columns
        self.leaf_to_display_mapping = leaf_to_display_mapping


def deduplicate_columns(display_df: pl.DataFrame) -> DeduplicationResult:
    """Analyze columns for duplicates and create deduplication mapping.

    Args:
        display_df: Enhanced display DataFrame

    Returns:
        DeduplicationResult with duplicate groupings and mappings
    """
    if display_df.is_empty():
        return DeduplicationResult([], [], {})

    # Get all unique leaf_ids (columns)
    leaf_ids = sorted(display_df["leaf_id"].unique().to_list())

    if not leaf_ids:
        return DeduplicationResult([], [], {})

    # Create column signatures by hashing all field values for each column
    column_signatures = {}

    for leaf_id in leaf_ids:
        # Get all data for this column across all fields
        column_data = (
            display_df.filter(pl.col("leaf_id") == leaf_id)
            .sort("field_name")  # Consistent ordering
            .select(["field_name", "value"])
        )

        # Create a signature from all field-value pairs
        if not column_data.is_empty():
            # Convert to sorted list of (field, value) pairs for consistent hashing
            field_value_pairs = [
                (row["field_name"], row["value"])
                for row in column_data.iter_rows(named=True)
            ]
            signature = tuple(sorted(field_value_pairs))
        else:
            signature = ()  # Empty column

        column_signatures[leaf_id] = signature

    # Group columns by identical signatures
    signature_to_leaves = {}
    for leaf_id, signature in column_signatures.items():
        if signature not in signature_to_leaves:
            signature_to_leaves[signature] = []
        signature_to_leaves[signature].append(leaf_id)

    # Build deduplication structures
    duplicate_groups = []
    display_columns = []
    leaf_to_display_mapping = {}

    display_col_index = 0
    for leaf_group in signature_to_leaves.values():
        # Sort leaf IDs for consistent ordering
        leaf_group = sorted(leaf_group)

        # First leaf ID in group becomes the display representative
        representative_leaf = leaf_group[0]

        duplicate_groups.append(leaf_group)
        display_columns.append(representative_leaf)

        # Map all leaves in this group to the same display column index
        for leaf_id in leaf_group:
            leaf_to_display_mapping[leaf_id] = display_col_index

        display_col_index += 1

    return DeduplicationResult(
        duplicate_groups, display_columns, leaf_to_display_mapping
    )


def create_evaluation_item(
    df: pl.DataFrame, source_configs: list[SourceConfig], cluster_id: int
) -> EvaluationItem:
    """Create a complete EvaluationItem with deduplication.

    Args:
        df: DataFrame with records as rows and qualified fields as columns
        source_configs: List of SourceConfig objects
        cluster_id: The cluster ID for this evaluation item

    Returns:
        Complete EvaluationItem with deduplication applied
    """
    # Create enhanced display DataFrame
    display_dataframe = create_display_dataframe(df, source_configs)

    # Perform deduplication analysis
    dedup_result = deduplicate_columns(display_dataframe)

    return EvaluationItem(
        cluster_id=cluster_id,
        dataframe=df,
        display_dataframe=display_dataframe,
        duplicate_groups=dedup_result.duplicate_groups,
        display_columns=dedup_result.display_columns,
        leaf_to_display_mapping=dedup_result.leaf_to_display_mapping,
        assignments={},
    )


def get_samples(
    n: int,
    user_id: int,
    resolution: ModelResolutionName | None = None,
    clients: dict[str, Any] | None = None,
    use_default_client: bool = False,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data, grouped by resolution cluster.

    Args:
        n: Number of clusters to sample
        user_id: ID of the user requesting the samples
        resolution: Model resolution proposing the clusters. If not set, will
            use a default resolution.
        clients: Dictionary from location names to valid client for each.
            Locations whose name is missing from the dictionary will be skipped.
        use_default_client: Whether to use for all unset location clients
            a SQLAlchemy engine for the default warehouse set in the environment
            variable `MB__CLIENT__DEFAULT_WAREHOUSE`.

    Returns:
        Dictionary of cluster ID to EvaluationItem with processed field data

    Raises:
        MatchboxSourceTableError: If a source cannot be queried from a location using
            provided or default clients.
    """
    if not resolution:
        resolution = DEFAULT_RESOLUTION

    if not clients:
        clients = {}

    default_client = None
    if use_default_client:
        if default_clients_uri := settings.default_warehouse:
            default_client = create_engine(default_clients_uri)
            logger.warning("Using default engine")
        else:
            raise ValueError("`MB__CLIENT__DEFAULT_WAREHOUSE` is unset")

    try:
        samples: pl.DataFrame = pl.from_arrow(
            _handler.sample_for_eval(n=n, resolution=resolution, user_id=user_id)
        )

        if not len(samples):
            return {}

        results_by_source = []
        source_configs = []
        for source_resolution in samples["source"].unique():
            source_config = _handler.get_source_config(source_resolution)
            source_configs.append(source_config)
            location_name = source_config.location.name
            if location_name in clients:
                source_config.location.add_client(client=clients[location_name])
            elif default_client:
                source_config.location.add_client(client=default_client)
            else:
                warnings.warn(
                    f"Skipping {source_resolution}, incompatible with given client.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            samples_by_source = samples.filter(pl.col("source") == source_resolution)
            keys_by_source = samples_by_source["key"].to_list()

            try:
                source_data = pl.concat(
                    source_config.query(
                        batch_size=10_000, qualify_names=True, keys=keys_by_source
                    )
                )
            except OperationalError as e:
                raise MatchboxSourceTableError(
                    "Could not find source using given client."
                ) from e

            samples_and_source = samples_by_source.join(
                source_data, left_on="key", right_on=source_config.qualified_key
            )
            desired_columns = ["root", "leaf", "key"] + source_config.qualified_fields
            results_by_source.append(samples_and_source[desired_columns])

        if not len(results_by_source):
            return {}

        all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

        results_by_root = {}
        for root in all_results["root"].unique():
            cluster_df = all_results.filter(pl.col("root") == root).drop("root")

            # Create EvaluationItem with deduplication
            evaluation_item = create_evaluation_item(
                cluster_df, source_configs, int(root)
            )

            results_by_root[int(root)] = evaluation_item

        return results_by_root
    finally:
        # Always dispose of the default client if we created one
        if default_client:
            default_client.dispose()


class EvalData:
    """Object which caches evaluation data to measure performance of models."""

    def __init__(self):
        """Initialise evaluation data from resolution name."""
        self.judgements, self.expansion = _handler.download_eval_data()

    def precision_recall(self, results: Results, threshold: float) -> PrecisionRecall:
        """Computes precision and recall at one threshold."""
        if not len(results.clusters):
            raise ValueError("No clusters suggested by these results.")

        threshold = int(threshold * 100)

        root_leaf = (
            results.root_leaf()
            .rename({"root_id": "root", "leaf_id": "leaf"})
            .to_arrow()
        )
        return precision_recall([root_leaf], self.judgements, self.expansion)[0]

    def pr_curve(self, results: Results) -> Figure:
        """Computes precision and recall for each threshold in results."""
        all_p = []
        all_r = []

        probs = pl.from_arrow(results.probabilities)
        thresholds = probs.select("probability").unique().to_series()
        for i, t in enumerate(sorted(thresholds)):
            float_thresh = t / 100
            p, r = self.precision_recall(results=results, threshold=float_thresh)
            all_p.append(p)
            all_r.append(r)
            plt.annotate(float_thresh, (all_r[i], all_p[i]))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(all_r, all_p, marker="o")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        ax.set_title("Precision-Recall Curve")
        ax.grid()

        return fig


def compare_models(resolutions: list[ModelResolutionName]) -> ModelComparison:
    """Compare metrics of models based on evaluation data.

    Args:
        resolutions: List of names of model resolutions to be compared.

    Returns:
        A model comparison object, listing metrics for each model.
    """
    return _handler.compare_models(resolutions)
