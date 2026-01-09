"""Objects representing the results of running a model client-side."""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, ParamSpec, Self, TypeVar

import polars as pl
from pydantic import ConfigDict

from matchbox.client.sources import Source
from matchbox.common.arrow import SCHEMA_RESULTS
from matchbox.common.hash import IntMap
from matchbox.common.logging import logger
from matchbox.common.transform import DisjointSet, to_clusters

if TYPE_CHECKING:
    from matchbox.client.dags import DAG
else:
    DAG = Any

T = TypeVar("T", bound=Hashable)
P = ParamSpec("P")
R = TypeVar("R")


class ModelResults:
    """Results of a model run.

    Contains:

    * The probabilities of each pair being a match
    * (Optional) The clusters of connected components at each threshold
    * (Optional) The leaf_id mapping to trace results back to source clusters

    Allows users to easily interrogate the outputs of models, explore decisions on
    choosing thresholds for clustering, and upload the results to Matchbox.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    probabilities: pl.DataFrame
    _clusters: pl.DataFrame | None = None

    def __init__(
        self,
        probabilities: pl.DataFrame,
        left_root_leaf: pl.DataFrame | None = None,
        right_root_leaf: pl.DataFrame | None = None,
    ) -> None:
        """Initialises and validates results.

        Args:
            probabilities: dataframe with SCHEMA_RESULTS
            left_root_leaf: optional dataframe with columns: id, leaf_id
            right_root_leaf: optional dataframe with columns: id, leaf_id
        """
        self.left_root_leaf = None
        self.right_root_leaf = None

        if left_root_leaf is not None:
            self.left_root_leaf = left_root_leaf
        if right_root_leaf is not None:
            self.right_root_leaf = right_root_leaf

        if not isinstance(probabilities, pl.DataFrame):
            raise ValueError(f"Expected a polars DataFrame, got {type(probabilities)}.")

        expected_fields = set(SCHEMA_RESULTS.names)
        if set(probabilities.columns) != expected_fields:
            raise ValueError(
                f"Expected {expected_fields}.\nFound {set(probabilities.column_names)}."
            )

        # Handle empty tables
        if probabilities.height == 0:
            probabilities = pl.DataFrame(schema=pl.Schema(SCHEMA_RESULTS))

        unique_probabilities = (
            probabilities.with_columns(
                pl.concat_list(
                    [pl.col("left_id").cast(pl.Utf8), pl.col("right_id").cast(pl.Utf8)]
                )
                .list.sort()
                .list.join("_")
                .alias("sorted_ids")
            )
            .sort(
                "probability", descending=True
            )  # sort so largest probability comes first
            .unique(
                subset=["sorted_ids"], keep="first"
            )  # keep first occurrence after sorting
        ).drop("sorted_ids")
        if len(probabilities) != len(unique_probabilities):
            logger.warning(
                "Duplicate pairs! Keeping only pairs with highest probability."
            )

        # Process probability field if it contains floating-point or decimal values
        probability_type = unique_probabilities["probability"].dtype
        if probability_type.is_float() or probability_type.is_decimal():
            probability_uint8 = pl.Series(
                unique_probabilities.select(
                    pl.col("probability").mul(100).round(0).cast(pl.UInt8)
                )
            )

            # Check max value only if the table is not empty
            max_prob = probability_uint8.max()
            if max_prob is not None and max_prob > 100:
                p_max = max_prob
                p_min = probability_uint8.min()
                raise ValueError(f"Probability range misconfigured: [{p_min}, {p_max}]")

            unique_probabilities = unique_probabilities.replace_column(
                unique_probabilities.get_column_index("probability"), probability_uint8
            )

        # Need schema in format recognised by polars
        self.probabilities = unique_probabilities.cast(pl.Schema(SCHEMA_RESULTS))

    @property
    def clusters(self) -> pl.DataFrame:
        """Retrieve new clusters implied by these results."""
        if self._clusters is None:
            im = IntMap()
            self._clusters = to_clusters(
                results=self.probabilities, dtype=pl.Int64, hash_func=im.index
            )
        return self._clusters

    def root_leaf(self) -> pl.DataFrame:
        """Returns all roots and leaves implied by these results."""
        if self.left_root_leaf is None:
            raise RuntimeError(
                "This Results object wasn't instantiated for validation features."
            )

        parents_root_leaf = self.left_root_leaf.select(["id", "leaf_id"])
        if self.right_root_leaf is not None:
            parents_root_leaf = pl.concat(
                [
                    parents_root_leaf,
                    self.right_root_leaf.select(["id", "leaf_id"]),
                ]
            )

        # Go from parent-child (where child could be the root of another model)
        # to root-leaf, where leaf is a source cluster ID
        root_leaf_res = (
            self.clusters.rename({"parent": "root_id"})
            .join(parents_root_leaf, left_on="child", right_on="id")
            .select(["root_id", "leaf_id"])
            .unique()
        )

        # Generate root-leaf for those input rows that weren't merged by this model
        unmerged_ids_rows = (
            parents_root_leaf.select("id", "leaf_id")
            .join(
                self.clusters.select("child"),
                left_on="id",
                right_on="child",
                how="anti",
            )
            .rename({"id": "root_id"})
            .select(["root_id", "leaf_id"])
            .unique()
        )

        return pl.concat([root_leaf_res, unmerged_ids_rows])


class ResolvedMatches:
    """Matches according to resolution."""

    def __init__(
        self, sources: list[Source], query_results: list[pl.DataFrame]
    ) -> None:
        """Initialise resolved data.

        Args:
            sources: List of Source objects
            query_results: List of tables with SCHEMA_QUERY_WITH_LEAVES
        """
        self.sources = sources
        self.query_results = query_results

        if not len(sources):
            raise ValueError("At least 1 source must be resolved.")

        if len(sources) != len(query_results):
            raise ValueError("Mismatched length of sources and query results.")

    @classmethod
    def from_dump(cls, cluster_key_map: pl.DataFrame, dag: DAG) -> Self:
        """Initialise ResolvedMatches from concatenated dataframe representation."""
        partitioned = cluster_key_map.partition_by("source")
        sources = [dag.get_source(p["source"][0]) for p in partitioned]
        query_results = [p.drop("source") for p in partitioned]

        return ResolvedMatches(sources=sources, query_results=query_results)

    def as_lookup(self) -> pl.DataFrame:
        """Return lookup across matchbox ID and source keys."""
        lookup = (
            self.query_results[0]
            .rename({"key": self.sources[0].config.qualified_key(self.sources[0].name)})
            .drop("leaf_id")
        )

        if len(self.sources) > 1:
            for source, source_results in zip(
                self.sources[1:], self.query_results[1:], strict=True
            ):
                lookup = (
                    lookup.join(source_results, on="id", how="full")
                    .with_columns(pl.coalesce(["id", "id_right"]).alias("id"))
                    .drop(["id_right"])
                )

                lookup = lookup.rename(
                    {"key": source.config.qualified_key(source.name)}
                ).drop("leaf_id")

        return lookup

    def as_dump(self) -> pl.DataFrame:
        """Return mapping across root, leaf, source and keys."""
        concat_dfs = []
        for source, query_res in zip(self.sources, self.query_results, strict=True):
            source_col = pl.lit(source.name).alias("source")
            df_with_source = query_res.with_columns([source_col])
            concat_dfs.append(df_with_source)
        return pl.DataFrame(pl.concat(concat_dfs))

    def as_leaf_sets(self) -> list[list[int]]:
        """Return grouping of lead IDs."""
        cluster_key_map = self.as_dump()
        groups = cluster_key_map.group_by("id").agg("leaf_id")["leaf_id"].to_list()
        return [sorted(set(g)) for g in groups]

    def view_cluster(self, cluster_id: int, merge_fields: bool = False) -> pl.DataFrame:
        """Return source data for all records in cluster.

        Args:
            cluster_id: ID of root cluster to view
            merge_fields: whether to remove source qualifier when concatenating rows.
                Only applies to index fields - key fields are not affected.
        """
        cluster_rows = []
        key_cols = []
        for source, query_res in zip(self.sources, self.query_results, strict=True):
            # For each source, get rows for selected cluster
            source_keys = query_res.filter(pl.col("id") == cluster_id)["key"].to_list()
            if not source_keys:
                continue

            key_cols.append(source.qualified_key)
            # Determine column names of output dataframe
            rename_keys = {source.key_field.name: source.qualified_key}
            if not merge_fields:
                rename_index_fields = {
                    field.name: source.qualify_field(field.name)
                    for field in source.index_fields
                }
                rename_dict = {**rename_keys, **rename_index_fields}
            else:
                rename_dict = rename_keys

            # Fetch data for this source
            source_data = pl.concat(
                source.fetch(keys=source_keys, qualify_names=False)
            ).rename(rename_dict)

            cluster_rows.append(source_data)

        # Coerce fields to their common super-type
        if not cluster_rows:
            raise KeyError(f"Cluster {cluster_id} not available")
        source_concat = pl.concat(cluster_rows, how="diagonal_relaxed")
        # Re-order columns to have keys at the beginning
        remaining_cols = [col for col in source_concat.columns if col not in key_cols]

        return source_concat.select(*key_cols, *remaining_cols)

    def merge(self, other: Self) -> Self:
        """Combine two instances of resolved matches by merging clusters.

        All cluster IDs will be replaced with negative integers and lose their
        association with cluster IDs on the backend.
        """
        if other.sources != self.sources:
            raise ValueError("Cannot merge resolved matches for different sources")

        djs = DisjointSet[int]()

        # Use disjoint sets to find new clusters
        for resolved_instance in [self, other]:
            unioned_root_leaf = pl.concat(
                [
                    result.select(["id", "leaf_id"])
                    for result in resolved_instance.query_results
                ]
            )
            components = (
                unioned_root_leaf.group_by("id")
                .agg(pl.col("leaf_id"))["leaf_id"]
                .to_list()
            )
            # Create component by connecting one leaf to all others within component
            for leaf_set in components:
                djs.add(leaf_set[0])
                for other_leaf in leaf_set[1:]:
                    djs.union(leaf_set[0], other_leaf)

        # Turn new clusters into dataframe
        new_components = []
        for i, component in enumerate(djs.get_components(), start=1):
            cluster_id = -i
            for leaf_id in component:
                new_components.append({"id": cluster_id, "leaf_id": leaf_id})

        new_components_df = pl.DataFrame(new_components)

        # Generate new combined query_results based on new clusters
        new_query_results = []
        for self_result, other_result in zip(
            self.query_results, other.query_results, strict=True
        ):
            unioned_leaf_key = pl.concat(
                [
                    self_result.select("leaf_id", "key"),
                    other_result.select("leaf_id", "key"),
                ]
            ).unique()
            source_query_results = unioned_leaf_key.join(
                new_components_df, on="leaf_id"
            )
            new_query_results.append(source_query_results)

        return ResolvedMatches(sources=self.sources, query_results=new_query_results)
