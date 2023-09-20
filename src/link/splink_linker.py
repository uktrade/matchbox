from src.link.linker import Linker
from src.data import utils as du
from src.data.datasets import Dataset
from src.data.probabilities import Probabilities
from src.data.clusters import Clusters

from splink.duckdb.linker import DuckDBLinker
from splink.comparison import Comparison

import json
import pandas as pd


class ComparisonEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return obj.__name__
        elif isinstance(obj, Comparison):
            return obj.as_dict()
        else:
            return json.JSONEncoder.default(self, obj)


class SplinkLinker(Linker):
    """
    A class to handle linking a dataset using Splink. Implements linking
    with DuckDB.

    Uses an internal lookup table unique_id_lookup to minimise memory
    usage during linking. Will create this during a job, and re-join the
    correct data back on afterwards.

    Parameters:
        * dataset: An object of class Dataset
        * probabilities: An object of class Probabilities
        * clusters: An object of class Clusters
        * n: The current step in the pipeline process
        * db_path: [Optional] If writing to disk, the location to use
        for duckDB

    Methods:
        * get_data(): retrieves the left and right tables: clusters
        and dimensions
        * prepare(linker_settings, cluster_pipeline, dim_pipeline,
        train_pipeline): cleans the data using a data processing dict,
        creates a linker with linker_settings, then trains it with a
        train_pipeline dict
        * link(threshold): performs linking and returns a match table
        appropriate for Probabilities. Drops observations below the specified
        threshold
        * evaluate(): runs prepare() and link() and returns a report of
        their performance
    """

    def __init__(
        self,
        dataset: Dataset,
        probabilities: Probabilities,
        clusters: Clusters,
        n: int,
        db_path: str = ":memory:",
    ):
        super().__init__(dataset, probabilities, clusters, n)

        self.linker = None
        self.linker_settings = None
        self.db_path = db_path
        self.con = du.get_duckdb_connection(path=self.db_path)
        self.id_lookup = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Only pickle linker settings
        state["linker"] = state["linker"]._settings_obj.as_dict()
        # Don't pickle connection
        del state["con"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add connection and linker back when loading pickle
        self.con = du.get_duckdb_connection(path=self.db_path)
        if self.cluster_processed is not None and self.dim_processed is not None:
            self._register_tables()
        if self.linker is not None:
            self._create_linker(linker_settings=self.linker)

    def _clean_data(self, cluster_pipeline: dict, dim_pipeline: dict):
        self.cluster_processed = super()._run_pipeline(
            self.cluster_raw, cluster_pipeline
        )
        self.dim_processed = super()._run_pipeline(self.dim_raw, dim_pipeline)

    def _substitute_ids(self):
        cls_len = self.cluster_processed.shape[0]
        dim_len = self.dim_processed.shape[0]

        self.cluster_processed["duckdb_id"] = range(cls_len)
        self.dim_processed["duckdb_id"] = range(cls_len, cls_len + dim_len)

        self.id_lookup = pd.concat(
            objs=[
                self.cluster_processed[["duckdb_id", "id"]],
                self.dim_processed[["duckdb_id", "id"]],
            ],
            axis=0,
        )
        self.cluster_processed["id"] = self.cluster_processed["duckdb_id"]
        self.cluster_processed.drop("duckdb_id", axis=1, inplace=True)
        self.dim_processed["id"] = self.dim_processed["duckdb_id"]
        self.dim_processed.drop("duckdb_id", axis=1, inplace=True)

    def _register_tables(self):
        self.con.register("cls", self.cluster_processed)
        self.con.register("dim", self.dim_processed)

    def _create_linker(self, linker_settings: dict):
        self.linker = DuckDBLinker(
            input_table_or_tables=["cls", "dim"],
            input_table_aliases=["cls", "dim"],
            connection=self.con,
            settings_dict=linker_settings,
        )
        self.linker_settings = self.linker._settings_obj.as_dict()

    def _train_linker(self, train_pipeline: dict):
        """
        Runs the pipeline of linker functions to train the linker object.

        Similar to _run_pipeline(), expects self.pipeline to be a dict
        of step keys with a value of a dict with "function" and "argument"
        keys. Here, however, the value of "function" should be a string
        corresponding to a method in the linker object. "argument"
        remains the same: a dictionary of method and value arguments to
        the referenced linker method.
        """
        for func in train_pipeline.keys():
            proc_func = getattr(self.linker, train_pipeline[func]["function"])
            proc_func(**train_pipeline[func]["arguments"])

        train_pipeline_json = json.dumps(
            train_pipeline, indent=4, cls=ComparisonEncoder
        )

        super()._add_log_item(
            name="train_pipeline",
            item=train_pipeline_json.encode(),
            item_type="artefact",
            path="config/train_pipeline.json",
        )

        model_json = json.dumps(
            self.linker._settings_obj.as_dict(), indent=4, cls=ComparisonEncoder
        )

        super()._add_log_item(
            name="model",
            item=model_json.encode(),
            item_type="artefact",
            path="model/model.json",
        )

    def prepare(
        self,
        cluster_pipeline: dict,
        dim_pipeline: dict,
        linker_settings: dict,
        train_pipeline: dict,
    ):
        self._clean_data(cluster_pipeline, dim_pipeline)
        self._substitute_ids()
        self._register_tables()
        self._create_linker(linker_settings)
        self._train_linker(train_pipeline)

    def link(self, threshold: int, log_output: bool = True):
        predictions = None

        if log_output:
            self.probabilities.add_probabilities(predictions)

        return predictions
