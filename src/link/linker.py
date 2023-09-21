from src.data import utils as du
from src.link import model_utils as mu
from src.data.datasets import Dataset
from src.data.probabilities import Probabilities
from src.data.clusters import Clusters

import mlflow
import logging
from pathlib import Path
import pickle
import io
from abc import ABC, abstractmethod
import json
from tempfile import NamedTemporaryFile


class Linker(ABC):
    """
    A class to build attributes and methods shared by all Linker subclasses.
    Standardises:

    * Retrieving the left table: cluster data
    * Retrieving the right table: dimension data
    * The structure of the core linking methods: prepare() and link()
    * The output shape and location
    * The evaluation and reporting methodology

    Assumes a Linker subclass will be instantiated with:

    * Dicts of settings that configure its process
    * A single step in the link_pipeline in config.py

    Parameters:
        * dataset: An object of class Dataset
        * probabilities: An object of class Probabilities
        * clusters: An object of class Clusters
        * n: The current step in the pipeline process

    Methods:
        * get_data(cluster_select, dim_select): retrieves the left and right
        tables: clusters and dimensions
        * prepare(): a method intended for linkers that need to clean data
        and train model parameters. Can output None to be skipped
        * link(): performs linking and returns a match table appropriate
        for Probabilities
        * evaluate(): runs prepare() and link() and returns a report of
        their performance
        * save(path): saves the linker object to a file as a pickle
        * load(path): loads an instance of a Linker class from a pickle file
    """

    def __init__(
        self, dataset: Dataset, probabilities: Probabilities, clusters: Clusters, n: int
    ):
        self.dataset = dataset
        self.probabilities = probabilities
        self.clusters = clusters
        self.n = n

        self.cluster_raw = None
        self.dim_raw = None

        self.cluster_processed = None
        self.dim_processed = None

        self.report_artefacts = {}
        self.report_parameters = {}
        self.report_metrics = {}

    def get_data(self, cluster_select: dict, dim_select: list, sample: float = None):
        """
        Returns the raw data from the cluster table and dimension table as per the
        dictionary specs of Clusters.get_data and Datasets.read_dim.

        The extra argument, sample, is the percentage of the tables to sample. It
        helps us run the pipeline quicker during development.
        """
        self.cluster_raw = self.clusters.get_data(
            cluster_select, cluster_uuid_to_id=True, n=self.n, sample=sample
        )
        self.dim_raw = self.dataset.read_dim(dim_select, sample)

    def _run_pipeline(self, table_in, pipeline):
        """
        Runs a pipeline of functions against an input table and
        returns the result.

        Arguments:
            table_in: The table to be processed
            pipeline: A dict with where each key is a pipeline step,
            and the value is another dict with keys "function" and
            "arguments". "function"'s value is a callable function
            that returns the input for the next step, and "arguments"
            should contain a dictionary of method and value arguments
            to the function.

        Returns:
            A output table with the pipeline of functions applied

        Examples:
            _run_pipeline(
                table_in = self.table,
                pipeline = {
                    "add_two_columns": {
                        "function": add_columns,
                        "arguments": {
                            "left_column": "q1_q2_profit",
                            "right_column": "q3_q4_profit",
                            "output": "year_profit"
                        }
                    },
                    "delete_columns": {
                        "function": delete_columns,
                        "arguments": {
                            "columns": ["q1_q2_profit", "q3_q4_profit"]
                        }
                    }
                }
            )
        """
        curr = table_in
        for func in pipeline.keys():
            curr = pipeline[func]["function"](curr, **pipeline[func]["arguments"])
        return curr

    @abstractmethod
    def prepare(self):
        """
        An optional method for functions like data cleaning and linker training.
        If you don't use it, must return False. If you do, must return True.

        All preprocessing should write the finalised data to self.dim_processed
        and self.cluster_processed.

        During a run, use the _add_log_item(item_type='artefact') method to record
        items you want evaluate() to save. Examples are plots, datasets or JSON
        objects.

        During a run, use the _add_log_item(item_type='parameter') method to record
        method parameters you want evaluate() to save. Examples are the
        Jaro-Winkler fuzzy matching value above which you consider something a
        match.

        _run_pipeline() is provided as a method to run pipelines of data cleaning
        using DuckDB and the functions in src/features.

        Returns
            Bool indicating whether code was run.
        """

        self.dim_processed = self.dim_raw
        self.cluster_processed = self.cluster_raw

        return False

    @abstractmethod
    def link(self, log_output: bool = True):
        """
        Runs whatever linking logic the subclass implements. Must finish by
        optionally calling Probabilities.add_probabilities(predictions), and then
        returning those predictions.

        Link jobs should take self.dim_processed and self.cluster_processed as
        their input.

        During a run, use the _add_log_item(item_type='artefact') method to record
        items you want evaluate() to save. Examples are plots, datasets or JSON
        objects.

        During a run, use the _add_log_item(item_type='parameter') method to record
        method parameters you want evaluate() to save. Examples are the
        Jaro-Winkler fuzzy matching value above which you consider something a
        match.

        Arguments:
            * log_output: whether to log outputs to the probabilities table
        """
        raise NotImplementedError("method link() must be implemented")

        predictions = None

        if log_output:
            self.probabilities.add_probabilities(
                probabilities=predictions, model=None, overwrite=False
            )

        return predictions

    def _add_log_item(
        self,
        name: str,
        item: object,
        item_type: str,
        path: str = None,
    ):
        """
        Adds an item to either the artefact, metric or parameter dictionary,
        ready to be recorded as part of a report in evaluate(). When using
        MLflow, this is attached to the run in the specified directory.

        Subclasses should not use item_type='metric', as all matching methods
        should be comparable.

        Arguments:
            name: the unique name the artifact will be keyed to
            path: [Optional] if saving an artefact, the relative path you want it
            saved in, including the name and file extension you want to use
            item: the object you want to save. Requires:
                * object, if item_type is 'artefact'
                * string, if item_type is 'parameter'
                * numeric, if item_type is 'metric'
            item_type: the type of item you're saving. One of 'artefact',
            'parameter' or 'metric'

        Raises:
            ValueError:
                * if one of 'artefact', 'parameter' or 'metric' not passed to
                item_type
                * if path not set when item_type is 'artefact'
            TypeError: if an unacceptable datatype is passed for the item_type
        """
        # TODO: prevent key duplication in same run, allow between runs

        if item_type not in ["artefact", "parameter", "metric"]:
            raise ValueError(
                """
                item_type must be one of 'artefact', 'parameter' or 'metric'
            """
            )

        if item_type == "artefact":
            if path is None:
                raise ValueError(
                    """
                    If item_type is 'artefact', must specify path
                """
                )
            self.report_artefacts[name] = {"path": path, "artefact": item}
        elif item_type == "parameter":
            if not isinstance(item, str):
                raise TypeError("Parameters must be logged as strings")
            self.report_parameters[name] = {"name": name, "value": item}
        elif item_type == "metric":
            if not isinstance(item, int):
                raise TypeError("Metrics must be logged as strings")
            self.report_metrics[name] = {"name": name, "value": item}

    def evaluate(
        self,
        link_experiment: str,
        evaluation_name: str,
        evaluation_description: str,
        prepare_kwargs: dict,
        link_kwargs: dict,
        report_dir: str = None,
        log_mlflow: bool = False,
        log_output: bool = False,
    ) -> dict:
        """
        Runs the prepare() and link() functions, and records evaluations.

        Arguments:
            * link_experiment: the experiment for the link, defined in config
            * evaluation_name: the name of this specific evaluation run
            * evaluation_description: a description of this specific
            evaluation run
            * prepare_kwargs: a dictionary of keyword arguments to pass to the
            child class's implemented prepare() method
            * link_kwargs: a dictionary of keyword arguments to pass to the
            child class's implemented link() method
            * report_dir: [optional] if not None, will write report parameters,
            metrics and artefacts to this directory
            * log_mlflow: whether to use MLflow to log this run
            * log_output: whether to log outputs to the probabilities table

        Returns:
            A dict of analysis
        """
        logging.basicConfig(
            level=logging.INFO,
            format=du.LOG_FMT,
        )
        logger = logging.getLogger(__name__)

        logger.info("Running pipeline")

        if log_output:
            logger.info("Logging outputs to the Probabilities table")

        if log_mlflow:
            logger.info("Logging as MLflow experiment")
            with mu.mlflow_run(
                experiment_name=link_experiment,
                run_name=evaluation_name,
                description=evaluation_description,
                dev_mode=True,
            ):
                logger.info("Running prepare() function")
                self.prepare(**prepare_kwargs)

                logger.info("Running link() function")
                self.link(log_output=log_output, **link_kwargs)

                # TODO: Evaluation method based on validation table
                # Table not yet implemented

                for artefact in self.report_artefacts.keys():
                    path = Path(self.report_artefacts[artefact]["path"])
                    artefact = self.report_artefacts[artefact]["artefact"]
                    artefact_io = io.BytesIO(artefact)

                    with NamedTemporaryFile(
                        suffix=path.suffix, prefix=f"{path.stem}_"
                    ) as f:
                        f.write(artefact_io.read())
                        mlflow.log_artifact(
                            local_path=f.name, artifact_path=path.parent.as_posix()
                        )

                for param in self.report_parameters.keys():
                    mlflow.log_param(
                        key=self.report_parameters[param]["name"],
                        value=self.report_parameters[param]["value"],
                    )

                for metric in self.report_metrics.keys():
                    mlflow.log_metric(
                        key=self.report_metrics[metric]["name"],
                        value=self.report_metrics[metric]["value"],
                    )

                # TODO: Make dict of outputs to return

        else:
            logger.info("Experiment not automatically logged on MLFlow")

            logger.info("Running prepare() function")
            self.prepare(**prepare_kwargs)

            logger.info("Running link() function")
            self.link(log_output=log_output, **link_kwargs)

        if report_dir is not None:
            logger.info(f"Writing parameters to {report_dir}")
            with open(Path(report_dir, "parameters.json"), "w") as f:
                json.dump(self.report_parameters, f)

            logger.info(f"Writing metrics to {report_dir}")
            with open(Path(report_dir, "metrics.json"), "w") as f:
                json.dump(self.report_metrics, f)

            logger.info(f"Writing artefacts to {report_dir}")
            for artefact in self.report_artefacts.keys():
                artefact_path = self.report_artefacts[artefact]["path"]
                artefact = self.report_artefacts[artefact]["artefact"]
                artefact_io = io.BytesIO(artefact)

                path = Path(report_dir, artefact_path)
                path.parent.mkdir(parents=True, exist_ok=True)

                with open(path, "wb") as f:
                    f.write(artefact_io.getbuffer())

        logger.info("Done!")

    def save(self, path: str):
        """
        Saves the pickled linker object to a file.

        Arguments:
            path: a valid pathlike string ending with ".pickle"

        Raises:
            ValueError: if path doesn't end ".pickle"
        """

        to_write = Path(path)

        if to_write.suffix != ".pickle":
            raise ValueError("Path must end '.pickle'")

        with open(to_write, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """
        Loads a pickled linker object from a file.

        Arguments:
            path: a valid pathlike string ending with ".pickle"

        Raises:
            ValueError: if path doesn't end ".pickle"

        Returns:
            An instance of Linker or its subclasses
        """

        to_read = Path(path)

        if to_read.suffix != ".pickle":
            raise ValueError("Path must end '.pickle'")

        with open(to_read, "rb") as f:
            return pickle.load(f)
