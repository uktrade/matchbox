from src.locations import DATA_SUBDIR, PROJECT_DIR, DATA_HOME

import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text as sql_text
from sqlalchemy.exc import MultipleResultsFound
import glob
from pathlib import Path
import requests
import duckdb

import os
from contextlib import closing

DEFAULT_BATCH_SIZE = 10000
DEFAULT_DF_FORMAT = "fea"  # can be switched to csv

HTTPFS_PATH = Path(PROJECT_DIR) / "scratch" / "httpfs.duckdb_extension"
DEFAULT_DUCKDB_PATH = Path(DATA_HOME) / "company_matching.duckdb"

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

sql_engine = sqlalchemy.create_engine("postgresql://")


def query(sql, params=None, **kwargs):
    """
    Read full results set from Data Workspace based on arbitrary query

    Parameters:
        sql: a valid Postgres-SQL query
        params: a dictionary of parameters to format the SQL string
            See https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql

    Returns:
        df: pandas dataframe read from Data Workspace
    """

    with sql_engine.connect() as connection:
        return pd.read_sql(sql_text(sql), connection, params=params, **kwargs)


def query_iter(sql, params=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Read lazily (in chunks) from Data Workspace based on arbitrary query.
    Usage:
    ```
    sql_query = ...
    with query_iter(sql_query) as qiter:
        for chunk in qiter:
            print(chunk)
    ```

    Parameters:
        sql: a valid Postgres-SQL query
        params: a dictionary of parameters to format the SQL string
            See https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql
        batch_size: the number of rows to be read at a time

    Returns:
        a context manager yielding results in batch
    """

    def query_iter_logic():
        # The server-side cursor allows to read individual chunks through a generator
        # (see SQLAlchemy documentation)
        with sql_engine.connect().execution_options(
            stream_results=True
        ) as server_side_cursor:
            yield from pd.read_sql(
                sql_text(sql), server_side_cursor, params=params, chunksize=batch_size
            )

    # This returns a context manager that closes the connection at the end of the block
    return closing(query_iter_logic())


def query_nonreturn(sql, params=None):
    """
    Execute an arbitrary query and don't return the result

    Parameters:
        sql: a valid Postgres-SQL query
        params: a dictionary of parameters to format the SQL string
            See https://docs.sqlalchemy.org/en/14/core/tutorial.html#using-textual-sql

    """
    with sql_engine.begin() as connection:
        connection.execute(sql_text(sql), params)


def dataset(dataset_name):
    """
    Read dataset from Data Workspace

    Parameters:
        dataset_name: specified in the format 'schema.table'

    Returns:
        df: pandas dataframe read from Data Workspace
    """
    sql = f"SELECT * FROM {dataset_name};"
    return query(sql)


def dataset_iter(dataset_name, batch_size=DEFAULT_BATCH_SIZE):
    """
    Read dataset lazily (in chunks) from Data Workspace.
    Usage:
    ```
    dataset_name = ...
    with dataset_iter(dataset_name) as diter:
        for chunk in diter:
            print(chunk)
    ```

    Parameters:
        dataset_name: specified in the format 'schema.table'
        batch_size: the number of rows to be read at a time

    Returns:
        a context manager yielding results in batch
    """
    sql = f"SELECT * FROM {dataset_name};"
    return query_iter(sql, batch_size=batch_size)


def data_workspace_write(schema, table, df, if_exists="fail", index=False, **kwargs):
    """
    Persist dataset as table in Data Workspace.

    Parameters:
        schema: (str) the name of the schema where 'table' will be held
        table: (str) the name of the table to use
        df: (Pandas DataFrame) the data to be stored in the table
        if_exists: (str) argument that will be passed to the underlying
            Pandas function. See Pandas docs for DataFrame.to_sql
        index: (bool) whether to write the index of the dataframe
        **kwargs: all other key-value arguments will be passed to the
            pandas.to_sql function

    NOTE: this function will fail if the DataFrame has individual cells
    that contain complex objects (e.g. numpy arrays). In this case, it may
    be possible to convert the cells to another format (e.g. a list-of-lists,
    or a JSON object)
    """
    with sql_engine.connect() as connection:
        df.to_sql(
            table,
            con=connection,
            schema=schema,
            index=index,
            if_exists=if_exists,
            **kwargs,
        )


def _get_df_path(data_subdir, ds_name, extension=DEFAULT_DF_FORMAT):
    if data_subdir not in DATA_SUBDIR:
        raise ValueError("The data location specified is invalid")

    return os.path.join(DATA_SUBDIR[data_subdir], ds_name + "." + extension)


def persist_df(df, data_subdir, ds_name, extension=DEFAULT_DF_FORMAT):
    """
    Store a dataframe in one of the data folders

    Raises:
        ValueError: when an unsupported extension is specified

    Parameters:
        df: pandas dataframe to be persisted
        data_subdir: subfolder within /data to use, e.g. "raw"
        ds_name: extension-less name of the file to be stored
        extension: Whether the file to be written is a feather or csv file
    """

    file_path = _get_df_path(data_subdir, ds_name, extension)
    if extension == "fea":
        df.to_feather(file_path)
    elif extension == "csv":
        df.to_csv(file_path, index=False)
    elif extension == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError("The format specified is not supported")


def load_df(data_subdir, ds_name, extension=DEFAULT_DF_FORMAT, **kwargs):
    """
    Reads a dataframe from one of the data folders

    Parameters:
        data_subdir: subfolder within /data to use, e.g. "raw"
        ds_name: extension-less name of the file to be read
        extension: Whether the file to be read is a feather or csv file
        **kwargs: keyword args passed to the underlying pandas function

    Raises:
        ValueError: when an unsupported extension is specified

    Returns:
        pandas dataframe read from disk
    """

    file_path = _get_df_path(data_subdir, ds_name, extension)

    if extension == "fea":
        return pd.read_feather(file_path, **kwargs)
    if extension == "csv":
        return pd.read_csv(file_path, low_memory=False, **kwargs)
    if extension == "parquet":
        return pd.read_parquet(file_path, **kwargs)

    raise ValueError("The format specified is not supported")


def get_schema_table_names(full_name: str, validate: bool = False) -> (str, str):
    """
    Takes a string table name and returns the unquoted schema and
    table as a tuple. If you insert these into a query, you need to
    add double quotes in from statements, or single quotes in where.

    Parameters:
        full_name: A string indicating a Postgres table
        validate: Whether to error if both schema and table aren't
        detected

    Raises:
        ValueError: When the function can't detect either a
        schema.table or table format in the input
        ValidationError: If both schema and table can't be detected
        when the validate argument is True

    Returns:
        (schema, table): A tuple of schema and table name. If schema
        cannot be inferred, returns None.
    """

    schema_name_list = full_name.replace('"', "").split(".")

    if len(schema_name_list) == 1:
        schema = None
        table = schema_name_list[0]
    elif len(schema_name_list) == 2:
        schema = schema_name_list[0]
        table = schema_name_list[1]
    else:
        raise ValueError(
            f"""
            Could not identify schema and table in {full_name}.
        """
        )

    if validate and schema is None:
        raise ("Schema could not be detected and validation required.")

    return (schema, table)


def check_table_exists(table: str) -> bool:
    """
    Returns true if a table exists

    Parameters:
        table: any valid query string for Postgres. Would prefer a
        schema to be included, but will attempt to check without

    Raises:
        ValueError: When a single answer can't be determined

    Returns:
        bool: whether or not the table exists
    """

    schema, tablename = get_schema_table_names(table)

    if schema is not None:
        schema_clause = f"and table_schema = '{schema}'"
    else:
        schema_clause = ""

    sql = f"""
        select exists (
            select from information_schema.tables
            where table_name = '{tablename}'
            {schema_clause}
        );
    """

    with sql_engine.connect() as connection:
        res = connection.execute(sql_text(sql))

        try:
            exists = res.scalar()
        except MultipleResultsFound:
            if "." not in table:
                schema_error = "Could not derive schema."
            else:
                schema_error = ""

            raise ValueError(
                f"Multiple results found. Table name unclear. {schema_error}"
            )

        return exists


def get_company_data(
    cols: str, dataset: str, where: str = "", sample: int = None, **kwargs
):
    """
    Generic function for getting company data from a variety of tables
    Args:
        cols [str]: A string to be concatenated into a SQL SELECT statement
        dataset [str]: A Data Workspace dataset in "schema.table" form
        where [str]: An optional condition for filtering the table
        sample [int, default: None]: A size of random sample to draw
        **kwargs: Keyword arguments passed to pandas.get_sql()
    Returns: the requested dataset
    """

    limit = ""

    if sample is not None:
        limit = f"order by random() limit {sample}"

    if where != "":
        where = f"where {where}"

    raw_query = f"""
        select {cols}
        from {dataset}
        {where}
        {limit}
    """

    df_raw = query(sql=raw_query, **kwargs)

    return df_raw


def clean_table_name(name):
    return name.replace('"', "").replace(".", "_")


def collapse_multiline_string(string: str):
    return "".join(line.strip() for line in string.splitlines())


def build_alias_path_dict(input_dir: str = None, match_pattern: str = "*"):
    """
    Takes a directory of processed data and returns the alias: path dict
    Args:
        input_dir [str]: the subdirectory of data/processed
    Returns: A dictionary of alias: path values
    """
    filepaths = glob.glob(
        os.path.join(DATA_SUBDIR["processed"], input_dir, match_pattern)
    )
    data = {}

    for df_path in filepaths:
        alias = Path(df_path).stem
        data[alias] = df_path

    return data


def get_aws_creds():
    r = requests.get(
        "http://169.254.170.2" + os.environ["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"]
    )

    return r.json()


def get_duckdb_s3_config_string():
    aws_creds = get_aws_creds()

    return f"""
        install '{HTTPFS_PATH.resolve()}';
        load '{HTTPFS_PATH.resolve()}';
        set s3_region='{os.environ['S3_REGION']}';
        set s3_access_key_id='{aws_creds['AccessKeyId']}';
        set s3_secret_access_key='{aws_creds['SecretAccessKey']}';
        set s3_session_token='{aws_creds['Token']}';
    """


def generate_dummy_df():
    """
    Returns a 2*2 pandas dataframe.
    """
    return pd.DataFrame(
        {"irrational": ["pi", "e", "phi"], "rounded": [3.14, 2.72, 1.62]}
    )


def get_duckdb_connection(path=DEFAULT_DUCKDB_PATH.as_posix()):
    return duckdb.connect(database=path, read_only=False)
