"""Illustrate new interface."""

from matchbox.client.collections import Collection
from matchbox.client.dags import run_dag
from matchbox.common.dags import DAG, DeduperClass
from matchbox.common.dtos import ModelConfig
from matchbox.common.sources import SourceConfig

hash_source = ...
run_query = ...
run_model = ...
QueryConfig = ...

# Imperative DAG building
engine = ...
version = Collection("default_companies").create().new_version()


source = SourceConfig(..., batch_size=10_000)
hashes = hash_source(source, client=...)
version.add_source(source, hashes)

left_query = QueryConfig(
    resolution_name=source.name,
    # All these attributes are optional
    select={source.name: source.f("field")},
    cleaning_dict={source.name: f"select * from {source.name}"},
    threshold=99,
    combine_type="explode",
)

model = ModelConfig(
    name="model_name",
    description="description",
    model_class=DeduperClass.NAIVE,
    model_settings={
        "id": "id",
        "unique_fields": ["field1", "field2"],
    },
    left_query=left_query,
)

results = run_model(model, left_df=run_query(left_query))

version.add_model(model, results)
version.set_model_threshold(99)

version.set_current()

# Declarative DAG building: easy to move to DAG when ready
engine = ...
version = Collection("default_companies").create().new_version()
dag = DAG()
source_a = SourceConfig(...)
dedupe_a = ModelConfig(...)
dag.add_steps(source_a, dedupe_a)
run_dag(dag, version=version, default_client=engine)  # can set clients per source

version.set_current()


# DAG re-running
# works regardless of whether the DAG was created imperatively or declaratively
engine = ...
version = Collection("default_companies").get().current_version()
dag = version.get_dag()
run_dag(dag, version=version, default_client=engine)
version.set_current()

# Re-running manually and imperatively
default_companies = Collection("default_companies").get()

current_version = default_companies.current_version()
source_config = current_version.get_source("companies_house")
model_config = current_version.get_model("dedupe_companies_house")

new_version = default_companies.new_version()

new_version.add_source(
    # Batch size is set in the source_config, but can be overwritten
    source_config,
    hash_source(source_config, client=engine, batch_size=10_000),
)

df = run_query(model_config.left_query)
new_version.add_model(model, run_model(model, left_df=df))

new_version.set_current()


# TODO:
# collection and version ORM
# _handler.get_collection; and API; and adapter
# _handler.get_current_version; and API; and adapter
# _handler.create_version; and API; and adapter
# _handler.delete_collection; and API; and adapter
# _handler.delete_version; and API; and adapter
# align arguments of _handler.delete_resolution; and API; and adapter
# _handler.set_current_verion; and API; and adapter
# align arguments of _handler.index and rename to _handler.creatr_source; an API;
#   and adapter
# align arguments of _handler.insert_model and rename to _handler.create_model; an API;
#   and adapter
# _handler.get_dag() and all connected things
