"""Illustrate new interface."""

from matchbox.client.collections import Collection
from matchbox.client.dags import run_dag
from matchbox.client.queries import query, select
from matchbox.common.dags import DAG
from matchbox.common.dtos import ModelConfig
from matchbox.common.sources import SourceConfig

hash_source = ...
run_model = ...

# Imperative DAG building
version = Collection("default_companies").create().new_version()

source = SourceConfig(...)
hashes = hash_source(source, client=..., batch_size=...)

version.add_source(source, hashes)

model = ModelConfig(...)
results = run_model(model, left_df=query(select("source")))

version.add_model(model, results)
version.set_model_threshold(99)

version.set_current()

# Declarative DAG building
engine = ...
version = Collection("default_companies").create().new_version()
dag = DAG()
dag.add_steps(..., ..., ...)
run_dag(dag, version=version, default_client=engine)  # can set clients per source

version.set_current()


# DAG re-running
# works regardless of whether the DAG was created imperatively or declaratively
engine = ...
version = Collection("default_companies").get().current_version()
run_dag(dag, version=version, default_client=engine)
version.set_current()

# Re-running manually and imperatively
default_companies = Collection("default_companies").get()

current_version = default_companies.current_version()
source = current_version.get_source("companies_house")
model_config = current_version.get_model("dedupe_companies_house")

new_version = default_companies.new_version()

new_version.add_source(source, hash_source(source, client=..., batch_size=...))

df = query(select("companies_house"))
new_version.add_model(model_config, run_model(model_config, left_df=df))

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
