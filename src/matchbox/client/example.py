"""Illustrate new interface."""

from matchbox.client.collections import Collection
from matchbox.client.dags import DAG
from matchbox.client.models.models import Model
from matchbox.client.queries import query, select
from matchbox.common.sources import SourceConfig

# Imperative DAG building
version = Collection("default_companies").create().new_version()
source = SourceConfig(...)

version.add_source(source)

model = Model(...)
results = model.run()

version.add_model(model)

version.set_current()

# Declarative DAG building
version = Collection("default_companies").create().new_version()
dag = DAG(version)
dag.add_steps(..., ..., ...)
dag.run()

version.set_current()


# DAG re-running
# works regardless of whether the DAG was created imperatively or declaratively
version = Collection("default_companies").get().current_version()
dag = version.get_dag()
dag.run()
version.set_current()

# Re-running manually and imperatively
default_companies = Collection("default_companies").get()

current_version = default_companies.current_version()
source = current_version.get_source("companies_house")
model = current_version.get_model("dedupe_companies_house")

df = query(select("companies_house"))
model.run(left_data=df)

new_version = default_companies.new_version()
new_version.add_source(source)
new_version.add_model(model)

new_version.set_current()


# TODO:
# collection ORM
# version ORM
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
