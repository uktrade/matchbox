"""Fetch existing DAGs, sources and models."""

from sqlalchemy import create_engine

from matchbox import pipelines as mbp

engine = create_engine("postgresql:///")
postgres = mbp.RelationalDBLocation(name="data_workspace_postgres", client=engine)

collection = mbp.Collection("default_companies")

# Fetch the DAG of the most recent active (not pending) version
dag = collection.get_dag()
# We need to pass a location - for our simple use case, it's
# the same for all data sources
# This will re-run the previously-run DAG, and create a new pending version
collection.add_dag(dag, default_location=postgres)

# And this will move our pending version to be the active one
collection.new_version()


# We can inspect the configurations for the active DAG
source_config = collection.get_source("companies_house")
model_config = collection.get_model("dedupe_companies_house")

print(source_config)
print(model_config)
