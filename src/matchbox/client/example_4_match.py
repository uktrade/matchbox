"""Use a collection to map IDs."""

from sqlalchemy import create_engine

from matchbox import pipelines as mbp

engine = create_engine("postgresql:///")
postgres = mbp.RelationalDBLocation(name="data_workspace_postgres", client=engine)

collection = mbp.Collection("default_companies")

# We can translate a Companies House number to a Data Hub ID
collection.map_key(
    from_source="companies_house",
    to_source="data_hub",
    key="8534735",
)
