"""Create and run pipeline step by step."""

from sqlalchemy import create_engine

# this import contains all the things you need to buld matchbox pipelines
from matchbox import pipelines as mbp

# Define your database and your credentials
engine = create_engine("postgresql:///")
postgres = mbp.RelationalDBLocation(name="data_workspace_postgres", client=engine)

# Name the collection of entities that your pipeline will capture
# The collection object is your interface to the server
collection = mbp.Collection("default_companies")

# Define a data source to index
ch = mbp.Source(
    location=postgres,
    name="companies_house",
    extract_transform="""
        select
            id::text as id,
            company_name,
            company_number::text as company_number,
            postcode,
            to_date(incorporation_date, 'DD/MM/YYYY') as incorporation_date
        from
            companieshouse.companies;
    """,
    key_field="id",
    index_fields=["company_name", "company_number", "postcode", "incorporation_date"],
)

# Can inspect the shape of the data - nothing has been sent to Matchbox server yet
ch_df = ch.fetch(batch_size=10_000)

# We're happy with it - we can now index it in the Matchbox server
collection.add_source(ch)

# Before we deduplicate, we want to define data cleaners
clean_company_name = ...  # your own function that produces SQL
clean_postcode = ...  # your own function that produces SQL
ch_clean = {
    "company_name": clean_company_name(ch.f("company_name")),
    "postcode": clean_postcode(ch.f("postcode")),
}


# Now we can define the model to deduplicate
dedupe_ch = mbp.Model(
    name="dedupe_companies_house",
    description="Basic deduplication on identical name and postcode.",
    model_class=mbp.DeduperType.DETERMINISTIC,
    settings={
        "unique_fields": ["company_name", "postcode"],
    },
    # We only define parameters for the "left" side - only linkers have a "right" side
    # This query implicitly connects the model to our previously indexed source
    left_query=mbp.Query({ch: ["company_name", "postcode"]}, batch_size=10_000),
    left_cleaning=ch_clean,
)

# Before we run the model, we can check how the data is cleaned and iterate
df_for_dedupe = dedupe_ch.fetch_left()
dedupe_ch.clean_left(df_for_dedupe)

# Running the model produces results, which we can inspect
dedupe_ch.run()
print(dedupe_ch.results)

# If we're happy with them, we can record results to the Matchbox server
collection.add_model(dedupe_ch)

# If all we want in our collection is a deduplicated version
# of Companies House, we're done. We can release the new version
# which will include all our pending changes

collection.new_version()
