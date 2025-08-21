"""Create and run pipeline as a DAG."""

from sqlalchemy import create_engine

from matchbox import pipelines as mbp

engine = create_engine("postgresql:///")
postgres = mbp.RelationalDBLocation(name="data_workspace_postgres", client=engine)

collection = mbp.Collection("default_companies")

# We've iterated a bunch on the various parts of a pipeline.
# We're now ready to construct a more complex pipeline as a DAG (Directed Acyclic Graph)
# It will look like this:
# companies house                               data hub
#         │                                          │
#         │                                          │
#         ▼                                          ▼
# deduplicated companies house             deduplicated data hub
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         │                                          │
#         └──────► companies house and data hub  ◄───┘

# Source for Companies House is the same as before
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

# Now we define a source for Data Hub
dh = mbp.Source(
    location=postgres,
    name="data_hub",
    extract_transform="""
        select
            companies.id::text as id,
            companies.name as company_name,-=-
            companies.company_number,
            companies.duns_number,
            companies.address_postcode as postcode
        from
            dbt.data_hub__companies companies;
    """,
    key_field="id",
    index_fields=["company_name", "company_number", "duns_number", "postcode"],
)


# For this simple DAG, we'll use the same cleaner for all data sources
clean_company_name = ...  # your own function that produces SQL
clean_postcode = ...  # your own function that produces SQL
cleaner = mbp.Cleaner(
    {
        "company_name": clean_company_name(ch.f("company_name")),
        "postcode": clean_postcode(ch.f("postcode")),
    }
)

# Same deduper for Companies House as before
dedupe_ch = mbp.Model(
    name="dedupe_companies_house",
    description="Basic deduplication on identical name and postcode.",
    model_class=mbp.DeduperType.DETERMINISTIC,
    settings={
        "unique_fields": ["company_name", "postcode"],
    },
    left_query=mbp.Query({ch: ["company_name", "postcode"]}, batch_size=10_000),
    left_cleaning=cleaner,
)

# Now we have a deduper for Data Hub
dedupe_dh = mbp.Model(
    name="dedupe_data_hub",
    description="Basic deduplication on identical name and postcode.",
    model_class=mbp.DeduperType.DETERMINISTIC,
    settings={
        "unique_fields": ["company_name", "postcode"],
    },
    left_query=mbp.Query({dh: ["company_name", "postcode"]}, batch_size=10_000),
    left_cleaning=cleaner,
)

# Ready for our first linker
link_ch_dh = mbp.Model(
    name="link_companies_house_data_hub",
    description="Basic linking on identical name and postcode.",
    model_class=mbp.LinkerType.DETERMINISTIC,
    settings={
        "unique_fields": ["company_name", "postcode"],
    },
    left_query=mbp.Query(
        {ch: ["company_name", "postcode"]},
        # We must now specify a resolution! We're not querying from the original source
        resolution="dedupe_companies_house",
        batch_size=10_000,
    ),
    left_cleaning=cleaner,
    # We now have a right query and right cleaner
    right_query=mbp.Query(
        {dh: ["company_name", "postcode"]},
        resolution_name="dedupe_data_hub",
        batch_size=10_000,
    ),
    right_cleaning=cleaner,
)

# Unlike first example, we haven't run anything yet, just defined DAG components

# Will validate our DAG makes sense (this is safer than running things one by one)
# It will also ensure things are run in the most efficient order
dag = mbp.DAG(ch, dh, dedupe_ch, dedupe_dh, link_ch_dh)

# This will run the whole DAG and send the results to Matchbox one at a time
collection.add_dag(dag)

# We can now commit the results of our DAG and allow users to query them!
collection.new_version()
