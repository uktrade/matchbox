# Data retrieval

## Match

Given a key and a source, retrieves all keys that share its cluster in both the source and target. Useful for making ad-hoc queries about specific items of data.

=== "Example"
    ```python
    import matchbox as mb
    from matchbox import select
    import sqlalchemy

    mb.match(
        "datahub_companies",
        source="companies_house",
        key="8534735",
        resolution="last_linker",
    )
    ```

=== "Output"
    ```json
    [
        {
            "cluster": 2354,
            "source": "dbt.companieshouse",
            "source_id": ["8534735", "8534736"],
            "target": "hmrc.exporters",
            "target_id": ["EXP123", "EXP124"]
        }
    ]
    ```

## Query

Retrieves entire data sources along with a unique entity identifier according to a point of resolution.

!!! tip "Use Cases"
    * Large-scale statistical analysis
    * Building linking or deduplication pipelines

=== "Example"
    ```python
    import matchbox as mb
    from matchbox import select
    import sqlalchemy

    engine = sqlalchemy.create_engine('postgresql://')

    mb.query(
        select(
            {
                "dbt.companieshouse": ["company_name"],
                "hmrc.exporters": ["year", "commodity_codes"],
            },
            credentials=engine,
        )
        combine_type="set_agg",
        resolution="companies",
    )
    ```

=== "Output"
    ```text
    id      dbt_companieshouse_company_name         hmrc_exporters_year     hmrc_exporters_commodity_codes
    122     Acme Ltd.                               2023                    ['85034', '85035']
    122     Acme Ltd.                               2024                    ['72142', '72143']
    5       Gamma Exports                           2023                    ['90328', '90329']
    ...
    ```

For more information on how to use the functions on this page, please check out the relevant examples in the [client API docs](../api/client/index.md).