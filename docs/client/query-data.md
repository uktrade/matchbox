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
            "source": "companieshouse",
            "source_id": ["8534735", "8534736"],
            "target": "datahub_companies",
            "target_id": ["EXP123", "EXP124"]
        }
    ]
    ```


For more information on how to use the functions on this page, please check out the relevant examples in the [client API docs](../api/client/index.md).