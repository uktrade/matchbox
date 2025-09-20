# Look up matches

## Single keys

Given a key and a source, you can retrieve all keys resolving to the same entity in other sources. Useful for making ad-hoc queries for single data items.

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    matches = DAG("companies").lookup_key(
        from_source="datahub_companies",
        to_sources=["companies_house"],
        key="8534735",
    )

    print(matches["datahub_companies"])
    print(matches["companies_house"])
    ```

=== "Output"
    ```
    ["8534735", "8534736"]
    ["EXP123", "EXP124"]
    ```

## Extract whole lookup

You can download an entire lookup as a PyArrow table.

=== "Example"
    ```python
    from matchbox.client.dags import DAG

    lookup = DAG("companies").extract_lookup()
    ```