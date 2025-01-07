from matchbox.client.helpers.selector import query, selector, match


def generate_query(backend, crn, fields, resolution, return_type):
    select_crn = selector(
        table=str(crn),
        fields=fields,
        engine=crn.database.engine,
    )

    df_crn, query_string = query(
        selector=select_crn,
        backend=backend,
        resolution=resolution,
        return_type=return_type,
    )
    return query_string


def generate_match(backend,  source, target, source_pk, resolution):
    res, query_string = match(
        backend= backend,
        source_pk= source_pk,
        source=source,
        target=target,
        resolution=resolution
    )
    return query_string

if __name__ == "__main__":
    # Generate variables to pass to the query function
    generate_query()
