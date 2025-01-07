from matchbox.client.helpers.selector import match, query, selector
from test.fixtures.data import crn_companies


# Generates a query that can be used for benchmarking purposes
def generate_query(crn, fields, backend, resolution, return_type):
    select_crn = selector(
        table=str(crn),
        fields=fields,
        engine=crn.database.engine,
    )

    query_string = query(
        selector=select_crn,
        backend=backend,
        resolution=resolution,
        return_type=return_type,
    )

    return query_string


if __name__ == "__main__":
    query=''
    # vars: crn, fields, backend, resolution, return_type
    # query = generate_query(crn, fields, backend, resolution, return_type)
    print(query)