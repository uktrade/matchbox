from io import BytesIO

from pyarrow import Table
from pyarrow.parquet import read_table

from matchbox.client._handler.main import CLIENT, http_retry, url_params
from matchbox.common.arrow import (
    SCHEMA_QUERY,
    SCHEMA_QUERY_WITH_LEAVES,
    check_schema,
)
from matchbox.common.dtos import (
    Match,
    ResolverResolutionPath,
    SourceResolutionPath,
)
from matchbox.common.exceptions import (
    MatchboxEmptyServerResponse,
)
from matchbox.common.logging import logger
from matchbox.common.resolvers import ResolverSettings


@http_retry
def query(
    source: SourceResolutionPath,
    return_leaf_id: bool,
    resolution: ResolverResolutionPath | None = None,
    resolver_overrides: ResolverSettings | None = None,
    limit: int | None = None,
) -> Table:
    """Query a source in Matchbox."""
    log_prefix = f"Query {source}"
    logger.debug(f"Using {resolution}", prefix=log_prefix)

    params = url_params(
        {
            "collection": source.collection,
            "run_id": source.run,
            "source": source.name,
            "resolution": resolution.name if resolution else None,
            "return_leaf_id": return_leaf_id,
            "limit": limit,
        }
    )
    if resolver_overrides is None:
        res = CLIENT.get("/query", params=params)
    else:
        res = CLIENT.post(
            "/query",
            params=params,
            json={"resolver_overrides": resolver_overrides.model_dump(mode="json")},
        )

    buffer = BytesIO(res.content)
    table = read_table(buffer)

    logger.debug("Finished", prefix=log_prefix)

    expected_schema = SCHEMA_QUERY
    if return_leaf_id:
        expected_schema = SCHEMA_QUERY_WITH_LEAVES

    check_schema(expected_schema, table.schema)

    if table.num_rows == 0:
        raise MatchboxEmptyServerResponse(operation="query")

    return table


@http_retry
def match(
    targets: list[SourceResolutionPath],
    source: SourceResolutionPath,
    key: str,
    resolution: ResolverResolutionPath,
    resolver_overrides: ResolverSettings | None = None,
) -> list[Match]:
    """Match a source against a list of targets."""
    log_prefix = f"Query {source}"
    logger.debug(
        f"{key} to {', '.join(str(targets))} using {resolution}",
        prefix=log_prefix,
    )

    params = url_params(
        {
            "collection": resolution.collection,
            "run_id": resolution.run,
            "targets": [t.name for t in targets],
            "source": source.name,
            "key": key,
            "resolution": resolution.name,
        }
    )
    if resolver_overrides is None:
        res = CLIENT.get("/match", params=params)
    else:
        res = CLIENT.post(
            "/match",
            params=params,
            json={"resolver_overrides": resolver_overrides.model_dump(mode="json")},
        )

    logger.debug("Finished", prefix=log_prefix)

    matches = [Match.model_validate(m) for m in res.json()]

    if not matches:
        raise MatchboxEmptyServerResponse(operation="match")

    return matches
