"""Functions abstracting the interaction with the server API."""

from matchbox.client._handler.admin import (
    auth_status,
    count_backend_items,
    delete_orphans,
    login,
)
from matchbox.client._handler.collections import (
    create_collection,
    create_resolution,
    create_run,
    delete_resolution,
    delete_run,
    get_collection,
    get_resolution,
    get_resolution_stage,
    get_results,
    get_run,
    list_collections,
    set_data,
    set_run_default,
    set_run_mutable,
    update_resolution,
)
from matchbox.client._handler.eval import (
    download_eval_data,
    sample_for_eval,
    send_eval_judgement,
)
from matchbox.client._handler.main import healthcheck
from matchbox.client._handler.query import match, query

__all__ = [
    # admin
    "login",
    "auth_status",
    "count_backend_items",
    "delete_orphans",
    # main
    "healthcheck",
    # eval
    "sample_for_eval",
    "send_eval_judgement",
    "download_eval_data",
    # query
    "query",
    "match",
    # collections
    "list_collections",
    "get_collection",
    "create_collection",
    "get_run",
    "create_run",
    "delete_run",
    "set_run_default",
    "set_run_mutable",
    "create_resolution",
    "update_resolution",
    "get_resolution",
    "set_data",
    "get_resolution_stage",
    "get_results",
    "delete_resolution",
]
