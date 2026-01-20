"""Functions abstracting the interaction with the server API."""

from matchbox.client._handler.admin import (
    add_user_to_group,
    count_backend_items,
    create_group,
    delete_group,
    delete_orphans,
    get_group,
    get_system_permissions,
    list_groups,
    remove_user_from_group,
)
from matchbox.client._handler.auth import auth_status, login
from matchbox.client._handler.collections import (
    create_collection,
    create_resolution,
    create_run,
    delete_resolution,
    delete_run,
    get_collection,
    get_collection_permissions,
    get_resolution,
    get_resolution_stage,
    get_results,
    get_run,
    grant_collection_permission,
    list_collections,
    revoke_collection_permission,
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
    # auth
    "auth_status",
    "login",
    # admin
    "add_user_to_group",
    "count_backend_items",
    "create_group",
    "delete_group",
    "delete_orphans",
    "get_group",
    "get_system_permissions",
    "list_groups",
    "remove_user_from_group",
    # main
    "healthcheck",
    # eval
    "download_eval_data",
    "sample_for_eval",
    "send_eval_judgement",
    # query
    "query",
    "match",
    # collections
    "create_collection",
    "create_resolution",
    "create_run",
    "delete_resolution",
    "delete_run",
    "get_collection",
    "get_collection_permissions",
    "get_resolution",
    "get_resolution_stage",
    "get_results",
    "get_run",
    "grant_collection_permission",
    "list_collections",
    "revoke_collection_permission",
    "set_data",
    "set_run_default",
    "set_run_mutable",
    "update_resolution",
]
