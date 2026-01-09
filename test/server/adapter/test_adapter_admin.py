"""Test the backend adapter's admin functions."""

from functools import partial
from typing import Literal

import pytest
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.dtos import (
    BackendResourceType,
    CollectionName,
    Group,
    GroupName,
    PermissionType,
    User,
)
from matchbox.common.exceptions import (
    MatchboxCollectionNotFoundError,
    MatchboxDataNotFound,
    MatchboxDeletionNotConfirmed,
    MatchboxGroupAlreadyExistsError,
    MatchboxGroupNotFoundError,
    MatchboxSystemGroupError,
    MatchboxUserNotFoundError,
)
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.docker
class TestMatchboxAdminBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

    # User management

    def test_login_creates_new_user(self) -> None:
        """Login creates a new user if they don't exist."""
        with self.scenario(self.backend, "bare") as _:
            user = User(user_name="alice", email="alice@example.com")
            result = self.backend.login(user)

            assert result.user_name == "alice"
            assert result.email == "alice@example.com"

    def test_login_returns_existing_user(self) -> None:
        """Login returns existing user with same ID."""
        with self.scenario(self.backend, "bare") as _:
            user1 = User(user_name="alice", email="alice@example.com")
            result1 = self.backend.login(user1)

            user2 = User(user_name="alice", email="alice@example.com")
            result2 = self.backend.login(user2)

            assert result1.user_name == result2.user_name

    def test_login_updates_email(self) -> None:
        """Login updates email if it changes."""
        with self.scenario(self.backend, "bare") as _:
            user1 = User(user_name="alice", email="alice@example.com")
            _ = self.backend.login(user1)

            user2 = User(user_name="alice", email="alice@newdomain.com")
            result2 = self.backend.login(user2)

            assert result2.email == "alice@newdomain.com"

    def test_login_different_users(self) -> None:
        """Login creates different IDs for different users."""
        with self.scenario(self.backend, "bare") as _:
            alice = User(user_name="alice")
            bob = User(user_name="bob")

            alice_result = self.backend.login(alice)
            bob_result = self.backend.login(bob)

            assert alice_result.user_name != bob_result.user_name

    # Group management

    def test_create_group(self) -> None:
        """Can create a new group."""
        with self.scenario(self.backend, "bare") as _:
            group = Group(name=GroupName("admins"), description="Administrator group")
            self.backend.create_group(group)

            # Verify it was created
            retrieved = self.backend.get_group(GroupName("admins"))
            assert retrieved.name == "admins"
            assert retrieved.description == "Administrator group"
            assert retrieved.is_system is False
            assert len(retrieved.members) == 0

    def test_create_group_duplicate_fails(self) -> None:
        """Cannot create a group with duplicate name."""
        with self.scenario(self.backend, "bare") as _:
            group1 = Group(name=GroupName("admins"))
            self.backend.create_group(group1)

            group2 = Group(name=GroupName("admins"))
            with pytest.raises(MatchboxGroupAlreadyExistsError):
                self.backend.create_group(group2)

    def test_list_groups_empty(self) -> None:
        """List groups returns empty list when no groups exist."""
        with self.scenario(self.backend, "bare") as _:
            groups = self.backend.list_groups()
            assert groups == []

    def test_list_groups(self) -> None:
        """List groups returns all groups."""
        with self.scenario(self.backend, "bare") as _:
            group1 = Group(name=GroupName("admins"), description="Admins")
            group2 = Group(name=GroupName("users"), description="Users")

            self.backend.create_group(group1)
            self.backend.create_group(group2)

            groups = self.backend.list_groups()
            assert len(groups) == 2
            group_names = {g.name for g in groups}
            assert group_names == {"admins", "users"}

    def test_get_group_not_found(self) -> None:
        """Get group raises error if group doesn't exist."""
        with self.scenario(self.backend, "bare") as _:  # noqa: SIM117
            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.get_group(GroupName("nonexistent"))

    def test_delete_group(self) -> None:
        """Can delete a group."""
        with self.scenario(self.backend, "bare") as _:
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            # Verify it exists
            retrieved = self.backend.get_group(GroupName("admins"))
            assert retrieved.name == "admins"

            # Delete it
            self.backend.delete_group(GroupName("admins"), certain=True)

            # Verify it's gone
            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.get_group(GroupName("admins"))

    def test_delete_group_requires_confirmation(self) -> None:
        """Delete group requires certain=True."""
        with self.scenario(self.backend, "bare") as _:
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            with pytest.raises(MatchboxDeletionNotConfirmed):
                self.backend.delete_group(GroupName("admins"), certain=False)

    def test_delete_group_not_found(self) -> None:
        """Delete group raises error if group doesn't exist."""
        with self.scenario(self.backend, "bare") as _:  # noqa: SIM117
            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.delete_group(GroupName("nonexistent"), certain=True)

    def test_delete_system_group_fails(self) -> None:
        """Cannot delete a system group."""
        with self.scenario(self.backend, "bare") as _:
            system_group = Group(name=GroupName("system_admins"), is_system=True)
            self.backend.create_group(system_group)

            with pytest.raises(MatchboxSystemGroupError):
                self.backend.delete_group(GroupName("system_admins"), certain=True)

    # User-group membership

    def test_add_user_to_group(self) -> None:
        """Can add a user to a group."""
        with self.scenario(self.backend, "bare") as _:
            # Create group
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            # Create user
            user = User(user_name="alice")
            self.backend.login(user)

            # Add user to group
            self.backend.add_user_to_group("alice", GroupName("admins"))

            # Verify membership
            groups = self.backend.get_user_groups("alice")
            assert GroupName("admins") in groups

    def test_add_user_to_nonexistent_user_fails(self) -> None:
        """Cannot add non-existent user to group."""
        with self.scenario(self.backend, "bare") as _:
            # Create group
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            # Try to add non-existent user to group
            with pytest.raises(MatchboxUserNotFoundError):
                self.backend.add_user_to_group("nonexistent", GroupName("admins"))

    def test_add_user_to_group_idempotent(self) -> None:
        """Adding user to group twice doesn't cause error."""
        with self.scenario(self.backend, "bare") as _:
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            self.backend.add_user_to_group("alice", GroupName("admins"))
            self.backend.add_user_to_group("alice", GroupName("admins"))

            groups = self.backend.get_user_groups("alice")
            assert GroupName("admins") in groups

    def test_add_user_to_nonexistent_group_fails(self) -> None:
        """Cannot add user to non-existent group."""
        with self.scenario(self.backend, "bare") as _:
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.add_user_to_group("alice", GroupName("nonexistent"))

    def test_remove_user_from_group(self) -> None:
        """Can remove a user from a group."""
        with self.scenario(self.backend, "bare") as _:
            # Setup
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)
            self.backend.add_user_to_group("alice", GroupName("admins"))

            # Verify user is in group
            groups = self.backend.get_user_groups("alice")
            assert GroupName("admins") in groups

            # Remove user from group
            self.backend.remove_user_from_group("alice", GroupName("admins"))

            # Verify user is no longer in group
            groups = self.backend.get_user_groups("alice")
            assert GroupName("admins") not in groups

    def test_remove_user_from_nonexistent_group_fails(self) -> None:
        """Cannot remove user from non-existent group."""
        with self.scenario(self.backend, "bare") as _:
            user = User(user_name="alice")
            self.backend.login(user)

            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.remove_user_from_group("alice", GroupName("nonexistent"))

    def test_remove_nonexistent_user_from_group_fails(self) -> None:
        """Cannot remove non-existent user from group."""
        with self.scenario(self.backend, "bare") as _:
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            with pytest.raises(MatchboxUserNotFoundError):
                self.backend.remove_user_from_group("nonexistent", GroupName("admins"))

    def test_get_user_groups_empty(self) -> None:
        """Get user groups returns empty list for user with no groups."""
        with self.scenario(self.backend, "bare") as _:
            user = User(user_name="alice")
            self.backend.login(user)

            groups = self.backend.get_user_groups("alice")
            assert groups == []

    def test_get_user_groups_nonexistent_user_fails(self) -> None:
        """Get user groups raises error for non-existent user."""
        with self.scenario(self.backend, "bare") as _:  # noqa: SIM117
            with pytest.raises(MatchboxUserNotFoundError):
                self.backend.get_user_groups("nonexistent")

    def test_get_user_groups_multiple(self) -> None:
        """Get user groups returns all groups for a user."""
        with self.scenario(self.backend, "bare") as _:
            # Create user
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            # Create groups
            group1 = Group(name=GroupName("admins"))
            group2 = Group(name=GroupName("users"))
            self.backend.create_group(group1)
            self.backend.create_group(group2)

            # Add user to both groups
            self.backend.add_user_to_group("alice", GroupName("admins"))
            self.backend.add_user_to_group("alice", GroupName("users"))

            # Verify membership
            groups = self.backend.get_user_groups("alice")
            assert len(groups) == 2
            assert set(groups) == {GroupName("admins"), GroupName("users")}

    def test_get_group_includes_members(self) -> None:
        """Get group returns members list."""
        with self.scenario(self.backend, "bare") as _:
            # Create group and users
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            alice = User(user_name="alice", email="alice@example.com")
            bob = User(user_name="bob", email="bob@example.com")
            self.backend.login(alice)
            self.backend.login(bob)

            # Add users to group
            self.backend.add_user_to_group("alice", GroupName("admins"))
            self.backend.add_user_to_group("bob", GroupName("admins"))

            # Get group and verify members
            retrieved = self.backend.get_group(GroupName("admins"))
            assert len(retrieved.members) == 2
            member_names = {m.user_name for m in retrieved.members}
            assert member_names == {"alice", "bob"}

    # Permissions

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    @pytest.mark.parametrize(
        ("granted_permission", "can_read", "can_write", "can_admin"),
        [
            (PermissionType.READ, True, False, False),
            (PermissionType.WRITE, True, True, False),
            (PermissionType.ADMIN, True, True, True),
        ],
        ids=["read-only", "write-implies-read", "admin-implies-all"],
    )
    def test_permission_hierarchy(
        self,
        resource: Literal[BackendResourceType.SYSTEM] | CollectionName,
        scenario: str,
        granted_permission: PermissionType,
        can_read: bool,
        can_write: bool,
        can_admin: bool,
    ) -> None:
        """Test permission hierarchy: ADMIN > WRITE > READ."""
        with self.scenario(self.backend, scenario) as _:
            group_name = f"{granted_permission}_group"
            user_name = f"user_{granted_permission}"

            # Setup: create group, user, and grant permission
            group = Group(name=GroupName(group_name))
            self.backend.create_group(group)

            user = User(user_name=user_name)
            self.backend.login(user)
            self.backend.add_user_to_group(user_name, GroupName(group_name))

            self.backend.grant_permission(
                GroupName(group_name),
                granted_permission,
                resource,
            )

            # Test: verify permission hierarchy
            assert (
                self.backend.check_permission(user_name, PermissionType.READ, resource)
                == can_read
            )
            assert (
                self.backend.check_permission(user_name, PermissionType.WRITE, resource)
                == can_write
            )
            assert (
                self.backend.check_permission(user_name, PermissionType.ADMIN, resource)
                == can_admin
            )

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    def test_grant_permission(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Can grant permission to a group."""
        with self.scenario(self.backend, scenario) as _:
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            self.backend.grant_permission(
                GroupName("admins"),
                PermissionType.ADMIN,
                resource,
            )

            # Verify permission was granted
            permissions = self.backend.get_permissions(resource)
            assert len(permissions) == 1
            assert permissions[0].group_name == "admins"
            assert permissions[0].permission == PermissionType.ADMIN

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    def test_grant_permission_idempotent(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Granting same permission twice doesn't cause error."""
        with self.scenario(self.backend, scenario) as _:
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            self.backend.grant_permission(
                GroupName("admins"),
                PermissionType.READ,
                resource,
            )
            self.backend.grant_permission(
                GroupName("admins"),
                PermissionType.READ,
                resource,
            )

            permissions = self.backend.get_permissions(resource)
            assert len(permissions) == 1

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    def test_revoke_permission(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Can revoke permission from a group."""
        with self.scenario(self.backend, scenario) as _:
            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)

            # Grant then revoke
            self.backend.grant_permission(
                GroupName("admins"),
                PermissionType.ADMIN,
                resource,
            )
            self.backend.revoke_permission(
                GroupName("admins"),
                PermissionType.ADMIN,
                resource,
            )

            # Verify permission was revoked
            permissions = self.backend.get_permissions(resource)
            assert len(permissions) == 0

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    def test_check_permission_granted(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Check permission returns True when user has permission."""
        with self.scenario(self.backend, scenario) as _:
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            group = Group(name=GroupName("admins"))
            self.backend.create_group(group)
            self.backend.add_user_to_group("alice", GroupName("admins"))
            self.backend.grant_permission(
                GroupName("admins"),
                PermissionType.READ,
                resource,
            )

            # Check permission
            has_permission = self.backend.check_permission(
                "alice", PermissionType.READ, resource
            )
            assert has_permission is True

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    def test_check_permission_denied(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Check permission returns False when user doesn't have permission."""
        with self.scenario(self.backend, scenario) as _:
            user = User(user_name="alice")
            self.backend.login(user)

            # Check permission
            has_permission = self.backend.check_permission(
                "alice", PermissionType.ADMIN, resource
            )
            assert has_permission is False

    def test_check_permission_nonexistent_user(self) -> None:
        """Check permission returns False for non-existent user."""
        with self.scenario(self.backend, "bare") as _:
            has_permission = self.backend.check_permission(
                "nonexistent", PermissionType.READ, BackendResourceType.SYSTEM
            )
            assert has_permission is False

    def test_check_collection_permission_nonexistent_collection(self) -> None:
        """Check permission errors for non-existent collection."""
        with self.scenario(self.backend, "bare") as _:
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            group = Group(name=GroupName("readers"))
            self.backend.create_group(group)
            self.backend.add_user_to_group("alice", GroupName("readers"))

            with pytest.raises(MatchboxCollectionNotFoundError):
                _ = self.backend.check_permission(
                    "alice", PermissionType.READ, "nonexistent"
                )

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    def test_get_permissions_empty(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Get permissions returns empty list when no permissions granted."""
        with self.scenario(self.backend, scenario) as _:
            permissions = self.backend.get_permissions(resource)
            assert permissions == []

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            (BackendResourceType.SYSTEM, "bare"),
            ("collection", "dedupe"),
        ],
        ids=["system", "collection"],
    )
    def test_get_permissions_multiple_groups(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Get permissions returns all permissions from multiple groups."""
        with self.scenario(self.backend, scenario) as _:
            group1 = Group(name=GroupName("readers"))
            group2 = Group(name=GroupName("writers"))
            self.backend.create_group(group1)
            self.backend.create_group(group2)

            self.backend.grant_permission(
                GroupName("readers"),
                PermissionType.READ,
                resource,
            )
            self.backend.grant_permission(
                GroupName("writers"),
                PermissionType.WRITE,
                resource,
            )

            permissions = self.backend.get_permissions(resource)
            assert len(permissions) == 2

            perm_dict = {p.group_name: p.permission for p in permissions}
            assert perm_dict["readers"] == PermissionType.READ
            assert perm_dict["writers"] == PermissionType.WRITE

    def test_permissions_across_multiple_groups(self) -> None:
        """User inherits permissions from all their groups."""
        with self.scenario(self.backend, "dedupe") as _:
            # Create user
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            # Create two groups with different permissions
            readers = Group(name=GroupName("readers"))
            writers = Group(name=GroupName("writers"))
            self.backend.create_group(readers)
            self.backend.create_group(writers)

            self.backend.grant_permission(
                GroupName("readers"),
                PermissionType.READ,
                "collection",
            )
            self.backend.grant_permission(
                GroupName("writers"),
                PermissionType.WRITE,
                "collection",
            )

            # Add user to both groups
            self.backend.add_user_to_group("alice", GroupName("readers"))
            self.backend.add_user_to_group("alice", GroupName("writers"))

            # User should have both permissions
            assert self.backend.check_permission(
                "alice", PermissionType.READ, "collection"
            )
            assert self.backend.check_permission(
                "alice", PermissionType.WRITE, "collection"
            )

    # Data management

    def test_validate_ids(self) -> None:
        """Test validating data IDs."""
        with self.scenario(self.backend, "dedupe") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            df_crn = self.backend.query(
                source=crn_testkit.source.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )

            ids = df_crn["id"].to_pylist()
            assert len(ids) > 0
            self.backend.validate_ids(ids=ids)

            with pytest.raises(MatchboxDataNotFound):
                self.backend.validate_ids(ids=[-6])

    def test_clear(self) -> None:
        """Test deleting all rows in the database."""
        with self.scenario(self.backend, "dedupe"):
            assert self.backend.sources.count() > 0
            assert self.backend.source_clusters.count() > 0
            assert self.backend.models.count() > 0
            assert self.backend.model_clusters.count() > 0
            assert self.backend.creates.count() > 0
            assert self.backend.merges.count() > 0
            assert self.backend.proposes.count() > 0

            self.backend.clear(certain=True)

            assert self.backend.sources.count() == 0
            assert self.backend.source_clusters.count() == 0
            assert self.backend.models.count() == 0
            assert self.backend.model_clusters.count() == 0
            assert self.backend.creates.count() == 0
            assert self.backend.merges.count() == 0
            assert self.backend.proposes.count() == 0

    def test_clear_and_restore(self) -> None:
        """Test that clearing and restoring the database works."""
        with self.scenario(self.backend, "link") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            count_funcs = [
                self.backend.sources.count,
                self.backend.models.count,
                self.backend.source_clusters.count,
                self.backend.model_clusters.count,
                self.backend.all_clusters.count,
                self.backend.merges.count,
                self.backend.creates.count,
                self.backend.proposes.count,
            ]

            def get_counts() -> list[int]:
                return [f() for f in count_funcs]

            # Verify we have data
            pre_dump_counts = get_counts()
            assert all(count > 0 for count in pre_dump_counts)

            # Get some specific IDs to verify they're restored properly
            df_crn_before = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )
            sample_ids_before = df_crn_before["id"].to_pylist()[:5]  # Take first 5 IDs

            # Dump the database
            snapshot = self.backend.dump()

        with self.scenario(self.backend, "bare") as _:
            # Verify counts match pre-dump state
            assert all(c == 0 for c in get_counts())

            # Restore from snapshot
            self.backend.restore(snapshot)

            # Verify counts match pre-dump state
            assert get_counts() == pre_dump_counts

            # Verify specific data was restored correctly
            df_crn_after = self.backend.query(
                source=crn_testkit.resolution_path,
                point_of_truth=naive_crn_testkit.resolution_path,
            )
            sample_ids_after = df_crn_after["id"].to_pylist()[:5]  # Take first 5 IDs

            # The same IDs should be present after restoration
            assert set(sample_ids_before) == set(sample_ids_after)

            # Test that restoring also clears the database
            self.backend.restore(snapshot)

            # Verify counts still match
            assert get_counts() == pre_dump_counts

    def test_delete_orphans(self) -> None:
        """Can delete orphaned clusters."""
        with self.scenario(self.backend, "link") as dag_testkit:
            crn_testkit = dag_testkit.sources.get("crn")
            naive_crn_testkit = dag_testkit.models.get("naive_test_crn")

            # Get number of clusters
            initial_all_clusters = self.backend.all_clusters.count()

            # Delete orphans, none should be deleted yet
            orphans = self.backend.delete_orphans()
            assert orphans == 0
            assert initial_all_clusters == self.backend.all_clusters.count()

            # TODO: insert judgement for cluster, check that it is not deleted when
            # deleting model resolution. Then deleting the judgement should cause
            # exactly 1 orphan.

            model_res = naive_crn_testkit.resolution_path
            self.backend.delete_resolution(model_res, certain=True)

            # Delete orphans, some should be deleted and total clusters should reduce
            orphans = self.backend.delete_orphans()
            assert orphans > 0
            all_clusters_2 = self.backend.all_clusters.count()
            assert initial_all_clusters > all_clusters_2

            # Delete source resolution crn
            source_res = crn_testkit.resolution_path
            self.backend.delete_resolution(source_res, certain=True)

            # Delete orphans again and check number of clusters has reduced
            orphans = self.backend.delete_orphans()
            assert orphans > 0
            all_clusters_3 = self.backend.all_clusters.count()
            assert all_clusters_2 > all_clusters_3
