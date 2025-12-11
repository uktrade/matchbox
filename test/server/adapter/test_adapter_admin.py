"""Test the backend adapter's admin functions."""

from functools import partial

import pytest
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.dtos import (
    BackendResourceType,
    Group,
    GroupName,
    PermissionGrant,
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
        self, backend_instance: MatchboxDBAdapter, sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

    # Default database state

    def test_default_public_user_exists(self) -> None:
        """The _public user exists by default."""
        with self.scenario(self.backend, "bare") as _:
            # The _public user should exist even in bare scenario
            groups = self.backend.get_user_groups("_public")
            assert GroupName("public") in groups

    def test_default_groups_exist(self) -> None:
        """The admins and public groups exist by default."""
        with self.scenario(self.backend, "bare") as _:
            groups = self.backend.list_groups()
            group_names = {g.name for g in groups}
            assert "admins" in group_names
            assert "public" in group_names
            assert len(groups) == 2

            # Verify admins group properties
            admins = self.backend.get_group(GroupName("admins"))
            assert admins.name == "admins"
            assert admins.description == "System administrators."
            assert admins.is_system is True
            assert len(admins.members) == 0  # Empty until first login

            # Verify public group properties
            public = self.backend.get_group(GroupName("public"))
            assert public.name == "public"
            assert public.is_system is True
            assert len(public.members) == 1  # Contains _public user

    def test_default_system_permissions(self) -> None:
        """The admins group has system admin permission by default."""
        with self.scenario(self.backend, "bare") as _:
            permissions = self.backend.get_permissions(BackendResourceType.SYSTEM)
            assert len(permissions) == 1
            assert permissions[0].group_name == "admins"
            assert permissions[0].permission == PermissionType.ADMIN

    # User management

    def test_login_creates_new_user(self) -> None:
        """Login creates a new user if they don't exist."""
        with self.scenario(self.backend, "admin") as _:
            user = User(user_name="bob", email="bob@example.com")
            result = self.backend.login(user)

            assert result.user.user_name == "bob"
            assert result.user.email == "bob@example.com"
            assert result.setup_mode_admin is False  # admin user already exists

    def test_login_adds_new_user_to_public_group(self) -> None:
        """Login adds new users to the public group."""
        with self.scenario(self.backend, "admin") as _:
            user = User(user_name="bob", email="bob@example.com")
            self.backend.login(user)

            # Verify user was added to public group
            user_groups = self.backend.get_user_groups("bob")
            assert GroupName("public") in user_groups

            # Verify public group now has bob as a member
            public = self.backend.get_group(GroupName("public"))
            member_names = {m.user_name for m in public.members}
            assert "bob" in member_names
            assert "_public" in member_names  # Default user still present
            assert "alice" in member_names  # From admin scenario setup

    def test_login_returns_existing_user(self) -> None:
        """Login returns existing user with same ID."""
        with self.scenario(self.backend, "admin") as _:
            user1 = User(user_name="bob", email="bob@example.com")
            result1 = self.backend.login(user1)

            user2 = User(user_name="bob", email="bob@example.com")
            result2 = self.backend.login(user2)

            assert result1.user.user_name == result2.user.user_name
            assert result2.setup_mode_admin is False  # Alice was first

    def test_login_updates_email(self) -> None:
        """Login updates email if it changes."""
        with self.scenario(self.backend, "admin") as _:
            user1 = User(user_name="bob", email="bob@example.com")
            _ = self.backend.login(user1)

            user2 = User(user_name="bob", email="bob@newdomain.com")
            result2 = self.backend.login(user2)

            assert result2.user.email == "bob@newdomain.com"
            assert result2.setup_mode_admin is False  # Alice was first

    def test_login_different_users(self) -> None:
        """Login creates different IDs for different users."""
        with self.scenario(self.backend, "bare") as _:
            alice = User(user_name="alice")
            bob = User(user_name="bob")

            alice_result = self.backend.login(alice)
            bob_result = self.backend.login(bob)

            assert alice_result.user.user_name != bob_result.user.user_name
            assert alice_result.setup_mode_admin is True  # First user
            assert bob_result.setup_mode_admin is False  # Second user

    def test_login_first_user_added_to_admins_and_public(self) -> None:
        """First user login automatically adds them to admins and public groups."""
        with self.scenario(self.backend, "bare") as _:
            # Verify admins group exists but has no real users (only _public in public)
            admins = self.backend.get_group(GroupName("admins"))
            assert len(admins.members) == 0

            public = self.backend.get_group(GroupName("public"))
            assert len(public.members) == 1
            assert public.members[0].user_name == "_public"

            # First user login
            first_user = User(user_name="alice", email="alice@example.com")
            result = self.backend.login(first_user)

            # Verify response indicates setup mode
            assert result.setup_mode_admin is True
            assert result.user.user_name == "alice"

            # Verify user was added to both admins and public groups
            user_groups = self.backend.get_user_groups("alice")
            assert GroupName("admins") in user_groups
            assert GroupName("public") in user_groups

            # Verify admins group now has the user as a member
            admins = self.backend.get_group(GroupName("admins"))
            assert len(admins.members) == 1
            assert admins.members[0].user_name == "alice"

            # Verify public group now has alice and _public
            public = self.backend.get_group(GroupName("public"))
            assert len(public.members) == 2
            member_names = {m.user_name for m in public.members}
            assert member_names == {"alice", "_public"}

            # Second user login should only be added to public
            second_user = User(user_name="bob", email="bob@example.com")
            result2 = self.backend.login(second_user)

            # Verify response indicates normal mode
            assert result2.setup_mode_admin is False

            # Verify second user was NOT added to admins group
            bob_groups = self.backend.get_user_groups("bob")
            assert GroupName("admins") not in bob_groups
            assert GroupName("public") in bob_groups

            # Verify admins group still has only alice
            admins = self.backend.get_group(GroupName("admins"))
            assert len(admins.members) == 1
            assert admins.members[0].user_name == "alice"

            # Verify public group now has alice, bob, and _public
            public = self.backend.get_group(GroupName("public"))
            assert len(public.members) == 3
            member_names = {m.user_name for m in public.members}
            assert member_names == {"alice", "bob", "_public"}

    # Group management

    def test_create_group(self) -> None:
        """Can create a new group."""
        with self.scenario(self.backend, "bare") as _:
            group = Group(name=GroupName("g"), description="Test group")
            self.backend.create_group(group)

            # Verify it was created
            retrieved = self.backend.get_group(GroupName("g"))
            assert retrieved.name == "g"
            assert retrieved.description == "Test group"
            assert retrieved.is_system is False
            assert len(retrieved.members) == 0

    def test_create_group_duplicate_fails(self) -> None:
        """Cannot create a group with duplicate name."""
        with self.scenario(self.backend, "bare") as _:
            group1 = Group(name=GroupName("g"))
            self.backend.create_group(group1)

            group2 = Group(name=GroupName("g"))
            with pytest.raises(MatchboxGroupAlreadyExistsError):
                self.backend.create_group(group2)

    def test_list_groups(self) -> None:
        """List groups returns all groups."""
        with self.scenario(self.backend, "bare") as _:
            group1 = Group(name=GroupName("g"), description="Group")
            group2 = Group(name=GroupName("users"), description="Users")

            self.backend.create_group(group1)
            self.backend.create_group(group2)

            groups = self.backend.list_groups()
            assert len(groups) == 4  # admins, public, g, users
            assert {g.name for g in groups} == {"admins", "public", "g", "users"}

    def test_get_group_not_found(self) -> None:
        """Get group raises error if group doesn't exist."""
        with self.scenario(self.backend, "bare") as _:  # noqa: SIM117
            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.get_group(GroupName("nonexistent"))

    def test_delete_group(self) -> None:
        """Can delete a group."""
        with self.scenario(self.backend, "bare") as _:
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            # Verify it exists
            retrieved = self.backend.get_group(GroupName("g"))
            assert retrieved.name == "g"

            # Delete it
            self.backend.delete_group(GroupName("g"), certain=True)

            # Verify it's gone
            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.get_group(GroupName("g"))

    def test_delete_group_requires_confirmation(self) -> None:
        """Delete group requires certain=True."""
        with self.scenario(self.backend, "bare") as _:
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            with pytest.raises(MatchboxDeletionNotConfirmed):
                self.backend.delete_group(GroupName("g"), certain=False)

    def test_delete_group_not_found(self) -> None:
        """Delete group raises error if group doesn't exist."""
        with self.scenario(self.backend, "bare") as _:  # noqa: SIM117
            with pytest.raises(MatchboxGroupNotFoundError):
                self.backend.delete_group(GroupName("nonexistent"), certain=True)

    def test_delete_system_group_fails(self) -> None:
        """Cannot delete a system group."""
        with self.scenario(self.backend, "bare") as _:
            # Try to delete the default admins group
            with pytest.raises(MatchboxSystemGroupError):
                self.backend.delete_group(GroupName("admins"), certain=True)

            # Try to delete the default public group
            with pytest.raises(MatchboxSystemGroupError):
                self.backend.delete_group(GroupName("public"), certain=True)

    # User-group membership

    def test_add_user_to_group(self) -> None:
        """Can add a user to a group."""
        with self.scenario(self.backend, "bare") as _:
            # Create group
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            # Create user
            user = User(user_name="alice")
            self.backend.login(user)

            # Add user to group
            self.backend.add_user_to_group("alice", GroupName("g"))

            # Verify membership
            groups = self.backend.get_user_groups("alice")
            assert GroupName("g") in groups

    def test_add_user_to_nonexistent_user_fails(self) -> None:
        """Cannot add non-existent user to group."""
        with self.scenario(self.backend, "bare") as _:
            # Create group
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            # Try to add non-existent user to group
            with pytest.raises(MatchboxUserNotFoundError):
                self.backend.add_user_to_group("nonexistent", GroupName("g"))

    def test_add_user_to_group_idempotent(self) -> None:
        """Adding user to group twice doesn't cause error."""
        with self.scenario(self.backend, "bare") as _:
            user = User(user_name="alice", email="alice@example.com")
            self.backend.login(user)

            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            self.backend.add_user_to_group("alice", GroupName("g"))
            self.backend.add_user_to_group("alice", GroupName("g"))

            groups = self.backend.get_user_groups("alice")
            assert GroupName("g") in groups

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

            group = Group(name=GroupName("g"))
            self.backend.create_group(group)
            self.backend.add_user_to_group("alice", GroupName("g"))

            # Verify user is in group
            groups = self.backend.get_user_groups("alice")
            assert GroupName("g") in groups

            # Remove user from group
            self.backend.remove_user_from_group("alice", GroupName("g"))

            # Verify user is no longer in group
            groups = self.backend.get_user_groups("alice")
            assert GroupName("g") not in groups

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
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            with pytest.raises(MatchboxUserNotFoundError):
                self.backend.remove_user_from_group("nonexistent", GroupName("g"))

    def test_get_user_groups(self) -> None:
        """Get user groups returns public group for new users."""
        with self.scenario(self.backend, "admin") as _:
            user = User(user_name="bob")
            self.backend.login(user)

            groups = self.backend.get_user_groups("bob")
            # All new users are added to public group
            assert GroupName("public") in groups
            assert len(groups) == 1  # Bob is not in admins, only public

    def test_get_user_groups_nonexistent_user_fails(self) -> None:
        """Get user groups raises error for non-existent user."""
        with self.scenario(self.backend, "bare") as _:  # noqa: SIM117
            with pytest.raises(MatchboxUserNotFoundError):
                self.backend.get_user_groups("nonexistent")

    def test_get_user_groups_multiple(self) -> None:
        """Get user groups returns all groups for a user."""
        with self.scenario(self.backend, "admin") as _:
            # Create user (alice already exists)
            user = User(user_name="bob", email="bob@example.com")
            self.backend.login(user)

            # Create groups
            group1 = Group(name=GroupName("g"))
            group2 = Group(name=GroupName("users"))
            self.backend.create_group(group1)
            self.backend.create_group(group2)

            # Add user to both groups
            self.backend.add_user_to_group("bob", GroupName("g"))
            self.backend.add_user_to_group("bob", GroupName("users"))

            # Verify membership
            groups = self.backend.get_user_groups("bob")
            assert len(groups) == 3  # public (auto), g, users
            assert set(groups) == {
                GroupName("public"),
                GroupName("g"),
                GroupName("users"),
            }

    def test_get_group_includes_members(self) -> None:
        """Get group returns members list."""
        with self.scenario(self.backend, "bare") as _:
            # Create group and users
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            alice = User(user_name="alice", email="alice@example.com")
            bob = User(user_name="bob", email="bob@example.com")
            self.backend.login(alice)
            self.backend.login(bob)

            # Add users to group
            self.backend.add_user_to_group("alice", GroupName("g"))
            self.backend.add_user_to_group("bob", GroupName("g"))

            # Get group and verify members
            retrieved = self.backend.get_group(GroupName("g"))
            assert len(retrieved.members) == 2
            member_names = {m.user_name for m in retrieved.members}
            assert member_names == {"alice", "bob"}

    # Permissions

    @pytest.mark.parametrize(
        ("resource_type", "scenario"),
        [
            pytest.param("system", "admin", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
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
        resource_type: str,
        scenario: str,
        granted_permission: PermissionType,
        can_read: bool,
        can_write: bool,
        can_admin: bool,
    ) -> None:
        """Test permission hierarchy: ADMIN > WRITE > READ."""
        with self.scenario(self.backend, scenario) as _:
            # Determine resource: use system or create isolated collection
            if resource_type == "system":
                resource = BackendResourceType.SYSTEM
            else:
                # Create new collection with no permissions
                # to avoid public group interference
                collection_name = f"col_{granted_permission}"
                self.backend.create_collection(
                    name=collection_name,
                    permissions=[],  # No default permissions
                )
                resource = collection_name

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
            pytest.param(BackendResourceType.SYSTEM, "bare", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
    )
    def test_grant_permission(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Can grant permission to a group."""
        with self.scenario(self.backend, scenario) as _:
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            self.backend.grant_permission(
                GroupName("g"),
                PermissionType.ADMIN,
                resource,
            )

            # Verify permission was granted
            expected_permission = PermissionGrant(
                group_name="g", permission=PermissionType.ADMIN
            )

            permissions = self.backend.get_permissions(resource)
            assert expected_permission in permissions

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            pytest.param(BackendResourceType.SYSTEM, "bare", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
    )
    def test_grant_permission_idempotent(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Granting same permission twice doesn't cause error."""
        with self.scenario(self.backend, scenario) as _:
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            self.backend.grant_permission(
                GroupName("g"),
                PermissionType.READ,
                resource,
            )
            self.backend.grant_permission(
                GroupName("g"),
                PermissionType.READ,
                resource,
            )

            permissions = self.backend.get_permissions(resource)
            assert len(set(permissions)) == len(permissions)

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            pytest.param(BackendResourceType.SYSTEM, "bare", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
    )
    def test_revoke_permission(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Can revoke permission from a group."""
        with self.scenario(self.backend, scenario) as _:
            group = Group(name=GroupName("g"))
            self.backend.create_group(group)

            # Grant then revoke
            self.backend.grant_permission(
                GroupName("g"),
                PermissionType.ADMIN,
                resource,
            )
            self.backend.revoke_permission(
                GroupName("g"),
                PermissionType.ADMIN,
                resource,
            )

            # Verify permission was revoked
            expected_permission = PermissionGrant(
                group_name="g", permission=PermissionType.ADMIN
            )

            permissions = self.backend.get_permissions(resource)
            assert expected_permission not in permissions

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            pytest.param(BackendResourceType.SYSTEM, "bare", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
    )
    def test_check_permission_granted(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Check permission returns True when user has permission."""
        with self.scenario(self.backend, scenario) as _:
            user = User(user_name="bob", email="alice@example.com")
            self.backend.login(user)

            group = Group(name=GroupName("g"))
            self.backend.create_group(group)
            self.backend.add_user_to_group("bob", GroupName("g"))
            self.backend.grant_permission(
                GroupName("g"),
                PermissionType.READ,
                resource,
            )

            # Check permission
            has_permission = self.backend.check_permission(
                "bob", PermissionType.READ, resource
            )
            assert has_permission is True

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            pytest.param(BackendResourceType.SYSTEM, "admin", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
    )
    def test_check_permission_denied(
        self, resource: str | BackendResourceType, scenario: str
    ) -> None:
        """Check permission returns False when user doesn't have permission.

        Note: In non-bare scenarios, alice already exists as the admin user.
        Bob is a new user without special permissions.
        """
        with self.scenario(self.backend, scenario) as _:
            user = User(user_name="bob")
            self.backend.login(user)

            # Check bob only has public group membership, no special permissions
            has_permission = self.backend.check_permission(
                "bob", PermissionType.ADMIN, resource
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
        ("resource_type", "scenario"),
        [
            pytest.param("system", "bare", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
    )
    def test_get_permissions(
        self,
        resource_type: str,
        scenario: str,
    ) -> None:
        """Get permissions returns correct default permissions."""
        with self.scenario(self.backend, scenario) as _:
            # Determine resource based on type
            if resource_type == "system":
                resource = BackendResourceType.SYSTEM
            else:
                # Create new collection with no permissions
                self.backend.create_collection(
                    name="test_no_permissions",
                    permissions=[],
                )
                resource = "test_no_permissions"

            permissions = self.backend.get_permissions(resource)

            if resource_type == "system":
                # System should have default admin permission for admins group
                assert len(permissions) == 1
                assert permissions[0].group_name == "admins"
                assert permissions[0].permission == PermissionType.ADMIN
            else:
                # This collection has no permissions
                assert permissions == []

    @pytest.mark.parametrize(
        ("resource", "scenario"),
        [
            pytest.param(BackendResourceType.SYSTEM, "bare", id="system"),
            pytest.param("collection", "dedupe", id="collection"),
        ],
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
            perm_dict = {p.group_name: p.permission for p in permissions}
            assert perm_dict["readers"] == PermissionType.READ
            assert perm_dict["writers"] == PermissionType.WRITE

    def test_permissions_across_multiple_groups(self) -> None:
        """User inherits permissions from all their groups.

        Note: In 'dedupe' scenario, alice already exists as the admin user.
        Creating a new alice user here for testing multiple group permissions.
        """
        with self.scenario(self.backend, "dedupe") as _:
            # Create user (different from default alice)
            user = User(user_name="charlie", email="charlie@example.com")
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
            self.backend.add_user_to_group("charlie", GroupName("readers"))
            self.backend.add_user_to_group("charlie", GroupName("writers"))

            # User should have both permissions
            assert self.backend.check_permission(
                "charlie", PermissionType.READ, "collection"
            )
            assert self.backend.check_permission(
                "charlie", PermissionType.WRITE, "collection"
            )

    def test_public_group_permissions_inherited(self) -> None:
        """Users automatically inherit permissions from the public group."""
        with self.scenario(self.backend, "dedupe") as _:
            # Create new collection with only READ permission for public
            self.backend.create_collection(
                name="col2",
                permissions=[
                    PermissionGrant(
                        group_name=GroupName("public"),
                        permission=PermissionType.READ,
                    )
                ],
            )

            # Create new user (alice already exists, so use bob)
            user = User(user_name="bob", email="bob@example.com")
            self.backend.login(user)

            # User should have read permission through public group membership
            assert self.backend.check_permission("bob", PermissionType.READ, "col2")

            # But not write permission
            assert not self.backend.check_permission(
                "bob", PermissionType.WRITE, "col2"
            )

            # Alice should also have the permission through public group
            assert self.backend.check_permission("alice", PermissionType.READ, "col2")

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
