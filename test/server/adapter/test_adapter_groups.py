"""Test the backend adapter's admin functions."""

from functools import partial

import pytest
from sqlalchemy import Engine
from test.fixtures.db import BACKENDS

from matchbox.common.dtos import (
    DefaultGroup,
    Group,
    GroupName,
    User,
)
from matchbox.common.exceptions import (
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
class TestMatchboxGroupsBackend:
    @pytest.fixture(autouse=True)
    def setup(
        self, backend_instance: MatchboxDBAdapter, sqla_sqlite_warehouse: Engine
    ) -> None:
        self.backend: MatchboxDBAdapter = backend_instance
        self.scenario = partial(setup_scenario, warehouse=sqla_sqlite_warehouse)

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
            assert {g.name for g in groups} == {
                DefaultGroup.ADMINS,
                DefaultGroup.PUBLIC,
                "g",
                "users",
            }

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
                self.backend.delete_group(GroupName(DefaultGroup.ADMINS), certain=True)

            # Try to delete the default public group
            with pytest.raises(MatchboxSystemGroupError):
                self.backend.delete_group(GroupName(DefaultGroup.PUBLIC), certain=True)

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
            assert GroupName(DefaultGroup.PUBLIC) in groups
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
                GroupName(DefaultGroup.PUBLIC),
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
