"""Integration tests with real scenario data - comprehensive behaviour testing."""

from collections.abc import Callable
from functools import partial
from unittest.mock import Mock

import pytest
from sqlalchemy import Engine
from textual.widgets import Footer, Label

from matchbox.client.cli.eval.app import EntityResolutionApp, EvaluationQueue
from matchbox.client.cli.eval.widgets.table import ComparisonDisplayTable
from matchbox.client.dags import DAG
from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.scenarios import setup_scenario
from matchbox.server.base import MatchboxDBAdapter

backends = [
    pytest.param("matchbox_postgres", id="postgres"),
]


@pytest.fixture(scope="function")
def backend_instance(request: pytest.FixtureRequest, backend: str) -> MatchboxDBAdapter:
    """Create a fresh backend instance for each test."""
    backend_obj = request.getfixturevalue(backend)
    backend_obj.clear(certain=True)
    return backend_obj


class TestEvaluationQueue:
    """Unit tests for EvaluationQueue - no app or mocking needed."""

    def test_queue_initialises_empty(self) -> None:
        """Test queue starts empty."""
        queue = EvaluationQueue()

        assert queue.total_count == 0
        assert queue.current is None

    def test_add_items_increases_count(self) -> None:
        """Test adding items increases count."""
        queue = EvaluationQueue()
        items = [Mock(item=Mock(leaves=[1, 2])), Mock(item=Mock(leaves=[3, 4]))]

        added = queue.add_sessions(items)

        assert added == 2
        assert queue.total_count == 2
        assert queue.current is items[0]

    def test_add_items_prevents_duplicates(self) -> None:
        """Test that duplicate cluster IDs are not added."""
        queue = EvaluationQueue()
        item1 = Mock(item=Mock(leaves=[1, 2]))
        item2 = Mock(item=Mock(leaves=[2, 1]))  # Duplicate

        queue.add_sessions([item1])
        added = queue.add_sessions([item2])

        assert added == 0
        assert queue.total_count == 1

    def test_add_items_handles_empty_list(self) -> None:
        """Test that adding empty list returns 0."""
        queue = EvaluationQueue()

        added = queue.add_sessions([])

        assert added == 0
        assert queue.total_count == 0

    def test_skip_rotates_deque(self) -> None:
        """Test that skip moves current to back."""
        queue = EvaluationQueue()
        item1 = Mock(item=Mock(cluster_id=1))
        item2 = Mock(item=Mock(cluster_id=2))
        queue.sessions.extend([item1, item2])

        queue.skip_current()

        assert queue.current is item2
        assert queue.sessions[1] is item1

    def test_skip_with_single_item_does_nothing(self) -> None:
        """Test that skip with one item doesn't rotate."""
        queue = EvaluationQueue()
        item1 = Mock(item=Mock(cluster_id=1))
        queue.sessions.append(item1)

        queue.skip_current()

        assert queue.current is item1
        assert queue.total_count == 1

    def test_remove_current_pops_front(self) -> None:
        """Test that remove_current removes from front."""
        queue = EvaluationQueue()
        item1 = Mock(item=Mock(cluster_id=1))
        item2 = Mock(item=Mock(cluster_id=2))
        queue.sessions.extend([item1, item2])

        removed = queue.remove_current()

        assert removed is item1
        assert queue.total_count == 1
        assert queue.current is item2

    def test_remove_current_on_empty_returns_none(self) -> None:
        """Test that remove_current on empty queue returns None."""
        queue = EvaluationQueue()

        removed = queue.remove_current()

        assert removed is None
        assert queue.total_count == 0


@pytest.mark.parametrize("backend", backends)
@pytest.mark.docker
class TestScenarioIntegration:
    """Integration tests using real scenario data with backend."""

    @pytest.fixture
    def test_resolution(self) -> ModelResolutionPath:
        """Create test resolution path."""
        return ModelResolutionPath(
            collection="test_collection", run=1, name="test_resolution"
        )

    @pytest.fixture(autouse=True)
    def setup(self, backend_instance: str, sqla_sqlite_warehouse: Engine) -> None:
        """Set up test fixtures."""
        self.backend: MatchboxDBAdapter = backend_instance
        self.warehouse_engine: Engine = sqla_sqlite_warehouse
        self.scenario: Callable[..., TestkitDAG] = partial(
            setup_scenario, warehouse=sqla_sqlite_warehouse
        )

    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self) -> None:
        """Test complete evaluation workflow: start, load, label, clear, skip, submit.

        This comprehensive test covers the main user journey through the evaluation app:

        - App initialisation and sample loading
        - Keyboard-driven assignment workflow (letter → digit)
        - Clearing assignments
        - Skipping items in the queue
        - Submitting incomplete assignments (warning)
        - Completing and submitting assignments (saves judgement)
        - Status updates and help modal
        """
        with self.scenario(self.backend, "mega") as dag:
            model_name: str = "mega_product_linker"

            loaded_dag: DAG = dag.dag.load_pending().set_client(self.warehouse_engine)

            app = EntityResolutionApp(
                resolution=loaded_dag.get_model(model_name),
                num_samples=3,
                user="test_user",
                dag=loaded_dag,
                scroll_debounce_delay=None,
            )

            # Resolution carefully chosen so all columns on a page are always numbered
            async with app.run_test(size=(250, 150)) as pilot:
                await pilot.pause()

                # 1. Verify app initialisation and sample loading
                assert app.is_running
                assert app.user_name == "test_user"
                assert pilot.app.query("Footer")
                assert app.queue.total_count > 0
                assert app.queue.current is not None
                assert app.queue.current.item.leaves is not None

                # 2. Test keyboard workflow: letter then digit
                await pilot.press("a")
                await pilot.pause()
                table = app.query_one(ComparisonDisplayTable)
                assert table.current_group == "a"

                await pilot.press("1")
                await pilot.pause()
                current = app.queue.current
                assert current is not None
                assert 0 in current.assignments
                assert current.assignments[0] == "a"

                # 3. Verify assignment was saved
                assert len(current.assignments) == 1
                assert 0 in current.assignments

                # 3b. Test paging - scroll right and label different column
                initial_col_idx = table.cursor_column
                initial_scroll_x = table.scroll_x
                await pilot.press("right")
                await pilot.pause()

                # Verify paging behaviour
                assert table.scroll_x > initial_scroll_x, "Paging right should scroll"
                assert table.cursor_column != initial_col_idx, (
                    "Paging right should move cursor/headers"
                )

                # We pressed right, so '1' should now map to a new column
                new_cursor_idx = table.cursor_column
                await pilot.press("b")
                await pilot.press("1")
                await pilot.pause()
                expected_assignment_idx = new_cursor_idx - 1  # Cursor is 0-indexed
                assert expected_assignment_idx in current.assignments
                assert current.assignments[expected_assignment_idx] == "b"

                # Scroll back to start
                await pilot.press("left")
                await pilot.pause()
                assert table.scroll_x == initial_scroll_x

                # 4. Test status updates reactively
                assert isinstance(app.status, tuple)
                assert len(app.status) == 2

                # 5. Test clearing assignments
                await pilot.press("escape")
                await pilot.pause()
                assert len(current.assignments) == 0

                # 6. Test skip workflow
                first_item = app.queue.current
                await pilot.press("shift+right")
                await pilot.pause()
                assert app.queue.current is not first_item
                assert app.queue.sessions[-1] is first_item

                # 7. Test submitting incomplete assignment shows warning
                initial_count = app.queue.total_count
                await pilot.press("space")
                await pilot.pause()
                assert app.queue.total_count == initial_count
                status_message = app.status[0].lower()
                assert "incomplete" in status_message

                # 8. Test completing all columns via paging and successful submission
                current = app.queue.current
                assert current is not None

                initial_judgements, _ = self.backend.get_judgements()
                initial_judgement_count = len(initial_judgements)

                unique_groups = current.item.get_unique_record_groups()
                target_count = len(unique_groups)

                # Set group to assign
                await pilot.press("a")
                await pilot.pause()

                # Robust loop: Fill visible page, scroll right, repeat.
                iteration: int = 0
                while len(current.assignments) < target_count:
                    if iteration > 20:
                        pytest.fail("Failed to assign all columns in 20 iterations")

                    # Assign visible batch (1-9) to group 'a'
                    for key in "123456789":
                        await pilot.press(key)
                        await pilot.pause()

                    if len(current.assignments) == target_count:
                        break

                    await pilot.press("right")
                    await pilot.pause()
                    iteration += 1

                # Verify all columns are now assigned
                assert len(current.assignments) == target_count, (
                    "Not all columns assigned: "
                    f"{len(current.assignments)}/{target_count}\n"
                    f"Assignments: {sorted(current.assignments.keys())}\n"
                )

                # Now submit should succeed
                await pilot.press("space")
                await pilot.pause()

                final_judgements, _ = self.backend.get_judgements()
                assert len(final_judgements) == initial_judgement_count + 1, (
                    "Judgement was not saved after complete assignment"
                )

    @pytest.mark.asyncio
    async def test_app_widgets_are_visible(self) -> None:
        """Test that all app widgets are visible with non-zero dimensions."""
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = dag.dag.load_pending().set_client(self.warehouse_engine)

            app = EntityResolutionApp(
                resolution=loaded_dag.get_model(model_name),
                num_samples=3,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # Check table
                table = app.query_one(ComparisonDisplayTable)
                assert table is not None
                assert table.size.width > 0, "Table has no width - not rendering"
                assert table.size.height > 0, "Table has no height - not rendering"
                assert table.row_count > 0, "Table has no rows"
                assert len(table.ordered_columns) > 0, "Table has no columns"

                # Check status bar widgets
                current_group_label = app.query_one("#current-group-label", Label)
                assert current_group_label is not None
                assert current_group_label.size.width > 0, (
                    "Current group label not rendering"
                )
                assert current_group_label.size.height > 0

                status_right = app.query_one("#status-right", Label)
                assert status_right is not None
                assert status_right.size.width > 0, "Status right not rendering"
                assert status_right.size.height > 0

                # Check footer
                footer = app.query_one(Footer)
                assert footer is not None
                assert footer.size.width > 0, "Footer not rendering"
                assert footer.size.height > 0
