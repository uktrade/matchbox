"""Integration tests with real scenario data - comprehensive behaviour testing."""

from functools import partial
from unittest.mock import Mock

import pytest
from sqlalchemy import Engine

from matchbox.client.cli.eval.app import EntityResolutionApp, EvaluationQueue
from matchbox.client.dags import DAG
from matchbox.common.dtos import ModelResolutionPath
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
        items = [Mock(cluster_id=1), Mock(cluster_id=2)]

        added = queue.add_items(items)

        assert added == 2
        assert queue.total_count == 2
        assert queue.current is items[0]

    def test_add_items_prevents_duplicates(self) -> None:
        """Test that duplicate cluster IDs are not added."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=1)  # Duplicate

        queue.add_items([item1])
        added = queue.add_items([item2])

        assert added == 0
        assert queue.total_count == 1

    def test_add_items_handles_empty_list(self) -> None:
        """Test that adding empty list returns 0."""
        queue = EvaluationQueue()

        added = queue.add_items([])

        assert added == 0
        assert queue.total_count == 0

    def test_skip_rotates_deque(self) -> None:
        """Test that skip moves current to back."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=2)
        queue.items.extend([item1, item2])

        queue.skip_current()

        assert queue.current is item2
        assert queue.items[1] is item1

    def test_skip_with_single_item_does_nothing(self) -> None:
        """Test that skip with one item doesn't rotate."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        queue.items.append(item1)

        queue.skip_current()

        assert queue.current is item1
        assert queue.total_count == 1

    def test_remove_current_pops_front(self) -> None:
        """Test that remove_current removes from front."""
        queue = EvaluationQueue()
        item1 = Mock(cluster_id=1)
        item2 = Mock(cluster_id=2)
        queue.items.extend([item1, item2])

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
    def setup(self, backend_instance: str, sqlite_warehouse: Engine) -> None:
        """Set up test fixtures."""
        self.backend = backend_instance
        self.warehouse_engine = sqlite_warehouse
        self.scenario = partial(setup_scenario, warehouse=sqlite_warehouse)

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

        Additional edge cases (e.g., clusters too large for screen) should be
        separate tests as they may require different setup or scenarios.
        """
        with self.scenario(self.backend, "dedupe") as dag:
            model_name: str = "naive_test_crn"

            loaded_dag: DAG = (
                DAG(str(dag.dag.name)).load_pending().set_client(self.warehouse_engine)
            )

            app = EntityResolutionApp(
                resolution=model_name,
                num_samples=3,
                user="test_user",
                dag=loaded_dag,
            )

            async with app.run_test() as pilot:
                await pilot.pause()

                # 1. Verify app initialisation and sample loading
                assert app.is_running
                assert app.user_name == "test_user"
                assert app.user_id is not None
                assert pilot.app.query("Footer")
                assert app.queue.total_count > 0
                assert app.queue.current is not None
                assert app.queue.current.cluster_id is not None

                # 2. Test keyboard workflow: letter then digit
                await pilot.press("a")
                await pilot.pause()
                assert app.current_group == "a"

                await pilot.press("1")
                await pilot.pause()
                current = app.queue.current
                assert current is not None
                assert 0 in current.assignments
                assert current.assignments[0] == "a"

                # 3. Test making additional assignments
                await pilot.press("b")
                await pilot.press("2")
                await pilot.pause()
                assert len(current.assignments) > 1

                # 4. Test status updates reactively
                assert isinstance(app.status, tuple)
                assert len(app.status) == 2

                # 5. Test clearing assignments
                await pilot.press("escape")
                await pilot.pause()
                assert len(current.assignments) == 0
                assert app.current_group == ""

                # 6. Test skip workflow
                first_item = app.queue.current
                await pilot.press("right")
                await pilot.pause()
                assert app.queue.current is not first_item
                assert app.queue.items[-1] is first_item

                # 7. Test submitting incomplete assignment shows warning
                initial_count = app.queue.total_count
                await pilot.press("space")
                await pilot.pause()
                assert app.queue.total_count == initial_count
                status_message = app.status[0].lower()
                assert "incomplete" in status_message

                # 8. Test completing and submitting assignment sends judgement
                current = app.queue.current
                assert current is not None
                num_columns = len(current.display_columns)
                for i in range(num_columns):
                    await pilot.press("a")
                    await pilot.press(str((i % 9) + 1))
                    await pilot.pause()

                initial_judgements, _ = self.backend.get_judgements()
                initial_judgement_count = len(initial_judgements)

                await pilot.press("space")
                await pilot.pause()

                final_judgements, _ = self.backend.get_judgements()
                final_judgement_count = len(final_judgements)
                assert final_judgement_count == initial_judgement_count + 1

                # 9. Test help modal
                initial_stack_size = len(pilot.app.screen_stack)
                await pilot.press("f1")
                await pilot.pause()
                assert len(pilot.app.screen_stack) > initial_stack_size
