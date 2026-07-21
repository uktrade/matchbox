"""Tests for the DAGStats statistics class."""

import logging
import time

import pytest

from matchbox.common.stats import DAGStats


class _DummyStep:
    """Minimal stand-in for a step node with name and _stats."""

    def __init__(self, name: str, stats: DAGStats) -> None:
        self.name = name
        self._stats = stats

    @DAGStats.time("op")
    def do_something(self) -> int:
        return 42


class _StepWithoutStats:
    """A class without _stats attribute — decorator should no-op."""

    name = "no_stats"

    @DAGStats.time("op")
    def do_something(self) -> int:
        return 99


def test_time_as_decorator(caplog: pytest.LogCaptureFixture) -> None:
    """Decorator mode records timing and logs."""
    stats = DAGStats()
    step = _DummyStep(name="step_a", stats=stats)

    with caplog.at_level(logging.INFO, logger="matchbox"):
        result = step.do_something()

    assert result == 42
    assert "step_a" in stats.timings
    assert "op" in stats.timings["step_a"]
    assert stats.timings["step_a"]["op"] >= 0
    assert any("Stats" in r.message for r in caplog.records)


def test_time_as_context_manager(caplog: pytest.LogCaptureFixture) -> None:
    """Context manager mode records timing and logs."""
    stats = DAGStats()

    with (
        caplog.at_level(logging.INFO, logger="matchbox"),
        DAGStats.time("query", name="source_x", stats=stats),
    ):
        time.sleep(0.001)

    assert "source_x" in stats.timings
    assert "query" in stats.timings["source_x"]
    assert stats.timings["source_x"]["query"] >= 0


def test_record_mem(caplog: pytest.LogCaptureFixture) -> None:
    """record_mem populates mem dict and logs."""
    stats = DAGStats()

    with caplog.at_level(logging.INFO, logger="matchbox"):
        stats.record_mem("step_b")

    assert "step_b" in stats.mem
    assert stats.mem["step_b"] > 0
    assert any("Memory" in r.message for r in caplog.records)


def test_reset() -> None:
    """reset clears all collected statistics."""
    stats = DAGStats()
    stats.record_metric("step_a", "time", 1.0, operation="op")
    stats.record_mem("step_a")
    stats.dag_run_seconds = 5.0

    stats.reset()

    assert stats.timings == {}
    assert stats.mem == {}
    assert stats.dag_run_seconds is None


def test_total_run_seconds() -> None:
    """total_run_seconds sums all operation durations."""
    stats = DAGStats()
    stats.record_metric("step_a", "time", 1.0, operation="hash")
    stats.record_metric("step_a", "time", 2.0, operation="sync")
    stats.record_metric("step_b", "time", 3.5, operation="compute")

    assert stats.total_run_seconds == 6.5


def test_dag_run_seconds_excluded_from_total() -> None:
    """dag_run_seconds is not included in total_run_seconds."""
    stats = DAGStats()
    stats.record_metric("step_a", "time", 1.0, operation="op")
    stats.dag_run_seconds = 100.0

    assert stats.total_run_seconds == 1.0


def test_timer_no_stats_no_crash(caplog: pytest.LogCaptureFixture) -> None:
    """Decorator on a class without _stats does not crash."""
    step = _StepWithoutStats()

    with caplog.at_level(logging.INFO, logger="matchbox"):
        result = step.do_something()

    assert result == 99
