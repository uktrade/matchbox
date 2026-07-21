"""Query and processing time statistics for DAGs."""

import functools
import os
import time
from collections.abc import Callable
from contextlib import ContextDecorator
from typing import Any, Self

import psutil

from matchbox.common.dtos import StepName
from matchbox.common.logging import logger


def log_mem_usage(name: str) -> float:
    """Log current process memory usage for name and return the value in MiB."""
    usage = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    logger.info(f"Memory in `{name}`: {usage:.1f} MiB", prefix="Stats")
    return usage


class _StatLogger(ContextDecorator):
    """Context manager / decorator that measures a metric's change and records it.

    Generic over any zero-argument callable returning a float, e.g.
    time.perf_counter for elapsed time, or a memory-reading function.

    As decorator: resolves name and stats from the decorated instance at call
    time. As context manager: name and stats must be passed explicitly.
    """

    def __init__(
        self,
        operation: str,
        *,
        metric: str,
        measure: Callable[[], float],
        name: str | None = None,
        stats: "DAGStats | None" = None,
    ) -> None:
        self.operation = operation
        self.metric = metric
        self.measure = measure
        self.name = name
        self.stats = stats
        self._start: float | None = None
        self._instance: Any = None

    def _recreate_cm(self) -> "_StatLogger":
        """Return a fresh instance to avoid mutable state leakage across calls."""
        return _StatLogger(
            self.operation,
            metric=self.metric,
            measure=self.measure,
            name=self.name,
            stats=self.stats,
        )

    def __enter__(self) -> Self:
        self._start = self.measure()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._start is None:
            return
        delta = self.measure() - self._start

        stats = self.stats
        name = self.name

        if stats is not None and name is not None:
            stats.record_metric(name, self.metric, delta, operation=self.operation)

        logger.info(
            f"`{self.operation}` in `{name}`: {self.metric} changed by {delta:.3f}",
            prefix="Stats",
        )

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap func so each call is measured against the bound instance."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            instance = args[0] if args else None
            stat_logger = self._recreate_cm()
            stat_logger._instance = instance
            stat_logger.name = getattr(instance, "name", None)
            stat_logger.stats = getattr(instance, "_stats", None)
            with stat_logger:
                return func(*args, **kwargs)

        return wrapper


class DAGStats:
    """Collects per-step timing and memory statistics for a DAG run."""

    def __init__(self) -> None:
        """Initialise empty metric storage."""
        self._metrics: dict[StepName, dict[str, dict[str | None, float]]] = {}
        self.dag_run_seconds: float | None = None

    def reset(self) -> None:
        """Clear all collected statistics."""
        self._metrics.clear()
        self.dag_run_seconds = None

    def ensure_step(self, name: StepName) -> None:
        """Register a step even if no metric has been recorded for it yet."""
        self._metrics.setdefault(name, {})

    def record_metric(
        self, name: StepName, metric: str, value: float, *, operation: str | None = None
    ) -> None:
        """Record a metric value for a step, optionally scoped to an operation."""
        self._metrics.setdefault(name, {}).setdefault(metric, {})[operation] = value

    def record_mem(self, name: StepName) -> None:
        """Record current process memory usage as a point estimate for name."""
        self.record_metric(name, "mem", log_mem_usage(name))

    @staticmethod
    def time(
        operation: str,
        *,
        name: str | None = None,
        stats: "DAGStats | None" = None,
    ) -> _StatLogger:
        """Return a _StatLogger that times wall-clock duration for operation."""
        return _StatLogger(
            operation, metric="time", measure=time.perf_counter, name=name, stats=stats
        )

    @property
    def timings(self) -> dict[StepName, dict[str, float]]:
        """Per-step operation timings."""
        return {
            name: dict(metrics.get("time", {}))
            for name, metrics in self._metrics.items()
        }

    @property
    def mem(self) -> dict[StepName, float]:
        """Latest point-estimate memory reading per step, in MiB."""
        return {
            name: metrics["mem"][None]
            for name, metrics in self._metrics.items()
            if "mem" in metrics and None in metrics["mem"]
        }

    @property
    def total_run_seconds(self) -> float:
        """Sum of all recorded operation durations, excluding dag_run_seconds."""
        return sum(
            value
            for metrics in self._metrics.values()
            for value in metrics.get("time", {}).values()
        )
