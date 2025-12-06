"""Evaluation PostgreSQL mixin for Matchbox server."""

from itertools import chain
from typing import TYPE_CHECKING, Any

from pyarrow import Table

from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.eval import Judgement as CommonJudgement
from matchbox.server.postgresql.utils import evaluation

if TYPE_CHECKING:
    from pyarrow import Table as ArrowTable
else:
    ArrowTable = Any


class MatchboxPostgresEvaluationMixin:
    """Evaluation mixin for the PostgreSQL adapter for Matchbox."""

    def insert_judgement(self, judgement: CommonJudgement) -> None:  # noqa: D102
        # Check that all referenced cluster IDs exist
        ids = list(chain(*judgement.endorsed)) + [judgement.shown]
        self.validate_ids(ids)
        evaluation.insert_judgement(judgement)

    def get_judgements(self, tag: str | None = None) -> tuple[Table, Table]:  # noqa: D102
        return evaluation.get_judgements(tag)

    def sample_for_eval(  # noqa: D102
        self, n: int, path: ModelResolutionPath, user_name: str
    ) -> ArrowTable:
        return evaluation.sample(n=n, resolution_path=path, user_name=user_name)
