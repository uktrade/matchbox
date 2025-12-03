"""Evaluation PostgreSQL mixin for Matchbox server."""

from itertools import chain
from typing import TYPE_CHECKING, Any

import polars as pl
from pyarrow import Table

from matchbox.common.dtos import ModelResolutionPath
from matchbox.common.eval import Judgement as CommonJudgement
from matchbox.common.eval import ModelComparison
from matchbox.common.exceptions import MatchboxNoJudgements
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

    def get_judgements(self) -> tuple[Table, Table]:  # noqa: D102
        return evaluation.get_judgements()

    def compare_models(self, paths: list[ModelResolutionPath]) -> ModelComparison:  # noqa: D102
        judgements, expansion = self.get_judgements()
        if not len(judgements):
            raise MatchboxNoJudgements()
        return evaluation.compare_models(
            paths, pl.from_arrow(judgements), pl.from_arrow(expansion)
        )

    def sample_for_eval(  # noqa: D102
        self, n: int, path: ModelResolutionPath, user_name: str
    ) -> ArrowTable:
        return evaluation.sample(n=n, resolution_path=path, user_name=user_name)
