"""Deduplication and linking methodologies."""

from matchbox.client.models.comparison import comparison
from matchbox.client.models.models import Model, add_model_class

__all__ = (
    "comparison",
    "Model",
    "add_model_class",
)
