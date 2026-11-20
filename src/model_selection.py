"""Grid-search contract only; no training or parameter search is implemented."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contracts import Corpus, POSTagger, UPOS_TAGS


def grid_search_pos(
    model_factory: Callable[..., POSTagger],
    train_data: Corpus,
    dev_data: Corpus,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    labels: Sequence[str] = UPOS_TAGS,
    scoring: str = "token_accuracy",
) -> list[dict[str, Any]]:
    """Future CSV-ready rows: param_*, dev_*, rank, selected. Never accepts test."""
    raise NotImplementedError("Train/dev grid search is not implemented (TV3).")
