
"""Grid search for POS taggers using train/dev data only."""

from collections.abc import Callable, Mapping, Sequence
from itertools import groupby, product
from math import isfinite
from numbers import Real
from typing import Any

from .contracts import Corpus, POSTagger, UPOS_TAGS
from .metrics import evaluate_pos


SUPPORTED_SCORING = (
    "token_accuracy",
    "macro_f1",
    "oov_accuracy",
    "accuracy_no_punct",
)
MACRO_F1_TOLERANCE = 1e-12


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by score, Macro-F1, then lower cutoff and higher alpha.

    Anchor each tolerance group to its best Macro-F1, so ranking is
    deterministic even if adjacent Macro-F1 differences overlap.
    """
    def parameter_key(row):
        alpha = row.get("param_alpha", 0.0)
        cutoff = row.get("param_cutoff", 0)
        if not isinstance(alpha, Real) or not isinstance(cutoff, Real):
            raise ValueError("alpha and cutoff must be numeric for tie-breaking.")
        return cutoff, -alpha

    by_score = sorted(rows, key=lambda row: row["dev_score"], reverse=True)
    ranked = []
    for _, score_rows in groupby(by_score, key=lambda row: row["dev_score"]):
        macro_rows = sorted(score_rows, key=lambda row: row["dev_macro_f1"], reverse=True)
        group = [macro_rows[0]]
        for row in macro_rows[1:]:
            if group[0]["dev_macro_f1"] - row["dev_macro_f1"] > MACRO_F1_TOLERANCE:
                ranked.extend(sorted(group, key=parameter_key))
                group = []
            group.append(row)
        ranked.extend(sorted(group, key=parameter_key))
    return ranked


def grid_search_pos(
    model_factory: Callable[..., POSTagger],
    train_data: Corpus,
    dev_data: Corpus,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    labels: Sequence[str] = UPOS_TAGS,
    scoring: str = "token_accuracy",
) -> list[dict[str, Any]]:
    """Search using only train and dev, with gold-support Macro-F1 as secondary."""

    if not callable(model_factory):
        raise ValueError("model_factory must be callable.")

    if not isinstance(param_grid, Mapping) or not param_grid:
        raise ValueError("param_grid must be a non-empty mapping.")

    if scoring not in SUPPORTED_SCORING:
        raise ValueError(
            f"Unsupported scoring '{scoring}'. "
            f"Choose one of {SUPPORTED_SCORING}."
        )

    names = list(param_grid.keys())
    values = []

    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter names must be non-empty strings.")

        candidates = param_grid[name]
        if (
            isinstance(candidates, (str, bytes))
            or not isinstance(candidates, Sequence)
            or len(candidates) == 0
        ):
            raise ValueError(
                f"Parameter '{name}' must have a non-empty sequence of values."
            )

        values.append(list(candidates))

    rows = []

    for combination in product(*values):
        params = dict(zip(names, combination))

        model = model_factory(**params)
        fitted_model = model.fit(train_data)

        if fitted_model is not model:
            raise ValueError("fit() must return the same model instance.")

        if not hasattr(model, "train_vocab_"):
            raise ValueError(
                "The fitted model must expose train_vocab_ for OOV evaluation."
            )

        predictions = [
            model.predict(tokens)
            for tokens, _ in dev_data
        ]

        metrics = evaluate_pos(
            gold_data=dev_data,
            predictions=predictions,
            train_vocab=model.train_vocab_,
            labels=labels,
            macro_f1_gold_support=True,
        )

        score = metrics[scoring]

        if not isinstance(score, Real) or not isfinite(score):
            raise ValueError(f"Invalid Dev score for {scoring}: {score}")

        row = {
            f"param_{name}": value
            for name, value in params.items()
        }

        row.update({
            "dev_token_accuracy": metrics["token_accuracy"],
            "dev_macro_f1": metrics["macro_f1"],
            "dev_oov_accuracy": metrics["oov_accuracy"],
            "dev_oov_rate": metrics["oov_rate"],
            "dev_accuracy_no_punct": metrics["accuracy_no_punct"],
            "dev_score": float(score),
        })

        rows.append(row)

    rows = _rank_rows(rows)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1

    return rows
