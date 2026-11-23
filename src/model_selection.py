
"""Grid search for POS taggers using train/dev data only."""

from collections.abc import Callable, Mapping, Sequence
from itertools import product
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


def grid_search_pos(
    model_factory: Callable[..., POSTagger],
    train_data: Corpus,
    dev_data: Corpus,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    labels: Sequence[str] = UPOS_TAGS,
    scoring: str = "token_accuracy",
) -> list[dict[str, Any]]:
    """Search hyperparameters on Dev; never accepts or reads Test data."""

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

    # Higher score is better. Resolve HMM ties by alpha, then cutoff.
    def sort_key(row):
        alpha = row.get("param_alpha", 0.0)
        cutoff = row.get("param_cutoff", 0)

        if not isinstance(alpha, Real) or not isinstance(cutoff, Real):
            raise ValueError("alpha and cutoff must be numeric for tie-breaking.")

        return (
            -row["dev_score"],
            float(alpha),
            int(cutoff),
        )

    rows.sort(key=sort_key)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1

    return rows
