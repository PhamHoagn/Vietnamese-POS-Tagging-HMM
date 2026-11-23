"""Paired bootstrap over sentences for differences in micro token accuracy."""

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

from .contracts import UPOS_TAGS, Corpus, validate_corpus


BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 42
PLANNED_COMPARISONS = (
    ("h1_best", "b0"),
    ("h2_best", "h1_best"),
    ("h2_fixed", "h1_fixed"),
)


def paired_accuracy_intervals(
    gold_data: Corpus,
    predictions: Mapping[str, Sequence[Sequence[str]]],
    *,
    comparisons: Sequence[tuple[str, str]] = PLANNED_COMPARISONS,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict]:
    """Return percentile CI for each left-minus-right difference in accuracy.

    Every draw samples N sentence indices with replacement. The same draw is
    used for all models and comparisons. Each replicate sums correct counts
    and token counts before dividing, so longer sentences keep their weight.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, Integral) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not comparisons:
        raise ValueError("At least one comparison is required.")
    validate_corpus(gold_data)
    lengths = np.array([len(tags) for _, tags in gold_data], dtype=np.int64)
    if lengths.size == 0 or np.any(lengths == 0):
        raise ValueError("Bootstrap requires a corpus of non-empty sentences.")

    names = tuple(dict.fromkeys(name for pair in comparisons for name in pair))
    counts = []
    for name in names:
        if name not in predictions:
            raise ValueError(f"Missing prediction series: {name!r}")
        series = predictions[name]
        if len(series) != len(gold_data):
            raise ValueError(f"{name}: prediction sentence count differs from gold.")
        correct = []
        for (_, gold), predicted in zip(gold_data, series, strict=True):
            if len(predicted) != len(gold) or any(tag not in UPOS_TAGS for tag in predicted):
                raise ValueError(f"{name}: predictions have an invalid length or UPOS tag.")
            correct.append(sum(g == p for g, p in zip(gold, predicted, strict=True)))
        counts.append(correct)

    count_array = np.asarray(counts, dtype=np.int64)
    model_index = {name: index for index, name in enumerate(names)}
    effects = np.stack([
        count_array[model_index[left]] - count_array[model_index[right]]
        for left, right in comparisons
    ])
    observed = effects.sum(axis=1) / lengths.sum()
    draws = np.empty((len(comparisons), resamples), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        indices = rng.integers(lengths.size, size=(stop - start, lengths.size))
        totals = lengths[indices].sum(axis=1)
        draws[:, start:stop] = effects[:, indices].sum(axis=2) / totals
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=1)
    return [
        {
            "left": left,
            "right": right,
            "delta_token_accuracy": float(observed[index]),
            "ci95_lower": float(lower[index]),
            "ci95_upper": float(upper[index]),
            "resamples": int(resamples),
            "seed": int(seed),
            "sentences": int(lengths.size),
            "tokens": int(lengths.sum()),
        }
        for index, (left, right) in enumerate(comparisons)
    ]
