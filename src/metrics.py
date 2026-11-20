"""Basic evaluation. Reserved metrics explicitly return None until implemented."""

from collections.abc import Sequence, Set
from typing import TypedDict

from sklearn.metrics import f1_score

from .contracts import Corpus, UPOS_TAGS, validate_corpus


class POSMetrics(TypedDict):
    token_accuracy: float
    macro_f1: float
    oov_accuracy: float | None
    oov_rate: float | None
    accuracy_no_punct: float | None


def evaluate_pos(
    gold_data: Corpus,
    predictions: list[list[str]],
    train_vocab: Set[str],
    labels: Sequence[str],
) -> POSMetrics:
    validate_corpus(gold_data)
    if len(gold_data) != len(predictions):
        raise ValueError("Gold and predictions must contain the same number of sentences.")
    fixed_labels = list(labels)
    if (not fixed_labels or len(fixed_labels) != len(set(fixed_labels))
            or any(label not in UPOS_TAGS for label in fixed_labels)):
        raise ValueError("labels must be a non-empty sequence of unique UPOS labels.")
    if any(not isinstance(word, str) or not word for word in train_vocab):
        raise ValueError("train_vocab must contain non-empty strings from training only.")
    gold_flat: list[str] = []
    pred_flat: list[str] = []
    for index, ((tokens, gold), predicted) in enumerate(zip(gold_data, predictions)):
        if len(tokens) != len(predicted):
            raise ValueError(f"Sentence {index}: token/prediction lengths differ.")
        if any(tag not in UPOS_TAGS for tag in predicted):
            raise ValueError(f"Sentence {index}: predictions must be valid UPOS labels.")
        gold_flat.extend(gold)
        pred_flat.extend(predicted)
    if not (set(gold_flat) | set(pred_flat)).issubset(fixed_labels):
        raise ValueError("labels must include every gold and predicted tag.")
    accuracy = sum(g == p for g, p in zip(gold_flat, pred_flat)) / len(gold_flat) if gold_flat else 0.0
    macro = float(f1_score(gold_flat, pred_flat, labels=fixed_labels,
                          average="macro", zero_division=0)) if gold_flat else 0.0
    return {
        "token_accuracy": accuracy,
        "macro_f1": macro,
        "oov_accuracy": None,
        "oov_rate": None,
        "accuracy_no_punct": None,
    }
