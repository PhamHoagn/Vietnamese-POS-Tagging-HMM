"""Shared prediction and aggregate CSV helpers for DEV and the locked run."""

import csv
from collections.abc import Mapping, Sequence, Set
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .baseline import MostFrequentTagger
from .contracts import UPOS_TAGS, Corpus
from .data_loader import Sentence
from .hmm_bigram import BigramHMMTagger
from .hmm_common import HMMTrainingCache
from .hmm_trigram import TrigramHMMTagger


MODEL_CLASSES = {
    "baseline": MostFrequentTagger,
    "bigram": BigramHMMTagger,
    "trigram": TrigramHMMTagger,
}


def as_corpus(sentences: Sequence[Sentence]) -> Corpus:
    return [(sentence.tokens, sentence.tags) for sentence in sentences]


def fit_predict(
    kind: str,
    params: Mapping,
    train_data: Corpus,
    evaluation_data: Corpus,
    cache: HMMTrainingCache | None = None,
) -> tuple[list[list[str]], frozenset[str]]:
    """Fit on train only and pass only token lists to the predictor."""
    if kind not in MODEL_CLASSES:
        raise ValueError(f"Unknown model kind: {kind!r}")
    constructor_options = dict(params)
    if kind != "baseline" and cache is not None:
        constructor_options["cache"] = cache
    model = MODEL_CLASSES[kind](**constructor_options)
    model.fit(train_data)
    predictions = [model.predict(list(tokens)) for tokens, _ in evaluation_data]
    return predictions, model.train_vocab_


def write_csv(path: Path, fields: Sequence[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    sentences: Sequence[Sentence],
    predictions: Mapping[str, list[list[str]]],
    train_vocab: Set[str],
) -> None:
    """Write an aligned token-level file. Keep this file in ignored storage."""
    columns = tuple(predictions)
    if set(columns) & {"sent_id", "idx", "token", "gold", "is_oov"}:
        raise ValueError("Prediction names must differ from metadata column names.")
    for model, sequences in predictions.items():
        if len(sequences) != len(sentences):
            raise ValueError(f"{model}: prediction sentence count differs from gold.")
        for sentence, tags in zip(sentences, sequences, strict=True):
            if len(sentence.tokens) != len(tags) or any(tag not in UPOS_TAGS for tag in tags):
                raise ValueError(f"{model}: predictions have an invalid length or UPOS tag.")

    def records():
        for sentence_index, sentence in enumerate(sentences):
            for index, (token, gold) in enumerate(zip(sentence.tokens, sentence.tags, strict=True)):
                yield {
                    "sent_id": sentence.sent_id,
                    "idx": index,
                    "token": token,
                    "gold": gold,
                    "is_oov": token not in train_vocab,
                    **{column: predictions[column][sentence_index][index] for column in columns},
                }

    write_csv(path, ("sent_id", "idx", "token", "gold", *columns, "is_oov"), records())


def write_label_tables(
    output_dir: Path,
    name: str,
    evaluation_data: Corpus,
    predictions: list[list[str]],
    labels: Sequence[str] = UPOS_TAGS,
) -> None:
    """Write per-UPOS scores and a matrix with gold rows and predicted columns."""
    gold_flat = []
    predicted_flat = []
    for (words, gold), tags in zip(evaluation_data, predictions, strict=True):
        if len(words) != len(gold) or len(gold) != len(tags):
            raise ValueError("Predictions must align with all gold tokens.")
        gold_flat.extend(gold)
        predicted_flat.extend(tags)
    if not gold_flat:
        raise ValueError("No tokens to evaluate.")
    if not labels or len(set(labels)) != len(labels) or not (set(gold_flat) | set(predicted_flat)) <= set(labels):
        raise ValueError("Label table requires unique labels covering every gold and prediction.")
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_flat, predicted_flat, labels=labels, zero_division=0,
    )
    write_csv(
        output_dir / f"per_tag_{name}.csv",
        ("tag", "precision", "recall", "f1", "support"),
        (dict(tag=tag, precision=p, recall=r, f1=f, support=int(n))
         for tag, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True)),
    )
    matrix = confusion_matrix(gold_flat, predicted_flat, labels=labels)
    write_csv(
        output_dir / f"confusion_{name}.csv",
        ("gold", *labels),
        ({"gold": gold, **dict(zip(labels, row, strict=True))}
         for gold, row in zip(labels, matrix, strict=True)),
    )
