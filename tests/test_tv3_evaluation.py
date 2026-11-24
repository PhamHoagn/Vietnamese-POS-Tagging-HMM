import csv
from collections.abc import Mapping
from typing import get_type_hints

import numpy as np
import pytest

from scripts.evaluate_dev import _save_grid, expanded_grid
from src.bootstrap import paired_accuracy_intervals
from src.data_loader import Sentence
from src.evaluation_outputs import as_corpus, write_label_tables, write_predictions


def test_grid_writer_annotations_resolve_even_when_python_defers_them():
    assert get_type_hints(_save_grid)["rows"] == Mapping[str, list[dict]]


def test_grid_expansion_is_joint_at_all_triggered_edges():
    winners = [
        {"param_alpha": 0.01, "param_cutoff": 0},
        {"param_alpha": 1.0, "param_cutoff": 2},
    ]
    grid, edges = expanded_grid(winners)
    assert grid == {"alpha": [0.001, 0.01, 0.1, 1.0, 3.0], "cutoff": [0, 1, 2, 3]}
    assert set(edges) == {"alpha_low", "alpha_high", "cutoff_high"}
    assert expanded_grid([{"param_alpha": 0.1, "param_cutoff": 0}]) == (None, [])


def test_paired_bootstrap_resamples_identical_sentences_and_weights_tokens():
    gold = [(["a"], ["NOUN"]), (["b", "c", "d"], ["NOUN"] * 3)]
    predictions = {
        "left": [["NOUN"], ["NOUN", "NOUN", "NOUN"]],
        "right": [["NOUN"], ["NOUN", "VERB", "VERB"]],
    }
    seed, resamples = 42, 200
    first, reverse = paired_accuracy_intervals(
        gold, predictions, comparisons=[("left", "right"), ("right", "left")],
        seed=seed, resamples=resamples,
    )
    # Independent hand-derived replicate: one sample of sentence 2 contributes
    # two extra correct tokens in a three-token sentence, sentence 1 adds none.
    rng = np.random.default_rng(seed)
    samples = rng.integers(2, size=(resamples, 2))
    count_sentence_2 = samples.sum(axis=1)
    expected = 2 * count_sentence_2 / (2 + 2 * count_sentence_2)
    assert first["delta_token_accuracy"] == 0.5
    assert reverse["delta_token_accuracy"] == -0.5
    assert first["ci95_lower"] == np.quantile(expected, 0.025)
    assert first["ci95_upper"] == np.quantile(expected, 0.975)
    assert reverse["ci95_lower"] == pytest.approx(-first["ci95_upper"])
    assert reverse["ci95_upper"] == pytest.approx(-first["ci95_lower"])


def test_paired_bootstrap_rejects_missing_or_misaligned_predictions():
    gold = [(["a"], ["NOUN"])]
    with pytest.raises(ValueError, match="Missing prediction series"):
        paired_accuracy_intervals(gold, {"b0": [["NOUN"]]})
    with pytest.raises(ValueError, match="invalid length"):
        paired_accuracy_intervals(
            gold, {"h1_best": [[]], "b0": [["NOUN"]]}, comparisons=[("h1_best", "b0")],
        )


def test_outputs_preserve_alignment_oov_and_aggregate_orientation(tmp_path):
    sentences = [Sentence("dev-1", ["Hà Nội", "."], ["PROPN", "PUNCT"])]
    predictions = [["NOUN", "PUNCT"]]
    path = tmp_path / "pred.csv"
    write_predictions(path, sentences, {"pred_b0": predictions}, {"."})
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["sent_id"], row["idx"], row["token"], row["gold"], row["pred_b0"], row["is_oov"])
            for row in rows] == [
                ("dev-1", "0", "Hà Nội", "PROPN", "NOUN", "True"),
                ("dev-1", "1", ".", "PUNCT", "PUNCT", "False"),
            ]

    write_label_tables(tmp_path, "b0", as_corpus(sentences), predictions)
    with (tmp_path / "confusion_b0.csv").open(newline="") as stream:
        matrix = {row["gold"]: row for row in csv.DictReader(stream)}
    assert matrix["PROPN"]["NOUN"] == "1"
    assert sum(int(row[label]) for label, row in matrix.items()) == 1
    with (tmp_path / "per_tag_b0.csv").open(newline="") as stream:
        scores = {row["tag"]: row for row in csv.DictReader(stream)}
    assert scores["PROPN"]["support"] == "1"
    assert scores["PROPN"]["f1"] == "0.0"
