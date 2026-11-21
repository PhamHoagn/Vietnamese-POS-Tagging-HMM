"""Offline checks of runtime measurement, train/dev isolation and CSV provenance."""

import csv
import json

import numpy as np
import pytest

from scripts.benchmark_hmm import benchmark, main, representative_indices
from src.hmm_bigram import BigramHMMTagger
from src.hmm_trigram import TrigramHMMTagger


def write_corpus(path, sentences):
    rows = []
    for words, tags in sentences:
        rows.extend(f"{index}\t{word}\t_\t{tag}\t_\t_\t0\troot\t_\t_"
                    for index, (word, tag) in enumerate(zip(words, tags), 1))
        rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


@pytest.fixture
def data_paths(tmp_path):
    train, dev = tmp_path / "train.conllu", tmp_path / "dev.conllu"
    write_corpus(train, [(["a", "b"], ["NOUN", "VERB"]), (["a"], ["NOUN"])])
    write_corpus(dev, [(["dev_only"], ["ADJ"]), (["a", "b", "dev_only"], ["NOUN", "VERB", "ADJ"])])
    return train, dev


def test_length_quantile_selection_is_deterministic_and_unique():
    indices = representative_indices([5, 1, 7, 2, 10, 3], 3)
    assert indices == [1, 5, 4]
    assert len(set(indices)) == 3
    assert representative_indices([3, 1], 100) == [1, 0]
    for lengths, count in (([], 100), ([0], 1), ([1], 0)):
        with pytest.raises(ValueError):
            representative_indices(lengths, count)


def test_benchmark_uses_only_train_for_fit_and_writes_real_timings(data_paths, tmp_path, monkeypatch):
    train, dev = data_paths
    for model_class in (BigramHMMTagger, TrigramHMMTagger):
        original = model_class.fit
        def guarded(self, corpus, _original=original):
            assert "dev_only" not in {word for words, _ in corpus for word in words}
            return _original(self, corpus)
        monkeypatch.setattr(model_class, "fit", guarded)
    output = tmp_path / "bench.csv"
    rows, metadata = benchmark(train, dev, output, model_kind="both", sample_size=100, repeats=1,
                               alphas=(.1,), cutoffs=(1,))
    assert len(rows) == 2
    assert {row["model"] for row in rows} == {"bigram", "trigram"}
    assert all(row["sample_sentences"] == 2 and row["sample_tokens"] == 4 for row in rows)
    assert all(row["decode_seconds_median"] > 0 and row["seconds_per_token"] > 0 for row in rows)
    assert metadata["projected_total_seconds"] == pytest.approx(
        1.5 * sum(row["fit_seconds"] + row["seconds_per_token"] * 4 for row in rows))
    assert len(metadata["train_sha256"]) == len(metadata["dev_sha256"]) == 64
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["within_budget"]
    with output.open(encoding="utf-8", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 2
    assert "token_accuracy" not in exported[0]


def test_cli_default_trigram_grid_has_nine_configs(data_paths, tmp_path, capsys):
    train, dev = data_paths
    output = tmp_path / "bench.csv"
    assert main(["--train", str(train), "--dev", str(dev), "--output", str(output), "--repeats", "1"]) == 0
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9 and {row["model"] for row in rows} == {"trigram"}
    assert "Runtime gate: PASS" in capsys.readouterr().out


def test_rejects_identical_data_invalid_options_and_input_overwrite(data_paths, tmp_path):
    train, dev = data_paths
    with pytest.raises(ValueError, match="separate"):
        benchmark(train, train, tmp_path / "x.csv")
    before = train.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        benchmark(train, dev, train)
    assert train.read_bytes() == before
    for kwargs in ({"repeats": 0}, {"sample_size": 0}, {"safety_factor": .5},
                   {"budget_minutes": np.nan}, {"alphas": (0,)}, {"cutoffs": (-1,)}):
        with pytest.raises(ValueError):
            benchmark(train, dev, tmp_path / "bad.csv", **kwargs)
