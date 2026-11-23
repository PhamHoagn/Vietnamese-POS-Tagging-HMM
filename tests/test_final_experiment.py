"""Exercise the review gate and final output using only the synthetic corpus."""

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from scripts import final_experiment as final


def _selection() -> dict:
    params = {"alpha": 0.1, "cutoff": 1}
    metrics = {"token_accuracy": 0.5, "macro_f1": 0.25}
    return {
        "schema_version": 1,
        "dataset": {"source_commit": final.CORPUS_REF, "train": {"sha256": final.TRAIN_SHA256},
                    "dev": {"sha256": final.DEV_SHA256}},
        "protocol": {"selection_split": "dev"},
        "models": {role: {"kind": kind, "params": dict(params) if kind != "baseline" else {},
                          "metrics": metrics} for role, kind in final.MODEL_KINDS.items()},
    }


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "_source_hashes", lambda: {"src/model.py": "same"})
    monkeypatch.setattr(final, "_code_commit", lambda: "reviewed-commit")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(_selection()), encoding="utf-8")
    candidate_path = tmp_path / "candidate.json"
    lock_path = tmp_path / "lock.json"
    final.prepare(selection_path, candidate_path)
    final.seal(candidate_path, lock_path, "Synthetic approval fixture")
    return candidate_path, lock_path


def test_unsealed_candidate_cannot_open_test(sealed, tmp_path, monkeypatch):
    candidate, _ = sealed

    def forbidden(*args):
        pytest.fail("Unsealed run attempted to access the test file.")

    monkeypatch.setattr(final, "_verified_test", forbidden)
    with pytest.raises(ValueError, match="locked protocol"):
        final.run_locked(candidate, tmp_path, tmp_path / "private", tmp_path / "summary")


def test_sealed_configuration_cannot_be_replaced_or_run_with_code_drift(sealed, monkeypatch):
    _, lock_path = sealed
    initial = lock_path.read_text(encoding="utf-8")
    assert final.read_lock(lock_path)["status"] == "locked"
    tampered = json.loads(initial)
    tampered["candidate"]["models"]["h1_best"]["params"]["alpha"] = 1.0
    lock_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        final.read_lock(lock_path)
    lock_path.write_text(initial, encoding="utf-8")
    monkeypatch.setattr(final, "_source_hashes", lambda: {"src/model.py": "edited"})
    with pytest.raises(ValueError, match="source files changed"):
        final.read_lock(lock_path)


def test_locked_runner_reports_all_roles_and_deduplicates_equivalent_models(sealed, tmp_path, monkeypatch):
    _, lock_path = sealed
    fixture = Path(__file__).parent / "fixtures" / "tiny.conllu"
    rows = fixture.read_text(encoding="utf-8").encode("utf-8").split(b"\n\n")
    (tmp_path / "vi_vtb-ud-train.conllu").write_bytes(b"\n\n".join(rows[:2]) + b"\n")
    (tmp_path / "vi_vtb-ud-dev.conllu").write_bytes(rows[2])
    # The approval and file hash stages are exercised above. Keep this execution offline.
    def read_fixture_lock(path):
        return lock

    lock = final.read_lock(lock_path)
    monkeypatch.setattr(final, "read_lock", read_fixture_lock)
    monkeypatch.setattr(final, "TRAIN_SHA256", final._sha256((tmp_path / "vi_vtb-ud-train.conllu").read_bytes()))
    monkeypatch.setattr(final, "DEV_SHA256", final._sha256((tmp_path / "vi_vtb-ud-dev.conllu").read_bytes()))
    (tmp_path / final.TEST_NAME).write_bytes(rows[2])
    monkeypatch.setattr(final, "_verified_test", lambda path: final._sha256(path.read_bytes()))

    private, summary = tmp_path / "private", tmp_path / "summary"
    with io.TextIOWrapper(io.BytesIO(), encoding="cp1252") as stream, redirect_stdout(stream):
        result = final.run_locked(lock_path, tmp_path, private, summary)
    assert result["unique_model_configurations"] == 3
    assert result["test_tokens"] == 3
    with (summary / "metrics_test.csv").open(encoding="utf-8") as stream:
        assert {row["model"] for row in csv.DictReader(stream)} == set(final.MODEL_KINDS)
    with (summary / "paired_bootstrap_test.csv").open(encoding="utf-8") as stream:
        comparisons = list(csv.DictReader(stream))
        assert [(row["left"], row["right"]) for row in comparisons] == list(final.COMPARISONS)
    assert (private / "predictions_test.csv").is_file()
    with pytest.raises(ValueError, match="output directories"):
        final.run_locked(lock_path, tmp_path, private, summary)
