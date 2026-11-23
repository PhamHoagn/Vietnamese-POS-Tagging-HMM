"""Check the TV1 audit gate offline against a locked synthetic run."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_final_tv1 as audit
from scripts import final_experiment as final


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    rows = (Path(__file__).parent / "fixtures" / "tiny.conllu").read_text(encoding="utf-8").split("\n\n")
    train, dev, test = ("\n\n".join(rows[:2]).encode(), rows[2].encode(), rows[2].encode())
    hashes = {"train": hashlib.sha256(train).hexdigest(), "dev": hashlib.sha256(dev).hexdigest()}
    for split, content in (("train", train), ("dev", dev), ("test", test)):
        (tmp_path / f"vi_vtb-ud-{split}.conllu").write_bytes(content)
    header = f"blob {len(test)}\0".encode("ascii")
    monkeypatch.setattr(final, "TEST_GIT_BLOB", hashlib.sha1(header + test).hexdigest())
    monkeypatch.setattr(final, "TEST_SIZE", len(test))
    monkeypatch.setattr(final, "TRAIN_SHA256", hashes["train"])
    monkeypatch.setattr(final, "DEV_SHA256", hashes["dev"])
    for split, digest in hashes.items():
        monkeypatch.setitem(audit.CHECKSUMS, split, digest)
    monkeypatch.setattr(final, "_code_commit", lambda: "toy-commit")
    selection = {
        "schema_version": 1,
        "dataset": {"source_commit": final.CORPUS_REF,
                    **{split: {"sha256": digest} for split, digest in hashes.items()}},
        "protocol": {"selection_split": "dev"},
        "models": {role: {"kind": kind, "params": {} if kind == "baseline" else final.FIXED,
                          "metrics": {"token_accuracy": 0.5, "macro_f1": 0.25}}
                   for role, kind in final.MODEL_KINDS.items()},
    }
    selection_path, candidate, lock = (tmp_path / name for name in ("selection.json", "candidate.json", "lock.json"))
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    final.prepare(selection_path, candidate)
    final.seal(candidate, lock, "Synthetic fixture")
    private, summary = tmp_path / "private", tmp_path / "summary"
    final.run_locked(lock, tmp_path, private, summary)
    return lock, summary / "run_metadata.json", private / "predictions_test.csv"


def test_final_audit_exports_verified_aggregates(completed_run, tmp_path):
    lock, metadata, predictions = completed_run
    tables, figures = tmp_path / "tables", tmp_path / "figures"
    manifest = audit.audit_final(lock, metadata, tmp_path, predictions, tables, figures)
    assert manifest["test"]["word_rows"] == 3
    assert manifest["duplicate_pairs"] == {"dev/test": 1}
    assert len(list(figures.glob("*.png"))) == 3
    with (tables / "duplicates.csv").open(encoding="utf-8", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["sent_id_b"] == "toy-dev-1"
    assert "tokens_json" not in row
    with (tables / "test_error_pairs.csv").open(encoding="utf-8", newline="") as stream:
        assert "token" not in (csv.DictReader(stream).fieldnames or [])
    assert sorted(path.name for path in tables.iterdir()) == [
        "audit.json", "duplicates.csv", "eda.csv", "oov_stats.csv", "sentence_lengths.csv",
        "test_error_pairs.csv", "upos_distribution.csv",
    ]


@pytest.mark.parametrize(("field", "message"), [
    ("lock_sha256", "khớp lock"), ("test_sha256", "Test lệch"), ("prediction_sha256", "Dự đoán test lệch"),
])
def test_final_audit_rejects_changed_run_identity(completed_run, tmp_path, monkeypatch, field, message):
    lock, metadata, predictions = completed_run
    record = json.loads(metadata.read_text(encoding="utf-8"))
    record[field] = "0" * 64
    metadata.write_text(json.dumps(record), encoding="utf-8")
    if field in {"lock_sha256", "test_sha256"}:
        parser = audit.read_conllu

        def forbid_test_parse(path):
            assert path.name != final.TEST_NAME
            return parser(path)

        monkeypatch.setattr(audit, "read_conllu", forbid_test_parse)
    tables = tmp_path / "tables"
    with pytest.raises(ValueError, match=message):
        audit.audit_final(lock, metadata, tmp_path, predictions, tables, tmp_path / "figures")
    assert not tables.exists()


def test_unlocked_final_audit_fails_before_opening_metadata(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="locked protocol"):
        audit.audit_final(candidate, missing, missing, missing, missing, missing)
