"""Đối chiếu dữ liệu và dự đoán với lần đánh giá test đã khóa, rồi xuất EDA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from scripts.download_data import RAW_ROOT, REPOSITORY, sha256
from scripts.final_experiment import TEST_NAME, read_lock
from scripts.make_figures import make_figures
from scripts.prepare_tv1 import (
    CHECKSUMS, DEFAULT_DATA, DEFAULT_FIGURES, DEFAULT_TABLES, REF, SPLITS,
    check_predictions, duplicate_pairs, pinned_split, source_counts,
)
from src.data_loader import read_conllu
from src.eda import audit_splits, export_eda
from src.error_script import export_error_analysis


def audit_final(
    lock_path: Path,
    metadata_path: Path,
    data_dir: Path,
    predictions: Path,
    tables_dir: Path,
    figures_dir: Path,
) -> dict:
    """Đọc test sau khi đối chiếu lock và metadata của lần đánh giá chính thức."""
    lock = read_lock(lock_path)
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("lock_sha256") != lock["lock_sha256"] or not metadata.get("run_completed_at_utc"):
        raise ValueError("Metadata đánh giá test đã hoàn tất phải khớp lock.")
    test_sha256 = metadata["test_sha256"]
    prediction_sha256 = metadata["prediction_sha256"]
    identity = lock["candidate"]["dataset"]
    split_order = (*SPLITS, "test")
    manifest = {
        "repository": REPOSITORY,
        "commit": REF,
        "scope": list(split_order),
        "lock_sha256": lock["lock_sha256"],
        "run_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "test_predictions_sha256": prediction_sha256,
    }
    splits = {}
    for split in SPLITS:
        path, url = pinned_split(data_dir, split)
        sentences = splits[split] = read_conllu(path)
        manifest[split] = {"filename": path.name, "source": url, "sha256": CHECKSUMS[split],
                           "bytes": path.stat().st_size, "sentences": len(sentences),
                           **source_counts(path, sentences)}

    with TemporaryDirectory(prefix="tv1-test-") as location:
        work = Path(location)
        test_bytes = (data_dir / TEST_NAME).read_bytes()
        git_header = f"blob {len(test_bytes)}\0".encode("ascii")
        if (len(test_bytes) != identity["test_size_bytes"]
                or hashlib.sha1(git_header + test_bytes).hexdigest() != identity["test_git_blob_sha1"]
                or hashlib.sha256(test_bytes).hexdigest() != test_sha256):
            raise ValueError("Test lệch Git blob đã khóa hoặc SHA-256 của lần đánh giá.")
        test = work / TEST_NAME
        test.write_bytes(test_bytes)
        sentences = splits["test"] = read_conllu(test)
        manifest["test"] = {
            "filename": TEST_NAME, "source": f"{RAW_ROOT}/{REF}/{TEST_NAME}",
            "sha256": test_sha256, "git_blob_sha1": identity["test_git_blob_sha1"],
            "bytes": len(test_bytes), "sentences": len(sentences), **source_counts(test, sentences),
        }
        vocabulary = {token for sentence in splits["train"] for token in sentence.tokens}
        expected_counts = {
            "train_sentences": len(splits["train"]), "test_sentences": len(sentences),
            "train_tokens": manifest["train"]["word_rows"], "test_tokens": manifest["test"]["word_rows"],
            "test_oov_tokens": sum(token not in vocabulary for sentence in sentences for token in sentence.tokens),
            "test_non_punct_tokens": sum(tag != "PUNCT" for sentence in sentences for tag in sentence.tags),
        }
        if any(metadata.get(field) != value for field, value in expected_counts.items()):
            raise ValueError("Số câu/token độc lập lệch metadata của lần đánh giá test.")
        prediction_snapshot = work / "predictions_test.csv"
        copyfile(predictions, prediction_snapshot)
        if sha256(prediction_snapshot) != prediction_sha256:
            raise ValueError("Dự đoán test lệch SHA-256 của lần đánh giá.")
        check_predictions(prediction_snapshot, sentences, split="test")
        error_paths = export_error_analysis(prediction_snapshot, vocabulary, work, split="test", quiet=True)

        export_eda(splits, tables_dir)
        duplicate_path = audit_splits(splits, tables_dir, include_tokens=False)
        manifest["duplicate_pairs"] = duplicate_pairs(duplicate_path)
        pairs = tables_dir / "test_error_pairs.csv"
        copyfile(error_paths["pairs"], pairs)
        manifest["test_error_pairs_sha256"] = sha256(pairs)
        make_figures(tables_dir, figures_dir)
        (tables_dir / "audit.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--run-metadata", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES / "final")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES / "final")
    args = parser.parse_args()
    try:
        audit_final(args.lock, args.run_metadata, args.data_dir, args.predictions,
                    args.tables_dir, args.figures_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")
    print(f"Bảng test: {args.tables_dir}\nHình test: {args.figures_dir}")


if __name__ == "__main__":
    main()
