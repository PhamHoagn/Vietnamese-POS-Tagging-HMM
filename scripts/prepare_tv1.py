"""Kiểm SHA-256 và xuất EDA của hai split train/dev UD Vietnamese-VTB."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unicodedata
from urllib.request import urlopen

from scripts.download_data import RAW_ROOT, REPOSITORY, sha256
from scripts.make_figures import make_figures
from src.data_loader import Sentence, read_conllu
from src.eda import audit_splits, export_eda, export_sparsity
from src.error_script import export_error_analysis

REF = "fa7130860b65db9db342542c51dc57badd30fa9e"
SPLITS = ("train", "dev")
CHECKSUMS = {
    "train": "1df180722db99b724f7368aca50597133bee28606de3394cd8d12a57da89978d",
    "dev": "0b779257b9225575b3b30e15d64cc4ef9945a77f8cceb9f92d794b483a2752dd",
}
DEFAULT_DATA = Path("data/raw/ud-vtb-fa7130860b65")
DEFAULT_TABLES = Path("results/tables/tv1")
DEFAULT_FIGURES = Path("results/figures/tv1")


def pinned_split(data_dir: Path, split: str, *, download: bool = False) -> tuple[Path, str]:
    """Chỉ lấy train hoặc dev, đối chiếu byte với checksum đã ghim."""
    expected = CHECKSUMS[split]
    name = f"vi_vtb-ud-{split}.conllu"
    path = data_dir / name
    url = f"{RAW_ROOT}/{REF}/{name}"
    if download and not path.exists():
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"SHA-256 không khớp nguồn đã ghim: {url}")
        data_dir.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 không khớp nguồn đã ghim: {path}")
    return path, url


def source_counts(path: Path, sentences: list[Sentence]) -> dict[str, int]:
    """Đối chiếu số hàng token nguyên với dữ liệu đã parse, gồm biên câu EOF."""
    counts: Counter[str] = Counter()
    has_metadata = has_token = False
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                counts["metadata_only_blocks"] += has_metadata and not has_token
                has_metadata = has_token = False
            elif stripped.startswith("#"):
                has_metadata = True
            else:
                identifier = stripped.split("\t", 1)[0]
                if "-" in identifier:
                    counts["multiword_rows_skipped"] += 1
                elif "." in identifier:
                    counts["empty_node_rows_skipped"] += 1
                else:
                    has_token = True
                    counts["word_rows"] += 1
    counts["metadata_only_blocks"] += has_metadata and not has_token
    if counts["word_rows"] != sum(len(sentence.tokens) for sentence in sentences):
        raise ValueError(f"Số hàng token và parser không khớp: {path.name}")
    ids = Counter(sentence.sent_id for sentence in sentences if sentence.sent_id is not None)
    return {
        **{field: counts[field] for field in
           ("word_rows", "multiword_rows_skipped", "empty_node_rows_skipped", "metadata_only_blocks")},
        "missing_sentence_ids": sum(sentence.sent_id is None for sentence in sentences),
        "repeated_sentence_ids": sum(count - 1 for count in ids.values()),
        "non_nfc_tokens": sum(
            not unicodedata.is_normalized("NFC", token)
            for sentence in sentences for token in sentence.tokens
        ),
    }


def check_predictions(path: Path, sentences: list[Sentence], *, split: str) -> None:
    """Xác nhận tệp dự đoán dùng đúng thứ tự câu/token và gold đã ghim."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for sentence in sentences:
            for index, (token, tag) in enumerate(zip(sentence.tokens, sentence.tags, strict=True)):
                row = next(reader, None)
                if row is None or (row.get("sent_id"), row.get("idx"), row.get("token"), row.get("gold")) != (
                    sentence.sent_id or "", str(index), token, tag,
                ):
                    raise ValueError(f"Dự đoán {split} lệch dữ liệu tại {sentence.sent_id}/{index}.")
        if next(reader, None) is not None:
            raise ValueError(f"Dự đoán {split} có hàng thừa.")


def duplicate_pairs(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as stream:
        return dict(Counter(
            f"{row['split_a']}/{row['split_b']}" for row in csv.DictReader(stream)
            if row["issue"] in ("duplicate_within", "duplicate_cross")
        ))


def prepare(
    data_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    *,
    download: bool = False,
    predictions: Path | None = None,
) -> None:
    splits: dict[str, list[Sentence]] = {}
    manifest: dict[str, object] = {
        "repository": REPOSITORY,
        "commit": REF,
        "license": "CC BY-SA 4.0",
        "license_source": f"{REPOSITORY}/blob/{REF}/LICENSE.txt",
        "token_policy": "FORM nguyên gốc; phân biệt hoa thường; giữ dấu cách; không NFC",
        "scope": list(SPLITS),
        "unopened_test_reference": {
            "filename": "vi_vtb-ud-test.conllu",
            "source_tree": f"{REPOSITORY}/tree/{REF}",
            "git_blob_sha1": "29cfccf7bc782829f77d039db5d906157a747ae6",
            "bytes": 538633,
        },
    }
    for split in SPLITS:
        path, url = pinned_split(data_dir, split, download=download)
        sentences = read_conllu(path)
        manifest[split] = {
            "filename": path.name,
            "source": url,
            "sha256": CHECKSUMS[split],
            "bytes": path.stat().st_size,
            "sentences": len(sentences),
            **source_counts(path, sentences),
        }
        splits[split] = sentences

    if predictions is not None:
        check_predictions(predictions, splits["dev"], split="dev")

    export_eda(splits, tables_dir, split_order=SPLITS)
    duplicate_path = audit_splits(splits, tables_dir, split_order=SPLITS, include_tokens=False)
    manifest["duplicate_pairs"] = duplicate_pairs(duplicate_path)
    export_sparsity(splits["train"], splits["dev"], tables_dir / "sparsity.csv")
    make_figures(tables_dir, figures_dir, splits=SPLITS)
    if predictions is not None:
        vocabulary = {token for sentence in splits["train"] for token in sentence.tokens}
        with TemporaryDirectory(prefix="tv1-dev-") as temp:
            paths = export_error_analysis(predictions, vocabulary, temp)
            pairs = tables_dir / "dev_error_pairs.csv"
            copyfile(paths["pairs"], pairs)
        (tables_dir / "dev_error_provenance.json").write_text(
            json.dumps({"dev_predictions_sha256": sha256(predictions),
                        "error_pairs_sha256": sha256(pairs)}, indent=2) + "\n", encoding="utf-8",
        )
    (tables_dir / "audit.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--download", action="store_true", help="Tải hai file đã ghim nếu chưa có.")
    parser.add_argument("--dev-predictions", type=Path, help="CSV dev gộp dự đoán B0/H1/H2.")
    args = parser.parse_args()
    try:
        prepare(args.data_dir, args.tables_dir, args.figures_dir,
                download=args.download, predictions=args.dev_predictions)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")
    print(f"Bảng train/dev: {args.tables_dir}\nHình train/dev: {args.figures_dir}")


if __name__ == "__main__":
    main()
