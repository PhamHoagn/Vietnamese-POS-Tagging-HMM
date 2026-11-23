"""Phân tích lỗi từ CSV dự đoán gộp B0/H1/H2, không fit hay chỉnh mô hình."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Collection, Sequence
import csv
from pathlib import Path

from .contracts import UPOS_TAGS
from .data_loader import read_conllu

BASE_FIELDS = ("sent_id", "idx", "token", "gold", "pred_b0", "pred_h1")
OPTIONAL_H2 = "pred_h2"
OPTIONAL_OOV = "is_oov"


def _parse_bool(value: str, row_number: int) -> bool:
    """Đọc OOV trong CSV, không ngầm coi chuỗi lạ là False."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Dòng {row_number}: is_oov phải là True hoặc False.")


def _sentence_keys(rows: Sequence[dict[str, object]]) -> list[int]:
    """Chia câu theo sent_id hoặc khi idx quay về; hỗ trợ sent_id rỗng."""
    keys: list[int] = []
    sentence_key = -1
    previous_id: str | None = None
    previous_idx: int | None = None
    for row in rows:
        sent_id, index = str(row["sent_id"]), int(row["idx"])
        if previous_id != sent_id or previous_idx is None or index <= previous_idx:
            sentence_key += 1
        keys.append(sentence_key)
        previous_id, previous_idx = sent_id, index
    return keys


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    """Ghi CSV UTF-8 có header cả khi không có lỗi."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_error_analysis(
    predictions_path: str | Path,
    vocab_train: Collection[str] | None = None,
    output_dir: str | Path = "results/tables/errors",
    *,
    split: str = "dev",
    example_limit: int = 10,
    quiet: bool = False,
) -> dict[str, Path]:
    """Xuất lỗi, cặp nhầm và 5–10 ví dụ xác định từ CSV dự đoán gộp.

    OOV lấy từ `is_oov` trong CSV hoặc `vocab_train` nguyên gốc của train.
    Nếu cả hai đều có, dữ liệu CSV được kiểm tra khớp vocabulary để phát hiện
    nhầm nguồn. `split` là metadata do người gọi cung cấp, không dùng để tune.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError("split phải là train/dev/test.")
    if not 5 <= example_limit <= 10:
        raise ValueError("example_limit phải nằm trong khoảng 5–10.")
    if vocab_train is not None and any(not isinstance(token, str) for token in vocab_train):
        raise TypeError("vocab_train phải chứa chuỗi FORM nguyên gốc.")

    source = Path(predictions_path)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not set(BASE_FIELDS).issubset(fields):
            missing = sorted(set(BASE_FIELDS) - fields)
            raise ValueError(f"CSV thiếu cột bắt buộc: {missing}.")
        if OPTIONAL_OOV not in fields and vocab_train is None:
            raise ValueError("Cần cột is_oov hoặc vocab_train nguyên gốc của train.")
        has_h2 = OPTIONAL_H2 in fields
        raw_rows = list(reader)

    records: list[dict[str, object]] = []
    for row_number, row in enumerate(raw_rows, start=2):
        if any(row.get(field) is None for field in BASE_FIELDS):
            raise ValueError(f"Dòng {row_number}: thiếu giá trị bắt buộc.")
        if not row["idx"].isdigit() or not row["token"]:
            raise ValueError(f"Dòng {row_number}: idx/token không hợp lệ.")
        if row["gold"] not in UPOS_TAGS:
            raise ValueError(f"Dòng {row_number}: gold ngoài 17 UPOS.")
        predictions = {"B0": row["pred_b0"], "H1": row["pred_h1"]}
        if has_h2:
            predictions["H2"] = row[OPTIONAL_H2]
        if any(pred not in UPOS_TAGS for pred in predictions.values()):
            raise ValueError(f"Dòng {row_number}: pred ngoài 17 UPOS.")
        from_csv = _parse_bool(row[OPTIONAL_OOV], row_number) if OPTIONAL_OOV in fields else None
        from_vocab = row["token"] not in vocab_train if vocab_train is not None else None
        if from_csv is not None and from_vocab is not None and from_csv != from_vocab:
            raise ValueError(f"Dòng {row_number}: is_oov không khớp vocab_train.")
        records.append({
            "sent_id": row["sent_id"], "idx": int(row["idx"]), "token": row["token"],
            "gold": row["gold"], "predictions": predictions,
            "is_oov": from_vocab if from_vocab is not None else from_csv,
            "source_row": row_number,
        })
    for record, key in zip(records, _sentence_keys(records)):
        record["sentence_key"] = key

    errors: list[dict[str, object]] = []
    pair_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    pair_totals: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        oov_group = "OOV" if record["is_oov"] else "IV"
        punct_group = "PUNCT" if record["gold"] == "PUNCT" else "NON_PUNCT"
        for model, prediction in record["predictions"].items():
            if prediction != record["gold"]:
                pair_key = (model, oov_group, punct_group, record["gold"], prediction)
                pair_counts[pair_key] += 1
                pair_totals[pair_key[:3]] += 1
                errors.append({
                    "split": split, "sent_id": record["sent_id"], "idx": record["idx"],
                    "token": record["token"], "model": model, "gold": record["gold"],
                    "pred": prediction, "is_oov": record["is_oov"],
                    "punctuation_group": punct_group, "source_row": record["source_row"],
                })

    pair_rows: list[dict[str, object]] = []
    for key, count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        model, oov_group, punct_group, gold, pred = key
        pair_rows.append({
            "split": split, "model": model, "oov_group": oov_group,
            "punctuation_group": punct_group, "gold": gold, "pred": pred,
            "count": count, "percentage_within_group": 100.0 * count / pair_totals[key[:3]],
        })

    sentence_rows: dict[int, list[dict[str, object]]] = {}
    for record in records:
        sentence_rows.setdefault(int(record["sentence_key"]), []).append(record)
    selected: list[tuple[dict[str, object], str]] = []
    selected_positions: set[tuple[int, int]] = set()
    for (model, oov_group, punct_group, gold, pred), _ in sorted(
        pair_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        for record in records:
            position = (int(record["sentence_key"]), int(record["idx"]))
            if (position not in selected_positions
                    and record["gold"] == gold
                    and record["predictions"][model] == pred
                    and ("OOV" if record["is_oov"] else "IV") == oov_group
                    and ("PUNCT" if record["gold"] == "PUNCT" else "NON_PUNCT") == punct_group):
                selected.append((record, f"{model}: {gold}→{pred}; {oov_group}; {punct_group}"))
                selected_positions.add(position)
                break
        if len(selected) == example_limit:
            break

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "errors": output / f"errors_{split}.csv",
        "pairs": output / f"top_error_pairs_{split}.csv",
        "examples": output / f"error_examples_{split}.md",
    }
    _write_csv(paths["errors"], errors, (
        "split", "sent_id", "idx", "token", "model", "gold", "pred", "is_oov",
        "punctuation_group", "source_row",
    ))
    _write_csv(paths["pairs"], pair_rows, (
        "split", "model", "oov_group", "punctuation_group", "gold", "pred", "count",
        "percentage_within_group",
    ))
    lines = [
        "| Câu/token | Gold | B0 | H1 | H2 | OOV? | Nhóm lỗi/giải thích |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record, group in selected:
        context = sentence_rows[int(record["sentence_key"])]
        focus = next(index for index, item in enumerate(context) if item["idx"] == record["idx"])
        window = context[max(0, focus - 3): focus + 4]
        context_text = " ".join(
            f"[{item['token']}]" if item["idx"] == record["idx"] else str(item["token"])
            for item in window
        )
        predictions = record["predictions"]
        sentence_token = f"{record['sent_id'] or '[không sent_id]'} / {record['idx']}: {context_text}"
        lines.append(
            f"| {sentence_token} | {record['gold']} | {predictions['B0']} | "
            f"{predictions['H1']} | {predictions.get('H2', 'N/A')} | "
            f"{'Có' if record['is_oov'] else 'Không'} | {group}; [điền sau khi xem] |"
        )
    paths["examples"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not quiet:
        print(f"{split}: {len(records)} token; {len(errors)} lỗi; {len(selected)} ví dụ xác định.")
    return paths


def main() -> None:
    """Đọc CSV gộp; `--train-conllu` xác định OOV nếu CSV thiếu cột is_oov."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--train-conllu", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables/errors"))
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--example-limit", type=int, default=10)
    args = parser.parse_args()
    vocab = None
    if args.train_conllu is not None:
        vocab = {token for sentence in read_conllu(args.train_conllu) for token in sentence.tokens}
    try:
        for path in export_error_analysis(
            args.predictions, vocab, args.output_dir, split=args.split,
            example_limit=args.example_limit,
        ).values():
            print(path)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main()
