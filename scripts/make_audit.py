"""Sinh báo cáo audit từ manifest và CSV EDA, không đọc lại corpus."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import re

EXPECTED = {"train": 1400, "dev": 1123, "test": 800}
MISSING = "[KHÔNG CÓ TRONG DỮ LIỆU]"
LICENSE = "[ĐIỀN THỦ CÔNG sau khi đọc README repo]"


def cell(value: object) -> str:
    """Giữ nội dung nguồn trong ô Markdown, tránh phá bảng."""
    if value is None or value == "":
        return MISSING
    return html.escape(str(value)).replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def index_splits(rows: list, source: Path) -> dict[str, dict]:
    """Kiểm tra đủ ba split, không âm thầm ghi đè dòng trùng."""
    indexed: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{source}: bản ghi phải là object.")
        split = row.get("split")
        if not isinstance(split, str) or split not in EXPECTED or split in indexed:
            raise ValueError(f"{source}: split không hợp lệ hoặc trùng: {split!r}")
        indexed[split] = row
    if set(indexed) != set(EXPECTED):
        raise ValueError(f"{source}: cần đủ train/dev/test.")
    return indexed


def make_audit(
    manifest_path: str | Path = "data/raw/MANIFEST.json",
    eda_path: str | Path = "results/tables/eda.csv",
    output_path: str | Path = "data_audit.md",
) -> Path:
    """Ghi nguồn và chênh lệch; không suy đoán metadata hoặc giấy phép."""
    manifest_path, eda_path, output = Path(manifest_path), Path(eda_path), Path(output_path)
    if output.resolve() in (manifest_path.resolve(), eda_path.resolve()):
        raise ValueError("File đầu ra không được trùng file đầu vào.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError(f"{manifest_path}: thiếu danh sách files.")
    files = index_splits(manifest["files"], manifest_path)
    with eda_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"split", "sentence_count", "token_count"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{eda_path}: thiếu split/sentence_count/token_count.")
        stats = index_splits(list(reader), eda_path)
    for split, row in stats.items():
        for field in ("sentence_count", "token_count"):
            if not isinstance(row[field], str) or not re.fullmatch(r"\d+", row[field]):
                raise ValueError(f"{eda_path}: {split}/{field} phải là số nguyên không âm.")

    lines = [
        "# Audit dữ liệu", "",
        "Báo cáo đối chiếu metadata tải và thống kê EDA với số câu kỳ vọng trong kế hoạch.",
        "", f"Repository theo manifest: {cell(manifest.get('repository'))}",
        "", f"Giấy phép: {LICENSE}", "",
        "## Nguồn từng split", "",
        "| Split | Tên file | URL | Ref | Commit | SHA-256 | Kích thước (byte) | Thời điểm tải |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split in EXPECTED:
        record = files[split]
        values = [split, *(record.get(key) for key in
                  ("filename", "url", "ref", "commit", "sha256", "size_bytes", "downloaded_at"))]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines += ["", "## Đối chiếu số câu", "",
              "Số kỳ vọng từ kế hoạch: train 1.400, dev 1.123, test 800 câu. "
              "Đây là mốc đối chiếu, không phải số đo của phiên bản corpus hiện tại.", "",
              "| Split | Số câu EDA | Số token EDA | Số câu kỳ vọng | Chênh lệch (EDA − kỳ vọng) | Trạng thái |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for split, expected in EXPECTED.items():
        sentences, tokens = int(stats[split]["sentence_count"]), int(stats[split]["token_count"])
        delta = sentences - expected
        status = "KHỚP" if delta == 0 else "**CHÊNH LỆCH**"
        lines.append(f"| {split} | {sentences} | {tokens} | {expected} | {delta:+d} | {status} |")
    lines += ["", "## Phạm vi kiểm chứng", "",
              "Metadata được chép từ MANIFEST.json; số câu/token lấy từ eda.csv. "
              "Script không tải dữ liệu, không sửa split, không tự kiểm chứng giấy phép hay tính lại hash.",
              "", "Hai file đầu vào chưa cung cấp liên kết hash chứng minh EDA được sinh từ đúng "
              "các file trong manifest. Cần đối chiếu nguồn của lần chạy EDA trước khi xác nhận audit.",
              "", "Chênh lệch cần được điều tra; báo cáo không suy đoán nguyên nhân và không "
              "tự đổi split cho khớp kế hoạch.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    """Nhận đường dẫn đầu vào và in vị trí báo cáo đã tạo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/MANIFEST.json"))
    parser.add_argument("--eda", type=Path, default=Path("results/tables/eda.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_audit.md"))
    args = parser.parse_args()
    try:
        print(make_audit(args.manifest, args.eda, args.output))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main()
