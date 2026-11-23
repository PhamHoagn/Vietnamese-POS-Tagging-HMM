"""Vẽ ba PNG từ CSV EDA, không đọc lại corpus hay tính lại OOV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Giữ thứ tự UPOS của dự án; chạy được cả python scripts/make_figures.py.
UPOS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
SPLITS = ("train", "dev", "test")
COLORS = {"train": "#2878B5", "dev": "#E58B26", "test": "#29946D"}


def read_table(path: Path, fields: set[str]) -> list[dict[str, str]]:
    """Đọc CSV UTF-8 và kiểm tra các cột cần dùng."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: thiếu cột {sorted(fields)}")
        return list(reader)


def rates(rows: list[dict[str, str]], splits: tuple[str, ...],
          field: str, maximum: float) -> dict[tuple[str, str], float]:
    """Kiểm tra đủ nhãn; ô trống là thiếu dữ liệu, không phải 0."""
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        split, tag = row["split"], row["upos"]
        if tag == "ALL" and field == "oov_rate":
            continue
        key = (split, tag)
        if split not in splits or tag not in UPOS or key in values:
            raise ValueError(f"Dòng split/UPOS không hợp lệ hoặc trùng: {key}")
        value = float(row[field]) if row[field] else math.nan
        if row[field] and (not math.isfinite(value) or not 0 <= value <= maximum):
            raise ValueError(f"{field} ngoài miền hợp lệ: {key}")
        values[key] = value
    if set(values) != {(split, tag) for split in splits for tag in UPOS}:
        raise ValueError(f"{field}: cần đủ 17 UPOS cho mỗi split.")
    return values


def make_figures(tables_dir: str | Path = "results/tables",
                 output_dir: str | Path = "results/figures",
                 *, splits: tuple[str, ...] = SPLITS) -> list[Path]:
    """Lưu ba figure riêng ở 200 dpi, chỉ lấy số liệu từ CSV."""
    if splits not in (SPLITS, SPLITS[:2]):
        raise ValueError("Chỉ hỗ trợ train/dev hoặc train/dev/test.")
    tables, output = Path(tables_dir), Path(output_dir)
    lengths = read_table(tables / "sentence_lengths.csv", {"split", "length", "count"})
    upos = rates(read_table(tables / "upos_distribution.csv", {"split", "upos", "percentage"}),
                 splits, "percentage", 100)
    oov = rates(read_table(tables / "oov_stats.csv", {"split", "upos", "oov_rate"}),
                splits[1:], "oov_rate", 1)
    frequencies: dict[str, dict[int, int]] = {split: {} for split in splits}
    for row in lengths:
        split, length, count = row["split"], int(row["length"]), int(row["count"])
        if split not in frequencies or length < 0 or count <= 0 or length in frequencies[split]:
            raise ValueError("Phân bố độ dài không hợp lệ hoặc trùng.")
        frequencies[split][length] = count
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        for split in splits:
            pairs = sorted(frequencies[split].items())
            # Các cột rộng 1 token biểu diễn histogram đã tổng hợp, không tái tạo corpus.
            ax.bar([x for x, _ in pairs], [y for _, y in pairs], width=1,
                   alpha=0.4, edgecolor=COLORS[split], color=COLORS[split], label=split)
        ax.set(title="Phân bố độ dài câu theo tập dữ liệu", xlabel="Độ dài câu (số token)",
               ylabel="Số câu")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(title="Tập dữ liệu")
        path = output / "sentence_lengths.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

        for name, values, chart_splits, title, ylabel, scale in (
            ("upos_distribution", upos, splits, "Phân bố UPOS theo tập dữ liệu",
             "Tỷ lệ token trong tập (%)", 1),
            ("oov_by_upos", oov, splits[1:], "Tỷ lệ OOV theo UPOS — từ vựng train nguyên gốc",
             "Tỷ lệ OOV trong từng UPOS (%)", 100),
        ):
            fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
            width = 0.8 / len(chart_splits)
            for index, split in enumerate(chart_splits):
                positions = [i + (index - (len(chart_splits) - 1) / 2) * width for i in range(len(UPOS))]
                heights = [values[split, tag] * scale for tag in UPOS]
                ax.bar(positions, heights, width=width, label=split, color=COLORS[split])
                for x, y in zip(positions, heights):
                    if math.isnan(y):
                        ax.text(x, 1, "N/A", ha="center", va="bottom", rotation=90, fontsize=7)
            ax.set_xticks(range(len(UPOS)), UPOS, rotation=45, ha="right")
            ax.set_xlim(-0.6, len(UPOS) - 0.4)
            ceiling = 105 if name == "oov_by_upos" else max(
                5, math.ceil(max((v for v in values.values() if math.isfinite(v)), default=0) / 5) * 5 + 2,
            )
            ax.set(title=title, xlabel="Nhãn từ loại UPOS", ylabel=ylabel, ylim=(0, ceiling))
            ax.legend(title="Tập dữ liệu")
            if any(math.isnan(v) for v in values.values()):
                fig.supxlabel("N/A: không có token làm mẫu số")
            path = output / f"{name}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(path)
    return paths


def main() -> None:
    """Nhận đường dẫn CSV và thư mục ảnh từ dòng lệnh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    try:
        for path in make_figures(args.tables_dir, args.output_dir):
            print(path)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main()
