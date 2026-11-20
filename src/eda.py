"""Thống kê các split đã parse; không học tham số hay sửa dữ liệu đầu vào."""

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from pathlib import Path
from statistics import mean, median

from .contracts import UPOS_TAGS
from .data_loader import Sentence


def export_eda(
    splits: Mapping[str, Sequence[Sentence]],
    output_dir: str | Path = "results/tables",
) -> dict[str, Path]:
    """Xuất hai CSV; chỉ truyền test thật sau khi đã khóa cấu hình."""
    split_order = ("train", "dev", "test")
    if set(splits) != set(split_order):
        raise ValueError("Đầu vào phải có đúng ba split train/dev/test.")

    summaries: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    for split in split_order:
        lengths: list[int] = []
        types: set[str] = set()
        counts: Counter[str] = Counter()
        for index, sentence in enumerate(splits[split]):
            if not isinstance(sentence, Sentence):
                raise TypeError(f"{split}[{index}]: cần Sentence từ read_conllu().")
            if len(sentence.tokens) != len(sentence.tags):
                raise ValueError(f"{split}[{index}]: số token và nhãn khác nhau.")
            if any(not isinstance(token, str) or not token for token in sentence.tokens):
                raise ValueError(f"{split}[{index}]: FORM phải là chuỗi không rỗng.")
            if any(tag not in UPOS_TAGS for tag in sentence.tags):
                raise ValueError(f"{split}[{index}]: nhãn ngoài 17 UPOS.")
            lengths.append(len(sentence.tokens))
            types.update(sentence.tokens)
            counts.update(sentence.tags)
        token_count = sum(lengths)
        summaries.append({
            "split": split,
            "sentence_count": len(lengths),
            "token_count": token_count,
            "sentence_length_min": min(lengths) if lengths else None,
            "sentence_length_mean": mean(lengths) if lengths else None,
            "sentence_length_median": median(lengths) if lengths else None,
            "sentence_length_max": max(lengths) if lengths else None,
            "token_type_count": len(types),
            "punct_token_count": counts["PUNCT"],
        })
        for tag in UPOS_TAGS:
            distributions.append({
                "split": split, "upos": tag, "count": counts[tag],
                "percentage": 100.0 * counts[tag] / token_count if token_count else None,
            })

    # Chỉ tạo file sau khi cả ba split đều qua kiểm tra.
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"eda": directory / "eda.csv",
             "upos_distribution": directory / "upos_distribution.csv"}
    for key, rows in (("eda", summaries), ("upos_distribution", distributions)):
        with paths[key].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths
