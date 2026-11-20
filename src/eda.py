"""Thống kê các split đã parse; không học tham số hay sửa dữ liệu đầu vào."""

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from itertools import combinations, product
import json
from pathlib import Path
from statistics import mean, median

from .contracts import UPOS_TAGS
from .data_loader import Sentence


def audit_splits(
    splits: Mapping[str, Sequence[Sentence]],
    output_dir: str | Path = "results/tables",
) -> Path:
    """Báo trùng/rỗng/nhãn lạ trên câu đã parse, không đổi dữ liệu hay split."""
    order = ("train", "dev", "test")
    if set(splits) != set(order):
        raise ValueError("Đầu vào phải có đúng ba split train/dev/test.")
    index: dict[str, dict[tuple[str, ...], list[int]]] = {}
    for split in order:
        index[split] = {}
        for position, sentence in enumerate(splits[split]):
            if not isinstance(sentence, Sentence):
                raise TypeError(f"{split}[{position}]: cần Sentence từ read_conllu().")
            index[split].setdefault(tuple(sentence.tokens), []).append(position)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "duplicates.csv"
    totals: Counter[str] = Counter()
    columns = ["issue", "split_a", "sentence_index_a", "sent_id_a", "split_b",
               "sentence_index_b", "sent_id_b", "token_index", "tag", "tokens_json"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        def report(issue: str, split: str, position: int, **extra: object) -> None:
            """Ghi vị trí theo chỉ số 0-based và token nguyên gốc dạng JSON."""
            sentence = splits[split][position]
            writer.writerow({
                "issue": issue, "split_a": split, "sentence_index_a": position,
                "sent_id_a": sentence.sent_id,
                "tokens_json": json.dumps(sentence.tokens, ensure_ascii=False), **extra,
            })

        for split in order:
            for position, sentence in enumerate(splits[split]):
                if not sentence.tokens:
                    report("empty_sentence", split, position)
                    totals[f"empty:{split}"] += 1
                for token_index, tag in enumerate(sentence.tags):
                    if tag not in UPOS_TAGS:
                        report("unknown_tag", split, position, token_index=token_index, tag=tag)
                        totals[f"unknown:{split}"] += 1
            for positions in index[split].values():
                for first, second in combinations(positions, 2):
                    report("duplicate_within", split, first, split_b=split,
                           sentence_index_b=second, sent_id_b=splits[split][second].sent_id)
                    totals[f"within:{split}"] += 1
        for first_split, second_split in combinations(order, 2):
            for tokens, first_positions in index[first_split].items():
                for first, second in product(first_positions, index[second_split].get(tokens, [])):
                    report("duplicate_cross", first_split, first, split_b=second_split,
                           sentence_index_b=second,
                           sent_id_b=splits[second_split][second].sent_id)
                    totals[f"cross:{first_split}:{second_split}"] += 1

    for split in order:
        print(f"{split}: {totals[f'within:{split}']} cặp câu trùng nội bộ; "
              f"{totals[f'empty:{split}']} câu rỗng; "
              f"{totals[f'unknown:{split}']} vị trí nhãn lạ.")
    for first, second in combinations(order, 2):
        print(f"{first} ∩ {second}: {totals[f'cross:{first}:{second}']} cặp câu trùng.")
    print(f"Báo cáo: {path}. Dữ liệu và split được giữ nguyên.")
    return path


def export_eda(
    splits: Mapping[str, Sequence[Sentence]],
    output_dir: str | Path = "results/tables",
) -> dict[str, Path]:
    """Xuất ba CSV; chỉ truyền test thật sau khi đã khóa cấu hình."""
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

    # Đếm FORM nguyên gốc chỉ từ train, không cutoff hay cập nhật từ dev/test.
    train_counts = Counter(token for sentence in splits["train"] for token in sentence.tokens)
    oov_rows: list[dict[str, object]] = []
    for split in ("dev", "test"):
        groups: dict[str, Counter[str]] = {tag: Counter() for tag in ("ALL", *UPOS_TAGS)}
        for sentence in splits[split]:
            for token, tag in zip(sentence.tokens, sentence.tags):
                frequency = train_counts[token]
                for group in (groups["ALL"], groups[tag]):
                    group["token"] += 1
                    group["oov"] += frequency == 0
                    for cutoff in (1, 2):
                        group[f"f_train_le_{cutoff}"] += frequency <= cutoff
                        group[f"seen_rare_le_{cutoff}"] += 1 <= frequency <= cutoff
        for tag, counts in groups.items():
            total = counts["token"]
            row: dict[str, object] = {"split": split, "upos": tag, "token_count": total}
            for metric in ("oov", "f_train_le_1", "f_train_le_2",
                           "seen_rare_le_1", "seen_rare_le_2"):
                row[f"{metric}_count"] = counts[metric]
                row[f"{metric}_rate"] = counts[metric] / total if total else None
            oov_rows.append(row)

    # Chỉ tạo file sau khi cả ba split đều qua kiểm tra.
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"eda": directory / "eda.csv",
             "upos_distribution": directory / "upos_distribution.csv",
             "oov_stats": directory / "oov_stats.csv"}
    for key, rows in (("eda", summaries), ("upos_distribution", distributions),
                      ("oov_stats", oov_rows)):
        with paths[key].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths
