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

DEFAULT_SPLIT_ORDER = ("train", "dev", "test")
TRAIN_DEV_SPLIT_ORDER = ("train", "dev")


def _check_splits(splits: Mapping[str, Sequence[Sentence]], split_order: tuple[str, ...]) -> None:
    if split_order not in (DEFAULT_SPLIT_ORDER, TRAIN_DEV_SPLIT_ORDER):
        raise ValueError("Thứ tự split phải là train/dev hoặc train/dev/test.")
    if set(splits) != set(split_order):
        raise ValueError(f"Đầu vào phải có đúng các split {'/'.join(split_order)}.")


def export_sparsity(
    train_sentences: Sequence[Sentence],
    dev_sentences: Sequence[Sentence],
    output_path: str | Path = "results/tables/sparsity.csv",
) -> Path:
    """Xuất sparsity học từ train và độ mới bigram nhãn của dev; không sửa đầu vào."""
    def checked(sentences: Sequence[Sentence], name: str) -> None:
        """Xác nhận Sentence, căn chỉnh và UPOS trước khi tạo thống kê."""
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, Sentence):
                raise TypeError(f"{name}[{index}]: cần Sentence từ read_conllu().")
            if len(sentence.tokens) != len(sentence.tags):
                raise ValueError(f"{name}[{index}]: số token và nhãn khác nhau.")
            if any(not isinstance(token, str) or not token for token in sentence.tokens):
                raise ValueError(f"{name}[{index}]: FORM phải là chuỗi không rỗng.")
            if any(tag not in UPOS_TAGS for tag in sentence.tags):
                raise ValueError(f"{name}[{index}]: nhãn ngoài 17 UPOS.")

    checked(train_sentences, "train")
    checked(dev_sentences, "dev")
    start, end = "<s>", "</s>"
    labels = tuple(tag for tag in UPOS_TAGS if any(tag in sentence.tags for sentence in train_sentences))
    form_counts = Counter(token for sentence in train_sentences for token in sentence.tokens)
    form_labels: dict[str, set[str]] = {}
    for sentence in train_sentences:
        for token, tag in zip(sentence.tokens, sentence.tags):
            form_labels.setdefault(token, set()).add(tag)
    ambiguous_types = {token for token, tags in form_labels.items() if len(tags) > 1}
    ambiguous_occurrences = sum(form_counts[token] for token in ambiguous_types)

    def transitions(tags: Sequence[str]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
        """Tạo transition bigram/trigram có start và EOS cho một câu."""
        padded_bigram = (start, *tags, end)
        padded_trigram = (start, start, *tags, end)
        return (
            set(zip(padded_bigram, padded_bigram[1:])),
            set(zip(padded_trigram, padded_trigram[1:], padded_trigram[2:])),
        )

    train_bigrams: set[tuple[str, str]] = set()
    train_trigrams: set[tuple[str, str, str]] = set()
    dev_bigram_count = 0
    dev_unseen_bigram_count = 0
    dev_trigram_count = 0
    dev_unseen_trigram_count = 0
    dev_unseen_trigram_history_count = 0
    for sentence in train_sentences:
        bigrams, trigrams = transitions(sentence.tags)
        train_bigrams.update(bigrams)
        train_trigrams.update(trigrams)
    train_trigram_histories = {(first, second) for first, second, _ in train_trigrams}
    for sentence in dev_sentences:
        # Một transition chỉ xuất hiện một lần trong mỗi câu sau khi dùng set;
        # cần tính theo lượt nên dựng lại chuỗi có lặp ở đây.
        padded = (start, *sentence.tags, end)
        for pair in zip(padded, padded[1:]):
            dev_bigram_count += 1
            dev_unseen_bigram_count += pair not in train_bigrams
        padded_trigram = (start, *padded)
        for triple in zip(padded_trigram, padded_trigram[1:], padded_trigram[2:]):
            dev_trigram_count += 1
            dev_unseen_trigram_count += triple not in train_trigrams
            dev_unseen_trigram_history_count += triple[:2] not in train_trigram_histories

    state_count = len(labels) + 1  # S cộng đúng một ký hiệu start/EOS theo từng chiều.
    token_count = sum(form_counts.values())
    row = {
        "train_label_count": len(labels),
        "train_token_count": token_count,
        "ambiguous_token_type_count": len(ambiguous_types),
        "ambiguous_token_occurrence_count": ambiguous_occurrences,
        "ambiguous_token_occurrence_percentage": (
            100.0 * ambiguous_occurrences / token_count if token_count else None
        ),
        "bigram_observed_count": len(train_bigrams),
        "bigram_possible_count": state_count ** 2,
        "bigram_observed_percentage": 100.0 * len(train_bigrams) / state_count ** 2,
        "trigram_observed_count": len(train_trigrams),
        "trigram_possible_count": state_count ** 3,
        "trigram_observed_percentage": 100.0 * len(train_trigrams) / state_count ** 3,
        "dev_bigram_transition_count": dev_bigram_count,
        "dev_unseen_bigram_transition_count": dev_unseen_bigram_count,
        "dev_unseen_bigram_percentage": (
            100.0 * dev_unseen_bigram_count / dev_bigram_count if dev_bigram_count else None
        ),
        "dev_trigram_transition_count": dev_trigram_count,
        "dev_unseen_trigram_transition_count": dev_unseen_trigram_count,
        "dev_unseen_trigram_percentage": (
            100.0 * dev_unseen_trigram_count / dev_trigram_count if dev_trigram_count else None
        ),
        "dev_unseen_trigram_history_count": dev_unseen_trigram_history_count,
        "dev_unseen_trigram_history_percentage": (
            100.0 * dev_unseen_trigram_history_count / dev_trigram_count if dev_trigram_count else None
        ),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output


def audit_splits(
    splits: Mapping[str, Sequence[Sentence]],
    output_dir: str | Path = "results/tables",
    *,
    split_order: tuple[str, ...] = DEFAULT_SPLIT_ORDER,
    include_tokens: bool = True,
) -> Path:
    """Báo trùng/rỗng/nhãn lạ trên câu đã parse, không đổi dữ liệu hay split."""
    _check_splits(splits, split_order)
    index: dict[str, dict[tuple[str, ...], list[int]]] = {}
    for split in split_order:
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
               "sentence_index_b", "sent_id_b", "token_index", "tag"]
    if include_tokens:
        columns.append("tokens_json")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()

        def report(issue: str, split: str, position: int, **extra: object) -> None:
            """Ghi vị trí theo chỉ số 0-based và token nguyên gốc dạng JSON."""
            sentence = splits[split][position]
            values = {
                "issue": issue, "split_a": split, "sentence_index_a": position,
                "sent_id_a": sentence.sent_id, **extra,
            }
            if include_tokens:
                values["tokens_json"] = json.dumps(sentence.tokens, ensure_ascii=False)
            writer.writerow(values)

        for split in split_order:
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
        for first_split, second_split in combinations(split_order, 2):
            for tokens, first_positions in index[first_split].items():
                for first, second in product(first_positions, index[second_split].get(tokens, [])):
                    report("duplicate_cross", first_split, first, split_b=second_split,
                           sentence_index_b=second,
                           sent_id_b=splits[second_split][second].sent_id)
                    totals[f"cross:{first_split}:{second_split}"] += 1

    for split in split_order:
        print(f"{split}: {totals[f'within:{split}']} cặp câu trùng nội bộ; "
              f"{totals[f'empty:{split}']} câu rỗng; "
              f"{totals[f'unknown:{split}']} vị trí nhãn lạ.")
    for first, second in combinations(split_order, 2):
        print(f"{first} ∩ {second}: {totals[f'cross:{first}:{second}']} cặp câu trùng.")
    print(f"Báo cáo: {path}. Dữ liệu và split được giữ nguyên.")
    return path


def export_eda(
    splits: Mapping[str, Sequence[Sentence]],
    output_dir: str | Path = "results/tables",
    *,
    split_order: tuple[str, ...] = DEFAULT_SPLIT_ORDER,
) -> dict[str, Path]:
    """Xuất bốn CSV; chỉ truyền test thật sau khi đã khóa cấu hình."""
    _check_splits(splits, split_order)

    summaries: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
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
        for length, count in sorted(Counter(lengths).items()):
            length_rows.append({"split": split, "length": length, "count": count})
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
    for split in split_order[1:]:
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

    # Chỉ tạo file sau khi tất cả split đã chọn đều qua kiểm tra.
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"eda": directory / "eda.csv",
             "upos_distribution": directory / "upos_distribution.csv",
             "oov_stats": directory / "oov_stats.csv",
             "sentence_lengths": directory / "sentence_lengths.csv"}
    for key, rows in (("eda", summaries), ("upos_distribution", distributions),
                      ("oov_stats", oov_rows), ("sentence_lengths", length_rows)):
        with paths[key].open("w", encoding="utf-8", newline="") as stream:
            fields = ["split", "length", "count"] if key == "sentence_lengths" else list(rows[0])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return paths
