"""Đếm lỗi dev theo FORM nguyên gốc, nhãn gold và nhãn đã quan sát trong train."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from scripts.prepare_tv1 import check_predictions, pinned_split
from src.data_loader import read_conllu


PREDICTIONS = {"b0": "pred_b0", "bigram": "pred_h1", "trigram": "pred_h2"}


def profile(counts: Counter) -> str:
    return ", ".join(f"{tag}={n}" for tag, n in counts.most_common())


def audit(data_dir: Path, predictions: Path, output: Path) -> None:
    train = read_conllu(pinned_split(data_dir, "train")[0])
    dev = read_conllu(pinned_split(data_dir, "dev")[0])
    check_predictions(predictions, dev, split="dev")
    train_tags = defaultdict(Counter)
    for sentence in train:
        for form, tag in zip(sentence.tokens, sentence.tags, strict=True):
            train_tags[form][tag] += 1

    dev_tags = defaultdict(Counter)
    by_form_gold = {}
    groups = defaultdict(Counter)
    with predictions.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            form, gold = row["token"], row["gold"]
            seen = train_tags[form]
            if not seen:
                group = "oov"
            elif gold not in seen:
                group = "new_tag"
            elif len(seen) == 1:
                group = "single_seen_tag"
            else:
                group = "multiple_seen_tags"
            dev_tags[form][gold] += 1
            key = form, gold
            if key not in by_form_gold:
                by_form_gold[key] = {
                    "group": group, "counts": {name: Counter() for name in PREDICTIONS}, "example": ""
                }
            entry = by_form_gold[key]
            groups[group]["tokens"] += 1
            for name, column in PREDICTIONS.items():
                pred = row[column]
                entry["counts"][name][pred] += 1
                if pred != gold:
                    groups[group][name] += 1
                    if name == "trigram" and not entry["example"]:
                        entry["example"] = f"{row['sent_id']}/{row['idx']}"

    rows = []
    for (form, gold), entry in by_form_gold.items():
        n = dev_tags[form][gold]
        errors = {name: n - counts[gold] for name, counts in entry["counts"].items()}
        if any(errors.values()):
            rows.append({
                "form": form, "gold": gold, "group": entry["group"], "gold_tokens_dev": n,
                "train_tags": profile(train_tags[form]), "dev_tags": profile(dev_tags[form]),
                **{f"{name}_errors": count for name, count in errors.items()},
                "trigram_predictions": profile(entry["counts"]["trigram"]),
                "trigram_error_sent_id_idx0": entry["example"],
            })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "word_errors_dev.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("form", "gold", "group", "gold_tokens_dev", "train_tags", "dev_tags",
                  "b0_errors", "bigram_errors", "trigram_errors", "trigram_predictions",
                  "trigram_error_sent_id_idx0")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (-row["trigram_errors"], row["form"], row["gold"])))
    with (output / "groups_dev.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "tokens", *(f"{name}_errors" for name in PREDICTIONS)])
        for group, counts in sorted(groups.items()):
            writer.writerow([group, counts["tokens"], *(counts[name] for name in PREDICTIONS)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.data_dir, args.predictions, args.output)
