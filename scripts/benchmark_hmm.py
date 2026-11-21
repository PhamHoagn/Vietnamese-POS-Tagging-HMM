"""Measure HMM CPU decoder runtime on train/dev only, without evaluating accuracy."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np

from src.contracts import validate_hmm_config
from src.data_loader import load_conllu
from src.hmm_bigram import BigramHMMTagger
from src.hmm_common import HMMTrainingCache
from src.hmm_trigram import TrigramHMMTagger


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representative_indices(lengths: list[int], count: int) -> list[int]:
    """Pick deterministic length quantiles, including shortest and longest.

    Sắp câu theo (độ dài, vị trí gốc), lấy các rank cách đều nhau.
    Không dùng token gold hoặc điểm mô hình để chọn mẫu; không lấy lặp câu.
    Khi corpus có ít câu hơn yêu cầu, dùng mọi câu và ghi đúng số đo thực tế.
    """
    if count < 1 or not lengths or any(length < 1 for length in lengths):
        raise ValueError("Need positive sample size and non-empty dev sentences.")
    ranked = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    positions = np.linspace(0, len(ranked) - 1, min(count, len(ranked)), dtype=int)
    return [ranked[int(position)] for position in positions]


def benchmark(
    train_path: str | Path, dev_path: str | Path, output_path: str | Path, *,
    model_kind: str = "trigram", sample_size: int = 100, repeats: int = 3,
    safety_factor: float = 1.5, budget_minutes: float = 90.0, corpus_ref: str = "",
    alphas: tuple[float, ...] = (.01, .1, 1.0), cutoffs: tuple[int, ...] = (0, 1, 2),
) -> tuple[list[dict], dict]:
    """Time fit and median decode for a fixed runtime grid; never select by gold.

    Chỉ truyền train vào fit; dev cung cấp token để đo decoder, không tính metric.
    Tái dùng cache counts/mapping giữa các cấu hình, warm-up ngoài thời gian decode.
    Ước lượng full-dev = safety * tổng(fit_seconds + seconds_per_token * dev_tokens).
    Đây là dự phóng chi phí CPU, không phải grid search chất lượng của TV3.
    """
    train_path, dev_path, output = Path(train_path), Path(dev_path), Path(output_path)
    if (model_kind not in ("bigram", "trigram", "both") or sample_size < 1 or repeats < 1
            or not np.isfinite(safety_factor) or safety_factor < 1
            or not np.isfinite(budget_minutes) or budget_minutes <= 0):
        raise ValueError("Invalid model, sample/repeats, safety factor or time budget.")
    if not alphas or not cutoffs or len(set(alphas)) != len(alphas) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Runtime parameter lists must be non-empty and unique.")
    for alpha in alphas:
        for cutoff in cutoffs:
            validate_hmm_config(alpha, cutoff)
    train_hash, dev_hash = file_sha256(train_path), file_sha256(dev_path)
    if train_path.resolve() == dev_path.resolve() or train_hash == dev_hash:
        raise ValueError("Train and dev must be separate, different files.")
    metadata_path = output.with_suffix(".json")
    if any(path.resolve() in (train_path.resolve(), dev_path.resolve()) for path in (output, metadata_path)):
        raise ValueError("Benchmark output must not overwrite input data.")
    if output.suffix.lower() != ".csv":
        raise ValueError("Benchmark output must have a .csv extension.")
    train = load_conllu(train_path)
    # Discard dev gold before choosing the sample or calling the decoder.
    dev_words = [words for words, _ in load_conllu(dev_path)]
    indices = representative_indices([len(words) for words in dev_words], sample_size)
    sample = [dev_words[index] for index in indices]
    sample_tokens = sum(map(len, sample))
    dev_tokens = sum(map(len, dev_words))
    train_tokens = sum(len(words) for words, _ in train)
    if not train_tokens:
        raise ValueError("Train must contain tokens.")
    cache = HMMTrainingCache()
    kinds = ("bigram", "trigram") if model_kind == "both" else (model_kind,)
    classes = {"bigram": BigramHMMTagger, "trigram": TrigramHMMTagger}
    measured_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for kind in kinds:
        for alpha in alphas:
            for cutoff in cutoffs:
                started = perf_counter()
                model = classes[kind](alpha, cutoff, cache=cache).fit(train)
                fit_seconds = perf_counter() - started
                model.predict(sample[0])  # Warm-up is excluded from decoder timings.
                durations = []
                for _ in range(repeats):
                    started = perf_counter()
                    predictions = model.predict_corpus(sample)
                    durations.append(perf_counter() - started)
                    if (len(predictions) != len(sample)
                            or any(len(words) != len(tags) for words, tags in zip(sample, predictions))):
                        raise RuntimeError("Decoder returned misaligned predictions.")
                decode_seconds = median(durations)
                seconds_per_token = decode_seconds / sample_tokens
                rows.append({
                    "measured_at_utc": measured_at, "model": kind,
                    "alpha": alpha, "cutoff": cutoff, "tags": len(model.tags),
                    "train_sentences": len(train), "train_tokens": train_tokens,
                    "dev_sentences": len(dev_words), "dev_tokens": dev_tokens,
                    "sample_sentences": len(sample), "sample_tokens": sample_tokens,
                    "sample_min_length": min(map(len, sample)), "sample_max_length": max(map(len, sample)),
                    "repeats": repeats, "fit_seconds": fit_seconds,
                    "decode_seconds_median": decode_seconds, "seconds_per_token": seconds_per_token,
                    "projected_full_dev_seconds": fit_seconds + seconds_per_token * dev_tokens,
                    "train_sha256": train_hash, "dev_sha256": dev_hash,
                })
    projections = {
        kind: safety_factor * sum(row["projected_full_dev_seconds"] for row in rows if row["model"] == kind)
        for kind in kinds
    }
    projected_total = sum(projections.values())
    source_root = Path(__file__).resolve().parents[1]
    metadata = {
        "measured_at_utc": measured_at, "corpus_ref": corpus_ref,
        "train_file": train_path.name, "dev_file": dev_path.name,
        "train_sha256": train_hash, "dev_sha256": dev_hash,
        "sample_indices_0based": indices,
        "sample_selection": "equally spaced ranks after sorting by (token_count, original_index)",
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "code_sha256": {name: file_sha256(source_root / name) for name in
                        ("src/hmm_common.py", "src/hmm_bigram.py", "src/hmm_trigram.py", "scripts/benchmark_hmm.py")},
        "alphas": list(alphas), "cutoffs": list(cutoffs), "configurations_per_model": len(alphas) * len(cutoffs),
        "safety_factor": safety_factor, "budget_minutes": budget_minutes,
        "projected_seconds_by_model": projections, "projected_total_seconds": projected_total,
        "within_budget": projected_total <= budget_minutes * 60,
        "scope": "Runtime only. Train fit; dev token decoding. No accuracy, tuning, or test corpus.",
        "limitations": "Token-rate extrapolation on this machine; excludes loading, metrics, CSV prediction export, and CI.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/tables/bench_dev.csv"))
    parser.add_argument("--model", choices=("bigram", "trigram", "both"), default="trigram")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--safety-factor", type=float, default=1.5)
    parser.add_argument("--budget-minutes", type=float, default=90)
    parser.add_argument("--corpus-ref", default="")
    args = parser.parse_args(argv)
    try:
        rows, metadata = benchmark(args.train, args.dev, args.output, model_kind=args.model,
                                   sample_size=args.sample_size, repeats=args.repeats,
                                   safety_factor=args.safety_factor, budget_minutes=args.budget_minutes,
                                   corpus_ref=args.corpus_ref)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
    print(f"Measured {rows[0]['sample_sentences']} dev sentences, {rows[0]['sample_tokens']} tokens.")
    for kind, seconds in metadata["projected_seconds_by_model"].items():
        print(f"{kind}: projected {metadata['configurations_per_model']} configs = {seconds / 60:.3f} minutes (safety included).")
    passed = metadata["within_budget"]
    print(f"Runtime gate: {'PASS' if passed else 'OVER BUDGET'}; budget={args.budget_minutes:g} minutes.")
    print(f"Saved {args.output} and {args.output.with_suffix('.json')}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
