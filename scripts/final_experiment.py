"""Prepare, seal and execute the Vietnamese VTB final evaluation."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CORPUS_REF = "fa7130860b65db9db342542c51dc57badd30fa9e"
RAW_ROOT = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/{CORPUS_REF}"
TRAIN_SHA256 = "1df180722db99b724f7368aca50597133bee28606de3394cd8d12a57da89978d"
DEV_SHA256 = "0b779257b9225575b3b30e15d64cc4ef9945a77f8cceb9f92d794b483a2752dd"
TEST_NAME = "vi_vtb-ud-test.conllu"
TEST_SIZE = 538633
TEST_GIT_BLOB = "29cfccf7bc782829f77d039db5d906157a747ae6"
MODEL_KINDS = {"b0": "baseline", "h1_best": "bigram", "h2_best": "trigram",
               "h1_fixed": "bigram", "h2_fixed": "trigram"}
FIXED = {"alpha": 0.1, "cutoff": 1}
BOOTSTRAP = {"replicates": 2000, "seed": 42, "confidence": 0.95}
COMPARISONS = (("h1_best", "b0"), ("h2_best", "h1_best"), ("h2_fixed", "h1_fixed"))
CODE_PATHS = (
    "requirements.txt", "scripts/__init__.py", "scripts/evaluate_dev.py", "scripts/final_experiment.py",
    "src/__init__.py", "src/baseline.py", "src/bootstrap.py", "src/contracts.py", "src/data_loader.py",
    "src/error_script.py", "src/evaluation_outputs.py", "src/hmm_bigram.py", "src/hmm_common.py",
    "src/hmm_tables.py", "src/hmm_trigram.py", "src/metrics.py", "src/model_selection.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(record: dict) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(encoded.encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_new_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _source_hashes() -> dict[str, str]:
    # Text-mode reads normalize CRLF and LF for equivalent Windows/Linux checkouts.
    return {name: _sha256((ROOT / name).read_text(encoding="utf-8").encode("utf-8"))
            for name in CODE_PATHS}


def _code_commit() -> str:
    def git(*args):
        return subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)

    tracked = git("ls-files", "--error-unmatch", "--", *CODE_PATHS)
    clean = git("diff", "--quiet", "HEAD", "--", *CODE_PATHS)
    head = git("rev-parse", "HEAD")
    if tracked.returncode or clean.returncode or head.returncode:
        raise ValueError("Commit the evaluation code before preparing the review candidate.")
    return head.stdout.strip()


def _model_configs(selection: dict) -> dict:
    from src.contracts import validate_hmm_config

    models = selection["models"]
    if set(models) != set(MODEL_KINDS):
        raise ValueError("Dev selection must contain the five planned model roles.")
    configs = {}
    for role, kind in MODEL_KINDS.items():
        model = models[role]
        params = model["params"]
        required = set() if kind == "baseline" else set(FIXED)
        if model["kind"] != kind or set(params) != required:
            raise ValueError(f"Unexpected model kind or parameters for {role}.")
        if kind != "baseline":
            validate_hmm_config(**params)
        if role.endswith("_fixed") and params != FIXED:
            raise ValueError(f"Unexpected fixed ablation parameters for {role}.")
        metrics = model["metrics"]
        for measure in ("token_accuracy", "macro_f1"):
            value = metrics[measure]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid dev {measure} for {role}.")
        configs[role] = {"kind": kind, "params": params, "dev_metrics": metrics}
    return configs


def prepare(selection_path: Path, output: Path) -> dict:
    """Record the reviewed dev selection and pinned test identity from Git metadata."""
    selected_bytes = selection_path.read_bytes()
    selected = json.loads(selected_bytes)
    if selected["schema_version"] != 1:
        raise ValueError("Unsupported dev selection schema.")
    dataset = selected["dataset"]
    if (dataset["source_commit"], dataset["train"]["sha256"], dataset["dev"]["sha256"]) != (
        CORPUS_REF, TRAIN_SHA256, DEV_SHA256,
    ):
        raise ValueError("Dev selection was not produced from the pinned train/dev files.")
    candidate = {
        "schema_version": 1,
        "prepared_at_utc": _utc_now(),
        "dataset": {
            "corpus_ref": CORPUS_REF, "train_sha256": TRAIN_SHA256, "dev_sha256": DEV_SHA256,
            "test_git_blob_sha1": TEST_GIT_BLOB, "test_size_bytes": TEST_SIZE,
        },
        "models": _model_configs(selected),
        "dev_selection_sha256": _sha256(selected_bytes),
        "dev_protocol": selected["protocol"],
        "dev_environment": selected.get("environment", {}),
        "code_commit": _code_commit(),
        "bootstrap": BOOTSTRAP,
        "comparisons": [list(pair) for pair in COMPARISONS],
        "source_sha256": _source_hashes(),
    }
    _write_new_json(output, candidate)
    return candidate


def _validate_candidate(candidate: dict) -> None:
    if candidate.get("schema_version") != 1 or candidate.get("source_sha256") != _source_hashes():
        raise ValueError("Unsupported candidate or source files changed since preparation.")
    if candidate.get("dataset") != {
        "corpus_ref": CORPUS_REF, "train_sha256": TRAIN_SHA256, "dev_sha256": DEV_SHA256,
        "test_git_blob_sha1": TEST_GIT_BLOB, "test_size_bytes": TEST_SIZE,
    }:
        raise ValueError("Candidate dataset identity does not match the pinned corpus.")
    if candidate.get("bootstrap") != BOOTSTRAP or candidate.get("comparisons") != [list(pair) for pair in COMPARISONS]:
        raise ValueError("Candidate inference protocol changed.")
    models = {role: {**model, "metrics": model["dev_metrics"]} for role, model in candidate["models"].items()}
    _model_configs({"models": models})


def seal(candidate_path: Path, output: Path, confirmation: str) -> dict:
    """Record the user's confirmation on behalf of the group after reviewing the candidate."""
    if not confirmation.strip():
        raise ValueError("Provide the recorded team confirmation reference.")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    _validate_candidate(candidate)
    locked = {"status": "locked", "locked_at_utc": _utc_now(), "candidate": candidate,
              "group_confirmation": confirmation.strip()}
    locked["lock_sha256"] = _digest(locked)
    _write_new_json(output, locked)
    return locked


def read_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked" or not lock.get("group_confirmation"):
        raise ValueError("Final evaluation requires the team's locked protocol.")
    expected = lock.pop("lock_sha256", None)
    if expected != _digest(lock):
        raise ValueError("Lock digest mismatch.")
    _validate_candidate(lock["candidate"])
    lock["lock_sha256"] = expected
    return lock


def _verified_test(path: Path) -> str:
    """Verify a Git blob by exact bytes before making it available to the parser."""
    existing = path.is_file()
    if existing:
        data = path.read_bytes()
    else:
        with urlopen(f"{RAW_ROOT}/{TEST_NAME}", timeout=60) as response:
            data = response.read(TEST_SIZE + 1)
    git_header = f"blob {len(data)}\0".encode("ascii")
    if len(data) != TEST_SIZE or hashlib.sha1(git_header + data).hexdigest() != TEST_GIT_BLOB:
        raise ValueError("Test bytes do not match the pinned Git blob.")
    if not existing:
        with path.open("xb") as stream:
            stream.write(data)
    return _sha256(data)


def run_locked(lock_path: Path, data_dir: Path, private_dir: Path, summary_dir: Path) -> dict:
    """Run the five frozen roles in a single evaluation with one fit per unique configuration."""
    lock = read_lock(lock_path)
    if (private_dir.exists() or summary_dir.exists()
            or private_dir.resolve().is_relative_to(summary_dir.resolve())
            or summary_dir.resolve().is_relative_to(private_dir.resolve())):
        raise ValueError("Use distinct, new private and summary output directories.")
    train_path = data_dir / "vi_vtb-ud-train.conllu"
    dev_path = data_dir / "vi_vtb-ud-dev.conllu"
    for path, expected in ((train_path, TRAIN_SHA256), (dev_path, DEV_SHA256)):
        if _sha256(path.read_bytes()) != expected:
            raise ValueError(f"SHA-256 mismatch for {path.name}.")
    test_path = data_dir / TEST_NAME
    test_sha256 = _verified_test(test_path)
    return _evaluate(lock, train_path, test_path, test_sha256, private_dir, summary_dir)


def _evaluate(lock, train_path, test_path, test_sha256, private_dir, summary_dir):
    import numpy as np
    import sklearn

    from src.bootstrap import paired_accuracy_intervals
    from src.contracts import UPOS_TAGS
    from src.data_loader import read_conllu
    from src.error_script import export_error_analysis
    from src.evaluation_outputs import as_corpus, fit_predict, write_csv, write_label_tables, write_predictions
    from src.hmm_common import HMMTrainingCache
    from src.metrics import evaluate_pos

    train = as_corpus(read_conllu(train_path))
    test_sentences = read_conllu(test_path)
    test = as_corpus(test_sentences)
    train_vocab = frozenset(token for tokens, _ in train for token in tokens)
    cache = HMMTrainingCache()
    unique = {}
    predictions = {}
    models = lock["candidate"]["models"]
    for role, model in models.items():
        kind, params = model["kind"], model["params"]
        key = (kind, tuple(sorted(params.items())))
        if key not in unique:
            tags, vocab = fit_predict(kind, params, train, test, cache)
            if vocab != train_vocab:
                raise ValueError(f"{role} must expose the raw training vocabulary.")
            unique[key] = tags
        predictions[role] = unique[key]

    private_dir.mkdir(parents=True, exist_ok=False)
    summary_dir.mkdir(parents=True, exist_ok=False)
    pred_path = private_dir / "predictions_test.csv"
    columns = {"b0": "pred_b0", "h1_best": "pred_h1", "h2_best": "pred_h2",
               "h1_fixed": "pred_h1_fixed", "h2_fixed": "pred_h2_fixed"}
    write_predictions(pred_path, test_sentences, {columns[role]: tags for role, tags in predictions.items()}, train_vocab)

    metrics = []
    for role, tags in predictions.items():
        score = evaluate_pos(test, tags, train_vocab, UPOS_TAGS, macro_f1_gold_support=True)
        metrics.append({"model": role, **models[role]["params"], **score})
        write_label_tables(summary_dir, role, test, tags)
    score_names = [name for name in metrics[0] if name not in {"model", "alpha", "cutoff"}]
    write_csv(summary_dir / "metrics_test.csv", ("model", "alpha", "cutoff", *score_names), metrics)
    intervals = paired_accuracy_intervals(
        test, predictions, comparisons=COMPARISONS, resamples=BOOTSTRAP["replicates"], seed=BOOTSTRAP["seed"],
    )
    write_csv(summary_dir / "paired_bootstrap_test.csv", tuple(intervals[0]), intervals)
    export_error_analysis(pred_path, train_vocab, private_dir / "errors", split="test", quiet=True)

    metadata = {
        "run_completed_at_utc": _utc_now(), "lock_sha256": lock["lock_sha256"],
        "test_sha256": test_sha256, "train_sentences": len(train), "test_sentences": len(test),
        "train_tokens": sum(len(tokens) for tokens, _ in train),
        "test_tokens": sum(len(tokens) for tokens, _ in test),
        "test_oov_tokens": sum(word not in train_vocab for tokens, _ in test for word in tokens),
        "test_non_punct_tokens": sum(tag != "PUNCT" for _, gold in test for tag in gold),
        "unique_model_configurations": len(unique), "registered_roles": list(models),
        "python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "prediction_sha256": _sha256(pred_path.read_bytes()),
        "summary_sha256": {file.name: _sha256(file.read_bytes()) for file in sorted(summary_dir.glob("*.csv"))},
    }
    _write_new_json(summary_dir / "run_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="write a review candidate without accessing test")
    prep.add_argument("--selection", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    freeze = commands.add_parser("seal", help="record the reviewed candidate and group confirmation")
    freeze.add_argument("--candidate", required=True, type=Path)
    freeze.add_argument("--confirmation", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run", help="execute a previously locked test evaluation")
    run.add_argument("--lock", required=True, type=Path)
    run.add_argument("--data-dir", required=True, type=Path)
    run.add_argument("--private-dir", required=True, type=Path)
    run.add_argument("--summary-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.selection, args.output)
            print(f"Prepared {args.output} for team review.")
        elif args.command == "seal":
            seal(args.candidate, args.output, args.confirmation)
            print(f"Sealed {args.output}.")
        else:
            run_locked(args.lock, args.data_dir, args.private_dir, args.summary_dir)
            print(f"Final aggregate results: {args.summary_dir}")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
