"""Run the fixed VTB train/dev protocol, including a single shared edge expansion."""

import argparse
from collections.abc import Mapping
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy
import sklearn

from src.contracts import UPOS_TAGS
from src.data_loader import read_conllu
from src.evaluation_outputs import (
    MODEL_CLASSES,
    as_corpus,
    fit_predict,
    write_csv,
    write_label_tables,
    write_predictions,
)
from src.hmm_common import HMMTrainingCache
from src.metrics import evaluate_pos
from src.model_selection import MACRO_F1_TOLERANCE, grid_search_pos


CORPUS_COMMIT = "fa7130860b65db9db342542c51dc57badd30fa9e"
DEV_FILES = {
    "train": ("vi_vtb-ud-train.conllu", "1df180722db99b724f7368aca50597133bee28606de3394cd8d12a57da89978d"),
    "dev": ("vi_vtb-ud-dev.conllu", "0b779257b9225575b3b30e15d64cc4ef9945a77f8cceb9f92d794b483a2752dd"),
}
INITIAL_GRID = {"alpha": [0.01, 0.1, 1.0], "cutoff": [0, 1, 2]}
FIXED_PARAMS = {"alpha": 0.1, "cutoff": 1}
METRIC_COLUMNS = (
    "token_accuracy", "macro_f1", "oov_accuracy", "oov_rate", "accuracy_no_punct",
)
DEFAULT_DATA = Path(".venv/benchmark_data/ud-vtb-fa7130860b65")
DEFAULT_SUMMARIES = Path("results/tables/dev")
DEFAULT_PREDICTIONS = Path(".venv/project-runs/dev/predictions_dev.csv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train_dev(data_dir: Path):
    """Verify Train and Dev before parsing. No Test file path is constructed."""
    dataset: dict[str, Any] = {"source_commit": CORPUS_COMMIT}
    loaded = {}
    for split, (filename, expected) in DEV_FILES.items():
        path = data_dir / filename
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{split} SHA-256 mismatch for the pinned VTB commit.")
        sentences = read_conllu(path, nfc=False)
        loaded[split] = sentences
        dataset[split] = {
            "sha256": actual,
            "sentences": len(sentences),
            "tokens": sum(len(sentence.tokens) for sentence in sentences),
        }
    return loaded["train"], loaded["dev"], dataset


def expanded_grid(winners: list[dict]) -> tuple[dict | None, list[str]]:
    """Extend any reached practical boundary in a common Cartesian grid."""
    alphas = list(INITIAL_GRID["alpha"])
    cutoffs = list(INITIAL_GRID["cutoff"])
    edges = []
    if any(row["param_alpha"] == alphas[0] for row in winners):
        edges.append("alpha_low")
        alphas.insert(0, 0.001)
    if any(row["param_alpha"] == alphas[-1] for row in winners):
        edges.append("alpha_high")
        alphas.append(3.0)
    if any(row["param_cutoff"] == cutoffs[-1] for row in winners):
        edges.append("cutoff_high")
        cutoffs.append(3)
    return ({"alpha": alphas, "cutoff": cutoffs} if edges else None), edges


def _params(row: dict) -> dict:
    return {"alpha": row["param_alpha"], "cutoff": row["param_cutoff"]}


def _save_grid(path: Path, rows: Mapping[str, list[dict]]) -> None:
    header = ("model", "param_alpha", "param_cutoff", *("dev_" + key for key in METRIC_COLUMNS),
              "dev_score", "rank", "selected")
    write_csv(
        path, header,
        ({"model": model, **row} for model, model_rows in rows.items() for row in model_rows),
    )


def evaluate_dev(data_dir: Path, output_dir: Path, predictions_path: Path) -> dict:
    """Fit on Train, select on Dev and export local predictions plus aggregates."""
    train_sentences, dev_sentences, dataset = load_train_dev(data_dir)
    train_data = as_corpus(train_sentences)
    dev_data = as_corpus(dev_sentences)
    cache = HMMTrainingCache()

    def search(kind, params):
        def build(**configuration):
            return MODEL_CLASSES[kind](cache=cache, **configuration)

        return grid_search_pos(build, train_data, dev_data, params)

    initial_rows = {kind: search(kind, INITIAL_GRID) for kind in ("bigram", "trigram")}
    expansion, edges = expanded_grid([rows[0] for rows in initial_rows.values()])
    final_rows = (
        {kind: search(kind, expansion) for kind in initial_rows}
        if expansion is not None else initial_rows
    )
    models = {
        "b0": ("baseline", {}),
        "h1_best": ("bigram", _params(final_rows["bigram"][0])),
        "h2_best": ("trigram", _params(final_rows["trigram"][0])),
        "h1_fixed": ("bigram", dict(FIXED_PARAMS)),
        "h2_fixed": ("trigram", dict(FIXED_PARAMS)),
    }
    fitted_predictions = {}
    model_predictions = {}
    for name, (kind, params) in models.items():
        key = (kind, tuple(sorted(params.items())))
        if key not in fitted_predictions:
            fitted_predictions[key] = fit_predict(kind, params, train_data, dev_data, cache)
        model_predictions[name] = fitted_predictions[key][0]
    train_vocab = fitted_predictions[("baseline", ())][1]
    if any(vocab != train_vocab for _, vocab in fitted_predictions.values()):
        raise ValueError("All models must use the same raw Train vocabulary for OOV.")
    write_predictions(
        predictions_path, dev_sentences,
        {
            "pred_b0": model_predictions["b0"],
            "pred_h1": model_predictions["h1_best"],
            "pred_h2": model_predictions["h2_best"],
            "pred_h1_fixed": model_predictions["h1_fixed"],
            "pred_h2_fixed": model_predictions["h2_fixed"],
        },
        train_vocab,
    )

    summary_models = {}
    for name, (kind, params) in models.items():
        predictions = model_predictions[name]
        metrics = evaluate_pos(
            dev_data, predictions, train_vocab, UPOS_TAGS, macro_f1_gold_support=True,
        )
        summary_models[name] = {"kind": kind, "params": params, "metrics": metrics}
        write_label_tables(output_dir, name, dev_data, predictions)
    write_csv(
        output_dir / "metrics_dev.csv", ("model", "alpha", "cutoff", *METRIC_COLUMNS),
        ({"model": name, **record["params"], **record["metrics"]}
         for name, record in summary_models.items()),
    )
    _save_grid(output_dir / "grid_initial_dev.csv", initial_rows)
    if expansion is not None:
        _save_grid(output_dir / "grid_expanded_dev.csv", final_rows)

    fixed_deltas = {
        key: summary_models["h2_fixed"]["metrics"][key] - summary_models["h1_fixed"]["metrics"][key]
        for key in METRIC_COLUMNS
    }
    write_csv(output_dir / "ablation_fixed_dev.csv", ("model", "alpha", "cutoff", *METRIC_COLUMNS), [
        {"model": "h1_fixed", **FIXED_PARAMS, **summary_models["h1_fixed"]["metrics"]},
        {"model": "h2_fixed", **FIXED_PARAMS, **summary_models["h2_fixed"]["metrics"]},
        {"model": "h2_fixed_minus_h1_fixed", **FIXED_PARAMS, **fixed_deltas},
    ])
    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "protocol": {
            "name": "v4.1",
            "selection": "dev_token_accuracy_desc, dev_macro_f1_desc, cutoff_asc, alpha_desc",
            "macro_f1_scope": "gold_support",
            "macro_f1_tolerance": MACRO_F1_TOLERANCE,
            "initial_grid": INITIAL_GRID,
            "expanded_grid": expansion,
            "expansion_triggers": edges,
            "initial_winners": {kind: _params(rows[0]) for kind, rows in initial_rows.items()},
            "fixed_params": FIXED_PARAMS,
            "token_normalization": "preserve_FORM",
        },
        "environment": {
            "python": platform.python_version(), "numpy": numpy.__version__, "sklearn": sklearn.__version__,
        },
        "models": summary_models,
        "ablation_h2_minus_h1": fixed_deltas,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARIES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    options = parser.parse_args()
    selection = evaluate_dev(options.data, options.output, options.predictions)
    for name, record in selection["models"].items():
        metrics = record["metrics"]
        print(f"{name} {record['params']}  acc={metrics['token_accuracy']:.6f}  macro_f1={metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
