"""Compare emission policies on the pinned train/dev split at the published best alphas."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

from scripts.evaluate_dev import DEFAULT_DATA, load_train_dev
from src.baseline import MostFrequentTagger
from src.evaluation_outputs import as_corpus
from src.hmm_bigram import BigramHMMTagger
from src.hmm_common import HMMTrainingCache, add_alpha_logs
from src.hmm_trigram import TrigramHMMTagger


def hapax_emissions(model):
    """Keep all observed FORM counts and use one extra UNK pseudocount per hapax."""
    if model.cutoff != 0:
        raise ValueError("Hapax emissions require the complete train vocabulary.")
    counts = model.emission_counts_.copy()
    hapax = [model.word_to_index_[word] for word, n in model.word_counts_.items() if n == 1]
    counts[:, model.unk_index_] = counts[:, hapax].sum(axis=1)
    return add_alpha_logs(counts, model.alpha)


def predict_hapax(model, dev):
    emission = hapax_emissions(model)
    for words, _ in dev:
        columns = [model.word_to_index_.get(word, model.unk_index_) for word in words]
        yield model._viterbi(emission[:, columns].T)


def score_by_frequency(dev, frequencies, predictions):
    sizes, correct = Counter(), Counter()
    for (words, gold), tags in zip(dev, predictions, strict=True):
        for word, target, prediction in zip(words, gold, tags, strict=True):
            n = frequencies[word]
            group = ("oov", "once", "repeated")[min(n, 2)]
            sizes[group] += 1
            correct[group] += target == prediction
    sizes["all"] = sum(sizes.values())
    correct["all"] = sum(correct.values())
    return sizes, correct


def diagnose(data_dir: Path, stream):
    train_sentences, dev_sentences, _ = load_train_dev(data_dir)
    train, dev = as_corpus(train_sentences), as_corpus(dev_sentences)
    frequencies = Counter(word for words, _ in train for word in words)
    b0 = MostFrequentTagger().fit(train)
    sizes, baseline = score_by_frequency(dev, frequencies, (b0.predict(words) for words, _ in dev))
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["model", "alpha", "emissions", "train_frequency", "dev_tokens",
                     "correct", "accuracy", "contribution_vs_b0_pp"])

    def report(model, alpha, policy, counts):
        for group in ("repeated", "once", "oov", "all"):
            gain = 100 * (counts[group] - baseline[group]) / sizes["all"]
            writer.writerow([model, alpha, policy, group, sizes[group], counts[group],
                             counts[group] / sizes[group], gain])

    report("b0", "", "most_frequent", baseline)
    cache = HMMTrainingCache()
    for name, tagger, alpha in (("bigram", BigramHMMTagger, 0.01), ("trigram", TrigramHMMTagger, 0.1)):
        for cutoff in (0, 1):
            model = tagger(alpha=alpha, cutoff=cutoff, cache=cache).fit(train)
            _, counts = score_by_frequency(dev, frequencies, (model.predict(words) for words, _ in dev))
            report(name, alpha, f"cutoff_{cutoff}", counts)
            if cutoff == 0:
                _, counts = score_by_frequency(dev, frequencies, predict_hapax(model, dev))
                report(name, alpha, "keep_words_hapax_unk", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, help="Optional aggregate CSV (default: stdout)")
    args = parser.parse_args()
    if args.output is None:
        diagnose(args.data, sys.stdout)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            diagnose(args.data, stream)
