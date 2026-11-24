"""Show gold, baseline and HMM tags on the repository's synthetic fixture."""

from pathlib import Path
import sys

from src.baseline import MostFrequentTagger
from src.contracts import Corpus
from src.data_loader import load_conllu
from src.hmm_bigram import BigramHMMTagger
from src.hmm_common import HMMTrainingCache
from src.hmm_trigram import TrigramHMMTagger


def load_demo_corpus() -> Corpus:
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny.conllu"
    return load_conllu(fixture)


def fit_models(train: Corpus, *, bigram_params: dict, trigram_params: dict) -> tuple[
    MostFrequentTagger, BigramHMMTagger, TrigramHMMTagger
]:
    cache = HMMTrainingCache()
    baseline = MostFrequentTagger().fit(train)
    bigram = BigramHMMTagger(**bigram_params, cache=cache).fit(train)
    trigram = TrigramHMMTagger(**trigram_params, cache=cache).fit(train)
    return baseline, bigram, trigram


def main() -> None:
    *train, (tokens, gold) = load_demo_corpus()
    params = {"alpha": 0.1, "cutoff": 1}
    baseline, bigram, trigram = fit_models(train, bigram_params=params, trigram_params=params)
    baseline_tags = baseline.predict(tokens)
    bigram_tags = bigram.predict(tokens)
    trigram_tags = trigram.predict(tokens)

    print("Ví dụ minh họa từ fixture tự tạo. Bigram/Trigram: alpha=0.1, cutoff=1.")
    print(f"{'Token':<12}{'Gold':<9}{'B0':<9}{'Bigram':<9}{'Trigram':<9}{'OOV'}")
    for token, expected, b0, h1, h2 in zip(tokens, gold, baseline_tags, bigram_tags, trigram_tags, strict=True):
        oov = "Có" if token not in baseline.train_vocab_ else "Không"
        print(f"{token:<12}{expected:<9}{b0:<9}{h1:<9}{h2:<9}{oov}")


if __name__ == "__main__":
    # Windows pipes may default to a legacy code page that cannot encode Vietnamese.
    sys.stdout.reconfigure(encoding="utf-8")
    main()
