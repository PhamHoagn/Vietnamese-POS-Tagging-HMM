"""Exercise fixture -> train/dev -> baseline -> metrics, with no real corpus."""

import inspect
import json
from math import isclose
from pathlib import Path

from src.baseline import MostFrequentTagger
from src.contracts import POSTagger, UPOS_TAGS
from src.data_loader import load_conllu
from src.hmm_bigram import BigramHMMTagger
from src.hmm_trigram import TrigramHMMTagger
from src.metrics import evaluate_pos


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny.conllu"
    corpus = load_conllu(fixture)
    require([len(tokens) for tokens, _ in corpus] == [5, 3, 3], "Fixture shape changed.")
    train_data, dev_data = corpus[:2], corpus[2:]
    model = MostFrequentTagger().fit(train_data)
    predictions = [model.predict(tokens) for tokens, _ in dev_data]
    require("học" not in model.train_vocab_, "Dev token leaked into training vocabulary.")
    result = evaluate_pos(dev_data, predictions, model.train_vocab_, UPOS_TAGS)
    require(isclose(result["token_accuracy"], 2 / 3), "Unexpected toy accuracy.")
    require(isclose(result["macro_f1"], 2 / 17), "Unexpected toy macro-F1.")
    require(all(result[key] is None for key in
                ("oov_accuracy", "oov_rate", "accuracy_no_punct")), "Reserved metric misreported.")
    for tagger_class in (BigramHMMTagger, TrigramHMMTagger):
        tagger = tagger_class(alpha=0.1, cutoff=1)
        require(isinstance(tagger, POSTagger), "Missing tagger interface.")
        require(list(inspect.signature(tagger.fit).parameters) == ["train_data"], "Invalid fit API.")
        require(list(inspect.signature(tagger.predict).parameters) == ["tokens"], "Invalid predict API.")
        # Do not call placeholder HMMs to produce predictions.
    print("PASS: synthetic fixture -> train/dev -> baseline -> evaluate_pos")
    print("Toy integration values only; not experimental results:")
    print(json.dumps(result, indent=2))
    print("PASS: Bigram/Trigram imports and interfaces checked; no HMM decoding in this smoke test.")


if __name__ == "__main__":
    main()
