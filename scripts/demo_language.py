"""Word segmentation and a contextual Viterbi decoder used by the web demo."""

from pathlib import Path
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.contracts import Corpus
from src.hmm_trigram import TrigramHMMTagger


def demo_training() -> Corpus:
    corpus = []
    for line in Path(__file__).with_name("demo_training.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            pairs = [item.strip().rsplit("/", 1) for item in line.split("|")]
            words, tags = zip(*pairs, strict=True)
            corpus.append((list(words), list(tags)))
    return corpus


class WordSegmenter:
    """Longest matching word from training, preserving the user's letter case."""

    pieces = re.compile(r"\d+(?:[.,]\d+)+|\w+|[^\w\s]", re.UNICODE)

    def __init__(self, corpus: Corpus):
        self.trie = {}
        for words, _ in corpus:
            for word in words:
                parts = word.casefold().split()
                if len(parts) < 2 or not all(part.isalnum() for part in parts):
                    continue
                node = self.trie
                for part in parts:
                    node = node.setdefault(part, {})
                node[None] = True

    def split(self, text: str) -> list[str]:
        pieces = self.pieces.findall(unicodedata.normalize("NFC", text))
        tokens = []
        i = 0
        while i < len(pieces):
            node, end = self.trie, i + 1
            for j in range(i, len(pieces)):
                node = node.get(pieces[j].casefold())
                if node is None:
                    break
                if None in node:
                    end = j + 1
            tokens.append(" ".join(pieces[i:end]))
            i = end
        return tokens


def token_features(tokens: list[str]) -> list[dict]:
    forms = [token.casefold() for token in tokens]
    features = []
    for i, (token, form) in enumerate(zip(tokens, forms, strict=True)):
        left = forms[i - 1] if i else "<start>"
        right = forms[i + 1] if i + 1 < len(tokens) else "<end>"
        syllables = form.split()
        row = {"word": form, "left": left, "right": right, "capital": token.istitle(),
               "number": any(c.isdecimal() for c in token), "upper": token.isupper(),
               "punct": all(unicodedata.category(c).startswith("P") for c in token),
               "left+word": left + "|" + form, "word+right": form + "|" + right}
        if len(syllables) > 1:
            row.update(first=syllables[0], last=syllables[-1])
        features.append(row)
    return features


class ContextualTrigram(TrigramHMMTagger):
    """Hybrid decoder: local tag posteriors corrected for the tag prior, then Viterbi."""

    def fit(self, train_data: Corpus):
        super().fit(train_data)
        self.vectorizer = DictVectorizer()
        matrix = self.vectorizer.fit_transform(row for words, _ in train_data for row in token_features(words))
        labels = [tag for _, tags in train_data for tag in tags]
        self.classifier = LogisticRegression(C=4, max_iter=250, random_state=42)
        with threadpool_limits(limits=1, user_api="blas"):
            self.classifier.fit(matrix, labels)
        indices = {tag: i for i, tag in enumerate(self.classifier.classes_)}
        self.class_order = [indices[tag] for tag in self.tags]
        self.log_prior = np.log(self.tag_counts_ / len(labels))
        return self

    def predict(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        matrix = self.vectorizer.transform(token_features(tokens))
        local = self.classifier.predict_log_proba(matrix)[:, self.class_order]
        return self._viterbi(local - self.log_prior)
