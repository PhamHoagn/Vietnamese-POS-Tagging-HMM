"""Baseline nhãn phổ biến nhất, chỉ học từ các câu train được truyền vào."""

from collections import Counter, defaultdict
from collections.abc import Sequence, Set
import csv
from pathlib import Path
from typing import Self

from .contracts import Corpus, UPOS_TAGS, validate_corpus, validate_tokens
from .data_loader import Sentence


_TAG_ORDER = {tag: index for index, tag in enumerate(UPOS_TAGS)}


def _most_frequent(counts: Counter[str]) -> str:
    """Chọn count lớn nhất; hòa theo thứ tự 17 UPOS cố định."""
    return min(counts, key=lambda tag: (-counts[tag], _TAG_ORDER[tag]))


class MostFrequentTag:
    """Ghi nhớ nhãn theo FORM; từ mới dùng nhãn phổ biến nhất train."""

    def fit(self, train_sentences: Sequence[Sentence]) -> Self:
        """Học chỉ từ train; fit lại thay toàn bộ trạng thái khi thành công."""
        corpus: Corpus = []
        for sentence in train_sentences:
            if not isinstance(sentence, Sentence):
                raise TypeError("train_sentences phải chứa Sentence từ read_conllu().")
            corpus.append((sentence.tokens, sentence.tags))
        validate_corpus(corpus)
        word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        tag_counts: Counter[str] = Counter()
        for tokens, tags in corpus:
            for token, tag in zip(tokens, tags):
                word_counts[token][tag] += 1
                tag_counts[tag] += 1
        if not tag_counts:
            raise ValueError("Train phải có ít nhất một token.")
        self.word_tags_ = {word: _most_frequent(counts) for word, counts in word_counts.items()}
        self.default_tag_ = _most_frequent(tag_counts)
        self.train_vocab_ = frozenset(word_counts)
        return self

    def predict(self, tokens: list[str]) -> list[str]:
        """Dự đoán không cập nhật counts, vocabulary hoặc trạng thái mô hình."""
        if not hasattr(self, "word_tags_"):
            raise RuntimeError("MostFrequentTag phải fit trước predict.")
        validate_tokens(tokens)
        return [self.word_tags_.get(token, self.default_tag_) for token in tokens]


class MostFrequentTagger(MostFrequentTag):
    """Adapter tương thích API tuple hiện có của dự án."""

    def fit(self, train_data: Corpus) -> Self:
        """Chuyển corpus tuple cũ thành Sentence trước khi học."""
        validate_corpus(train_data)
        super().fit([Sentence(None, tokens, tags) for tokens, tags in train_data])
        return self


def export_predictions(
    model: MostFrequentTag,
    sentences: Sequence[Sentence],
    vocab_train: Set[str],
    path: str | Path,
    *,
    split: str = "",
) -> Path:
    """Xuất dự đoán; split không truyền để trống, idx bắt đầu từ 0."""
    if split not in ("", "train", "dev", "test"):
        raise ValueError("split phải là train/dev/test hoặc để trống khi chưa xác định.")
    model.predict([])  # Kiểm tra đã fit kể cả khi không có câu cần xuất.
    if not isinstance(vocab_train, Set) or vocab_train != model.train_vocab_:
        raise ValueError("vocab_train phải khớp vocabulary nguyên gốc của train đã fit.")
    rows: list[dict[str, object]] = []
    for sentence in sentences:
        if not isinstance(sentence, Sentence):
            raise TypeError("sentences phải chứa Sentence từ read_conllu().")
        validate_corpus([(sentence.tokens, sentence.tags)])
        predictions = model.predict(sentence.tokens)
        if len(predictions) != len(sentence.tokens):
            raise ValueError("Số dự đoán không khớp số token.")
        for index, (token, gold, pred) in enumerate(zip(sentence.tokens, sentence.tags, predictions)):
            rows.append({"split": split, "sent_id": sentence.sent_id, "idx": index,
                         "token": token, "gold": gold, "pred": pred,
                         "is_oov": token not in vocab_train})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "sent_id", "idx", "token", "gold", "pred", "is_oov"])
        writer.writeheader()
        writer.writerows(rows)
    return output
