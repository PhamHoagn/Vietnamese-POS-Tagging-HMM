"""Read token-aligned FORM/UPOS sentences from a CoNLL-U file."""

from pathlib import Path
import re

from .contracts import Corpus, UPOS_TAGS


def load_conllu(path: str | Path) -> Corpus:
    corpus: Corpus = []
    tokens: list[str] = []
    tags: list[str] = []
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            row = line.rstrip("\r\n")
            if not row.strip():
                if tokens:
                    corpus.append((tokens, tags))
                    tokens, tags = [], []
                continue
            if row.startswith("#"):
                continue
            fields = row.split("\t")
            if len(fields) != 10:
                raise ValueError(f"{path}:{line_number}: expected 10 tab-separated columns.")
            token_id, form, _, upos, *_ = fields
            if re.fullmatch(r"[1-9]\d*[-.][1-9]\d*", token_id):
                continue  # Multiword-token ranges and empty nodes are not tokens.
            if not re.fullmatch(r"[1-9]\d*", token_id):
                raise ValueError(f"{path}:{line_number}: invalid token ID {token_id!r}.")
            if not form or upos not in UPOS_TAGS:
                raise ValueError(f"{path}:{line_number}: require non-empty FORM and valid UPOS.")
            tokens.append(form)  # Preserve internal spaces; never split FORM again.
            tags.append(upos)
    if tokens:
        corpus.append((tokens, tags))
    return corpus
