"""Đọc FORM/UPOS từ CoNLL-U, giữ nguyên ranh giới token."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

from .contracts import Corpus, UPOS_TAGS


@dataclass
class Sentence:
    """Một câu với ID gốc (nếu có), token và nhãn tương ứng."""

    sent_id: str | None
    tokens: list[str]
    tags: list[str]

    def __post_init__(self) -> None:
        """Kiểm tra căn chỉnh token và nhãn."""
        assert len(self.tokens) == len(self.tags), "Số token và nhãn phải bằng nhau."


def read_conllu(path: str | Path, nfc: bool = False) -> list[Sentence]:
    """Đọc câu; tùy chọn chuẩn hóa NFC riêng FORM, không học từ corpus."""
    corpus: list[Sentence] = []
    tokens: list[str] = []
    tags: list[str] = []
    sent_id: str | None = None
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            row = line.rstrip("\r\n")
            if not row.strip():
                if tokens:
                    corpus.append(Sentence(sent_id, tokens, tags))
                tokens, tags, sent_id = [], [], None
                continue
            if row.startswith("#"):
                match = re.fullmatch(r"#\s*sent_id\s*=\s*(.*)", row)
                if match:
                    sent_id = match.group(1).strip() or None
                continue
            fields = row.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue
            if len(fields) != 10:
                raise ValueError(f"{path}:{line_number}: cần 10 cột phân cách bằng tab.")
            token_id, form, _, upos, *_ = fields
            if not re.fullmatch(r"[1-9]\d*", token_id):
                raise ValueError(f"{path}:{line_number}: ID token không hợp lệ {token_id!r}.")
            if not form:
                raise ValueError(f"{path}:{line_number}: FORM không được rỗng.")
            if upos not in UPOS_TAGS:
                raise ValueError(
                    f"{path}:{line_number}: UPOS {upos!r} không hợp lệ; "
                    "phải thuộc 17 nhãn UPOS, không được là '_'."
                )
            tokens.append(unicodedata.normalize("NFC", form) if nfc else form)
            tags.append(upos)
    if tokens:
        corpus.append(Sentence(sent_id, tokens, tags))
    return corpus


def load_conllu(path: str | Path) -> Corpus:
    """Giữ API tuple cũ để các module hiện có tiếp tục hoạt động."""
    return [(sentence.tokens, sentence.tags) for sentence in read_conllu(path)]


def load_splits(data_dir: str | Path) -> dict[str, list[Sentence]]:
    """Đọc cả ba split theo MANIFEST.json; chỉ gọi sau khi khóa test."""
    root = Path(data_dir)
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError(f"{manifest_path}: cần đúng ba bản ghi train/dev/test.")
    paths: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{manifest_path}: bản ghi không hợp lệ.")
        split, name = record.get("split"), record.get("filename")
        if not isinstance(split, str) or split not in ("train", "dev", "test") or split in paths:
            raise ValueError(f"{manifest_path}: split thiếu, trùng hoặc không hợp lệ.")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.conllu", name):
            raise ValueError(f"{manifest_path}: tên file không hợp lệ {name!r}.")
        path = root / name
        if path.resolve().parent != root.resolve():
            raise ValueError(f"{manifest_path}: file nằm ngoài thư mục dữ liệu.")
        paths[split] = path
    if len(set(paths.values())) != 3:
        raise ValueError(f"{manifest_path}: ba split phải dùng ba file khác nhau.")
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu file split: {path}")
    return {split: read_conllu(paths[split]) for split in ("train", "dev", "test")}
