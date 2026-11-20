"""Tải corpus theo commit cố định và kiểm tra SHA-256 ngoại tuyến."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import urlopen
from urllib.error import URLError

REPOSITORY = "https://github.com/UniversalDependencies/UD_Vietnamese-VTB"
RAW_ROOT = "https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB"
GUIDE = (
    f"Kiểm tra tại {REPOSITORY}: mở Commits (hoặc commit của release), "
    "sao chép SHA đầy đủ 40 ký tự vào --ref. Tại đúng commit đó, "
    "kiểm tra tên ba file CoNLL-U và truyền --train-file, --dev-file, --test-file. "
    "Không dùng tên branch hoặc SHA rút gọn."
)
SPLITS = ("train", "dev", "test")


def file_name(value: str) -> str:
    """Chỉ chấp nhận tên file CoNLL-U ở thư mục gốc corpus."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.conllu", value):
        raise ValueError(f"Tên file CoNLL-U không hợp lệ: {value!r}")
    return value


def sha256(path: Path) -> str:
    """Tính hash theo từng khối byte, không đọc nhãn dữ liệu."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(ref: str, names: dict[str, str], output: Path) -> None:
    """Tải đủ ba file vào vùng tạm rồi ghi manifest cuối cùng."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        raise ValueError("--ref phải là SHA commit đầy đủ 40 ký tự. " + GUIDE)
    ref = ref.lower()
    names = {split: file_name(names[split]) for split in SPLITS}
    if len(set(names.values())) != 3:
        raise ValueError("Ba split phải dùng ba tên file khác nhau.")
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in names.values()] + [output / "MANIFEST.json"]
    if any(path.exists() for path in targets):
        raise ValueError("Đích đã có file dữ liệu/manifest; dùng --verify hoặc --output-dir khác.")
    records: list[dict[str, object]] = []
    with TemporaryDirectory(prefix=".download-", dir=output) as temporary:
        stage = Path(temporary)
        for split, name in names.items():
            url = f"{RAW_ROOT}/{ref}/{quote(name, safe='')}"
            path = stage / name
            with urlopen(url, timeout=60) as response, path.open("wb") as stream:
                expected_size = response.headers.get("Content-Length")
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            size = path.stat().st_size
            if size == 0 or (expected_size is not None and size != int(expected_size)):
                raise ValueError(f"File rỗng hoặc tải thiếu byte: {name}")
            records.append({
                "split": split, "filename": name, "url": url,
                "ref": ref, "commit": ref, "size_bytes": size,
                "sha256": sha256(path),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            })
        manifest = {"schema_version": 1, "repository": REPOSITORY, "files": records}
        (stage / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for target in targets:
            (stage / target.name).replace(target)
    print(f"Đã tải 3 file và ghi {output / 'MANIFEST.json'}")


def verify(output: Path) -> None:
    """Đối chiếu đủ ba file với hash và kích thước trong manifest."""
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Manifest không hợp lệ hoặc phiên bản không hỗ trợ.")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("Manifest phải có đúng ba split train/dev/test.")
    splits: set[str] = set()
    names: set[str] = set()
    commits: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Bản ghi manifest không hợp lệ.")
        split = record.get("split")
        if split not in SPLITS or split in splits:
            raise ValueError("Split thiếu, trùng hoặc không hợp lệ.")
        splits.add(split)
        name = file_name(record["filename"])
        if name in names:
            raise ValueError("Tên file bị trùng trong manifest.")
        names.add(name)
        commit = record["commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or record["ref"] != commit:
            raise ValueError("Commit/ref trong manifest không hợp lệ.")
        commits.add(commit)
        if record["url"] != f"{RAW_ROOT}/{commit}/{quote(name, safe='')}":
            raise ValueError(f"URL không khớp commit/tên file: {name}")
        path = output / name
        if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Sai kích thước hoặc SHA-256: {name}")
    if len(commits) != 1:
        raise ValueError("Ba split không cùng commit.")
    print("VERIFY OK: đủ 3 file, kích thước và SHA-256 khớp manifest.")


def main(argv: list[str] | None = None) -> int:
    """Nhận tham số CLI; trả mã khác 0 khi tải hoặc xác minh lỗi."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=GUIDE)
    parser.add_argument("--ref", help="SHA commit đầy đủ (bắt buộc khi tải)")
    for split in SPLITS:
        parser.add_argument(f"--{split}-file", help=f"Tên file {split} đã kiểm tra trên GitHub")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--verify", action="store_true", help="Kiểm tra manifest ngoại tuyến")
    args = parser.parse_args(argv)
    if args.verify:
        if args.ref or any(getattr(args, f"{split}_file") for split in SPLITS):
            parser.error("--verify chỉ dùng manifest; không truyền --ref hoặc tên file.")
    elif not args.ref or any(not getattr(args, f"{split}_file") for split in SPLITS):
        parser.error("Khi tải cần --ref, --train-file, --dev-file, --test-file. " + GUIDE)
    try:
        if args.verify:
            verify(args.output_dir)
        else:
            download(args.ref, {s: getattr(args, f"{s}_file") for s in SPLITS}, args.output_dir)
    except (OSError, URLError, ValueError, KeyError, TypeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
