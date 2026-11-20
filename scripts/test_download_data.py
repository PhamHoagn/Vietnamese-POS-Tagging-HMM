"""Kiểm thử tải bằng byte giả lập, không truy cập corpus thật."""

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

from scripts.download_data import download, main, verify


class DownloadTests(unittest.TestCase):
    """Kiểm tra tính toàn vẹn và xử lý tải lỗi."""

    def setUp(self) -> None:
        """Tạo vùng dữ liệu tạm độc lập."""
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "raw"
        self.ref = "a" * 40  # SHA giả chỉ dành cho mock, không phải commit corpus.
        self.names = {s: f"toy-{s}.conllu" for s in ("train", "dev", "test")}

    @staticmethod
    def response() -> io.BytesIO:
        """Tạo phản hồi HTTP giả chứa byte toy."""
        response = io.BytesIO(b"# toy bytes\n")
        response.headers = {"Content-Length": "12"}
        return response

    def fetch(self) -> None:
        """Tải ba file bằng phản hồi giả."""
        with patch("scripts.download_data.urlopen", side_effect=lambda *a, **k: self.response()):
            download(self.ref, self.names, self.output)

    def test_download_and_verify(self) -> None:
        """Manifest có đủ trường và xác minh không cần mạng."""
        self.fetch()
        manifest = json.loads((self.output / "MANIFEST.json").read_text())
        self.assertEqual(len(manifest["files"]), 3)
        for record in manifest["files"]:
            self.assertEqual(record["commit"], self.ref)
            self.assertEqual(record["size_bytes"], 12)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertTrue(record["downloaded_at"].endswith("+00:00"))
        with patch("scripts.download_data.urlopen", side_effect=AssertionError("network")):
            verify(self.output)

    def test_modified_same_size(self) -> None:
        """Phát hiện thay đổi ngay cả khi số byte giữ nguyên."""
        self.fetch()
        (self.output / self.names["train"]).write_bytes(b"X" * 12)
        self.assertEqual(main(["--verify", "--output-dir", str(self.output)]), 1)

    def test_missing_file(self) -> None:
        """File thiếu làm xác minh thất bại."""
        self.fetch()
        (self.output / self.names["dev"]).unlink()
        with self.assertRaises(FileNotFoundError):
            verify(self.output)

    def test_failed_download_leaves_no_dataset(self) -> None:
        """Lỗi mạng giữa chừng không công bố dữ liệu một phần."""
        with patch("scripts.download_data.urlopen", side_effect=[self.response(), URLError("toy")]):
            with self.assertRaises(URLError):
                download(self.ref, self.names, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_no_overwrite(self) -> None:
        """Từ chối ghi đè bộ dữ liệu đã có."""
        self.fetch()
        with self.assertRaises(ValueError):
            download(self.ref, self.names, self.output)
        verify(self.output)

    def test_invalid_inputs(self) -> None:
        """Không chấp nhận branch, đường dẫn thoát hoặc file trùng."""
        for ref, names in (
            ("master", self.names),
            (self.ref, {**self.names, "train": "../train.conllu"}),
            (self.ref, dict.fromkeys(self.names, "same.conllu")),
        ):
            with self.subTest(ref=ref, names=names), self.assertRaises(ValueError):
                download(ref, names, self.output)

    def test_cli_requires_explicit_inputs(self) -> None:
        """Thiếu ref/tên file trả lỗi tham số."""
        with self.assertRaises(SystemExit) as result:
            main([])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
