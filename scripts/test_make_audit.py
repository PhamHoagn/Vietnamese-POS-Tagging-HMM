"""Kiểm thử audit bằng metadata và số liệu toy, không đọc corpus thật."""

import json
from pathlib import Path

import pytest

from scripts.make_audit import LICENSE, MISSING, make_audit


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Tạo trường hợp khớp, thiếu và thừa số câu để kiểm chứng dấu chênh lệch."""
    manifest, eda = tmp_path / 'MANIFEST.json', tmp_path / 'eda.csv'
    manifest.write_text(json.dumps({'repository': 'https://example.invalid/toy', 'files': [
        {'split': s, 'filename': s + '.conllu', 'url': 'https://example.invalid/' + s,
         'commit': 'a' * 40, 'sha256': 'b' * 64}
        for s in ('train', 'dev', 'test')]}), encoding='utf-8')
    eda.write_text('split,sentence_count,token_count\ntrain,1400,4000\ndev,1122,3000\ntest,802,2000\n', encoding='utf-8')
    return manifest, eda


def test_report_values_and_missing_metadata(tmp_path: Path) -> None:
    """Số liệu được giữ nguyên; thiếu metadata không được suy ra từ URL/ref."""
    manifest, eda = inputs(tmp_path)
    before = manifest.read_bytes(), eda.read_bytes()
    report = make_audit(manifest, eda, tmp_path / 'audit.md').read_text(encoding='utf-8')
    assert '| train | 1400 | 4000 | 1400 | +0 | KHỚP |' in report
    assert '| dev | 1122 | 3000 | 1123 | -1 | **CHÊNH LỆCH** |' in report
    assert '| test | 802 | 2000 | 800 | +2 | **CHÊNH LỆCH** |' in report
    assert LICENSE in report and MISSING in report
    assert 'a' * 40 in report and 'b' * 64 in report
    assert 'https://example.invalid/train' in report
    assert 'CC BY' not in report
    assert before == (manifest.read_bytes(), eda.read_bytes())


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'negative', 'fraction'])
def test_bad_eda_rejected(tmp_path: Path, problem: str) -> None:
    """EDA không đầy đủ hoặc không hợp lệ phải báo lỗi, không tạo số liệu thay thế."""
    manifest, eda = inputs(tmp_path)
    text = eda.read_text(encoding='utf-8')
    if problem == 'duplicate':
        text += 'train,1,2\n'
    elif problem == 'missing':
        text = text.replace('test,802,2000\n', '')
    elif problem == 'negative':
        text = text.replace('802', '-1')
    else:
        text = text.replace('802', '802.5')
    eda.write_text(text, encoding='utf-8')
    output = tmp_path / 'audit.md'
    with pytest.raises(ValueError):
        make_audit(manifest, eda, output)
    assert not output.exists()
