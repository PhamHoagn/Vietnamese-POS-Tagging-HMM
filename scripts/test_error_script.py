"""Kiểm thử phân tích lỗi trên CSV toy tự tạo."""

import csv
from pathlib import Path

import pytest

from src.error_script import export_errors


def test_error_exports(tmp_path: Path) -> None:
    """Hai lỗi thật trong ba hàng tạo hai cặp nhầm; limit chỉ giới hạn ví dụ."""
    source = tmp_path / 'pred.csv'
    source.write_text('split,sent_id,idx,token,gold,pred,is_oov\n'
                      'dev,d,0,sinh viên,NOUN,VERB,True\n'
                      'dev,d,1,học,VERB,VERB,False\n'
                      'dev,d,2,.,PUNCT,NOUN,False\n', encoding='utf-8')
    before = source.read_bytes()
    paths = export_errors(source, tmp_path / 'out', limit=1)
    with paths['errors'].open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]['token'] == 'sinh viên'
    assert [r['source_row'] for r in rows] == ['2', '4']
    assert all(r['explanation'] == '' for r in rows)
    with paths['examples'].open(encoding='utf-8') as stream:
        assert len(list(csv.DictReader(stream))) == 1
    assert source.read_bytes() == before


@pytest.mark.parametrize('row', ['test,d,0,x,NOUN,VERB,True',
                                 'dev,d,0,x,_,VERB,True',
                                 'dev,d,0,x,NOUN,VERB,maybe'])
def test_bad_prediction_csv(tmp_path: Path, row: str) -> None:
    """Sai split/nhãn/OOV bị từ chối trước khi xuất kết quả."""
    source = tmp_path / 'bad.csv'
    source.write_text('split,sent_id,idx,token,gold,pred,is_oov\n' + row, encoding='utf-8')
    with pytest.raises(ValueError):
        export_errors(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_no_errors_writes_header(tmp_path: Path) -> None:
    """Không có lỗi thì không bịa ví dụ cho đủ 5–10 câu."""
    source = tmp_path / 'pred.csv'
    source.write_text('split,sent_id,idx,token,gold,pred,is_oov\ndev,d,0,x,NOUN,NOUN,False\n', encoding='utf-8')
    paths = export_errors(source, tmp_path / 'out')
    for path in paths.values():
        with path.open(encoding='utf-8') as stream:
            assert list(csv.DictReader(stream)) == []
