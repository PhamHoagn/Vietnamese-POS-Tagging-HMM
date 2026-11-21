"""Phân tích lỗi từ CSV dự đoán đã lưu, không fit model hoặc đọc corpus."""

import argparse
from collections import Counter
import csv
from pathlib import Path

from .contracts import UPOS_TAGS

FIELDS = ['split', 'sent_id', 'idx', 'token', 'gold', 'pred', 'is_oov']


def export_errors(predictions_path: str | Path, output_dir: str | Path = 'results/tables/errors',
                  *, split: str = 'dev', limit: int = 10) -> dict[str, Path]:
    """Xuất toàn bộ lỗi, ví dụ đầu tiên và cặp nhầm; giữ nguồn dòng CSV để truy vết."""
    if split not in ('train', 'dev', 'test') or limit < 1:
        raise ValueError('split không hợp lệ hoặc limit phải >= 1.')
    source, output = Path(predictions_path), Path(output_dir)
    errors: list[dict[str, str | int]] = []
    pairs: Counter[tuple[str, str]] = Counter()
    total = 0
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('CSV thiếu cột dự đoán bắt buộc.')
        for row_number, row in enumerate(reader, start=2):
            if any(row.get(field) is None for field in FIELDS) or None in row:
                raise ValueError(f'Dòng {row_number}: số cột không hợp lệ.')
            if row['split'] != split:
                raise ValueError(f'Dòng {row_number}: split không khớp {split}.')
            if row['gold'] not in UPOS_TAGS or row['pred'] not in UPOS_TAGS:
                raise ValueError(f'Dòng {row_number}: nhãn ngoài UPOS.')
            if not row['idx'].isascii() or not row['idx'].isdigit() or not row['token']:
                raise ValueError(f'Dòng {row_number}: idx/token không hợp lệ.')
            if row['is_oov'] not in ('True', 'False'):
                raise ValueError(f'Dòng {row_number}: is_oov phải là True/False.')
            total += 1
            if row['gold'] != row['pred']:
                errors.append({**{key: row[key] for key in FIELDS},
                               'source_file': source.as_posix(), 'source_row': row_number,
                               'explanation': ''})
                pairs[row['gold'], row['pred']] += 1
    paths = {name: output / f'{split}_{name}.csv' for name in ('errors', 'examples', 'error_pairs')}
    if any(path.resolve() == source.resolve() for path in paths.values()):
        raise ValueError('Không ghi đè CSV nguồn.')
    output.mkdir(parents=True, exist_ok=True)
    error_fields = [*FIELDS, 'source_file', 'source_row', 'explanation']
    pair_rows = [{'gold': gold, 'pred': pred, 'count': count}
                 for (gold, pred), count in sorted(pairs.items(), key=lambda item: (-item[1], item[0]))]
    for name, rows, fields in [('errors', errors, error_fields),
                               ('examples', errors[:limit], error_fields),
                               ('error_pairs', pair_rows, ['gold', 'pred', 'count'])]:
        with paths[name].open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f'{split}: {total} token, {len(errors)} lỗi; xuất {min(limit, len(errors))} ví dụ theo thứ tự CSV.')
    return paths


def main() -> None:
    """Nhận file dự đoán, mặc định phân tích dev; test chỉ dùng sau khóa cấu hình."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('results/tables/errors'))
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    try:
        export_errors(args.predictions, args.output_dir, split=args.split, limit=args.limit)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Lỗi: {exc}\n')


if __name__ == '__main__':
    main()
