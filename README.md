# Vietnamese-POS-Tagging-HMM

Khung dự án gán nhãn UPOS tiếng Việt bằng baseline Most Frequent Tag và HMM
Bigram/Trigram. Hiện parser, baseline, Token Accuracy và Macro-F1 đã có code;
HMM, các metric còn lại và grid search vẫn là placeholder.

Python **3.11 trở lên**, CPU; không cần GPU hoặc dataset thật để kiểm tra kết nối.
Chạy các lệnh từ thư mục gốc repository.

## Chạy trên Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke_test
```

## Chạy trên Google Colab CPU

Sau khi các thay đổi đã được nhóm review và đưa lên remote, clone branch chứa
khung dự án. Không chạy `git clone` lại nếu đã có checkout trong notebook.

```python
!git clone --branch main https://github.com/PhamHoagn/Vietnamese-POS-Tagging-HMM.git
%cd Vietnamese-POS-Tagging-HMM
!python -m pip install -r requirements.txt
!python -m pytest -q
!python -m scripts.smoke_test
```

Khi chưa push, có thể tải bản ZIP của checkout local lên Colab, giải nén và `%cd`
vào thư mục chứa `requirements.txt`, sau đó chạy ba lệnh cuối. Notebook import
bằng `from src...`; không thêm đường dẫn máy cá nhân vào `sys.path` và không chạy
trực tiếp `python src/<module>.py`.

Smoke test chỉ dùng fixture tự tạo trong `tests/fixtures/tiny.conllu` với toy
train/dev; số in ra dùng để kiểm tra kết nối, không phải kết quả thực nghiệm.

- [API Contract và phân công](docs/PROJECT_CONTRACT.md)
- [Bản nháp đăng ký trước đánh giá test](docs/pre_register_test.md)

Không đưa corpus gốc vào Git trước khi xác nhận giấy phép. Thư mục `data/raw/`
được ignore, ngoại trừ `.gitkeep`.

## Đọc CoNLL-U (TV1)

```python
from src.data_loader import read_conllu, load_splits

train = read_conllu("data/raw/TEN_FILE_TRAIN.conllu", nfc=False)
# train[0].sent_id, train[0].tokens, train[0].tags
# Chỉ sau khi khóa cấu hình test:
# splits = load_splits("data/raw")
```

`load_splits` đọc tên ba file từ `MANIFEST.json` do script tải dữ liệu tạo,
trả các khóa `train`, `dev`, `test`. Hàm này không xác minh hash; chạy
`python scripts/download_data.py --verify` trước khi sử dụng dữ liệu.
Trước khi khóa cấu hình, chỉ gọi `read_conllu` riêng cho train/dev.
`nfc=True` chuẩn hóa Unicode NFC của FORM; mặc định giữ nguyên văn bản.
ID câu lấy từ comment `# sent_id = ...`, hoặc `None` nếu không có.
`load_conllu` vẫn trả tuple `(tokens, tags)` tương thích với API hiện có;
`Sentence` dataclass mới nằm trong `src.data_loader`, không thay kiểu tuple
trong `src.contracts`.

Không dùng `split()` không tham số vì nó tách cả khoảng trắng trong FORM
(ví dụ `Hà Nội`), làm lệch token và nhãn. CoNLL-U có 10 cột phân cách bằng tab,
FORM ở cột 2 và UPOS ở cột 4; hàm `read_conll_2columns()` dành cho dữ liệu
hai cột không thể dùng trực tiếp. Cần bỏ comment, multi-word token và empty node.

Kiểm thử bằng dữ liệu toy (không mở test corpus):

```bash
python -m pytest tests/test_data_loader.py tests/test_baseline.py tests/test_contract.py -q
```

## Thống kê EDA (TV1)

Sau khi đã khóa cấu hình và được phép đọc test, truyền ba split đã parse:

```python
from src.eda import export_eda

paths = export_eda(splits)  # dict train/dev/test, mỗi giá trị là list[Sentence]
```

Hàm ghi `results/tables/eda.csv` và `results/tables/upos_distribution.csv`.
Có thể truyền `output_dir` để đổi thư mục xuất. Mỗi lần chạy ghi lại hai bảng.
EDA tính số câu/token, min/mean/median/max độ dài (bao gồm PUNCT), số FORM
phân biệt chính xác theo chuỗi đầu vào và số token có gold tag PUNCT.
Phân bố nhãn có đủ 17 UPOS theo thứ tự cố định cho từng split;
`percentage = 100 * count / token_count` của chính split đó, nhãn vắng có count 0.
Split không có câu để trống các thống kê độ dài; split không có token để trống
percentage vì không có mẫu số. Số đếm vẫn bằng 0. Không học vocab dùng cho mô hình,
không thay đổi đầu vào, không tự chuẩn hóa và không dùng thống kê test để tuning.

Kiểm thử chỉ dùng dữ liệu toy và xuất CSV vào thư mục tạm:

```bash
python -m pytest scripts/test_eda.py -q
```

### OOV và rare-token

`export_eda(splits)` xuất thêm `results/tables/oov_stats.csv` cho dev và test.
Mỗi split có dòng `upos=ALL` và 17 dòng UPOS đúng thứ tự. `token_count` là
mẫu số của chính dòng đó (toàn split hoặc các token có gold UPOS tương ứng).
Các cột `*_rate` là tỷ lệ 0–1; nhóm không có token để trống rate.

- `oov_count/rate`: FORM chưa xuất hiện trong train nguyên gốc, tức `f_train=0`.
- `f_train_le_1_count/rate`, `f_train_le_2_count/rate`: token có tần suất train
  ≤1 hoặc ≤2, **bao gồm OOV**, vì tần suất của OOV bằng 0.
- `seen_rare_le_1_count/rate`, `seen_rare_le_2_count/rate`: token đã thấy trong
  train với `1 ≤ f_train ≤ k`. **Rare-token KHÔNG phải OOV**; hai nhóm này tách biệt.

Tần suất đếm theo FORM chính xác (phân biệt hoa thường), chỉ lấy từ train đã parse,
trước mọi cutoff/UNK mapping; không cập nhật bằng dev/test. Với mỗi ngưỡng k:
`f_train_le_k_count = oov_count + seen_rare_le_k_count`.

### Audit câu trùng, câu rỗng và nhãn lạ

```python
from src.eda import audit_splits

path = audit_splits(splits)  # ghi results/tables/duplicates.csv và in tóm tắt
```

Hàm chỉ báo cáo trên ba split đã parse, không sửa/xóa câu, nhãn hay đổi split.
Gọi riêng trước `export_eda` nếu cần báo nhãn lạ (EDA từ chối nhãn không hợp lệ).
So trùng bằng tuple token chính xác, không ghép bằng dấu cách, không chuẩn hóa,
không xét sent_id hoặc tags. CSV ghi từng cặp vị trí trùng nội bộ/liên split
(train–dev, train–test, dev–test), từng câu rỗng và từng vị trí nhãn lạ.
Chỉ số câu/token bắt đầu từ 0; `tokens_json` giữ ranh giới token.
Tóm tắt đếm **cặp câu** (n bản giống nhau tạo n*(n-1)/2 cặp nội bộ), không phải
số nhóm trùng. Hai câu rỗng cũng khớp chuỗi token rỗng và được báo trùng.
Không có lỗi thì CSV chỉ có header.

Audit chỉ thấy đối tượng đầu vào: parser hiện bỏ câu rỗng và từ chối nhãn lạ,
nên không thể phục hồi các lỗi đã bị loại trước đó. Test kiểm tra hai trường hợp
này bằng Sentence toy. Chỉ chạy audit có test thật sau khi khóa cấu hình;
không dùng báo cáo để tự ý thay split hoặc điều chỉnh mô hình theo test.

### Vẽ biểu đồ từ CSV

Cài dependencies và chạy từ repo root:

```bash
python -m pip install -r requirements.txt
python scripts/make_figures.py
# Hoặc: python -m scripts.make_figures --tables-dir results/tables --output-dir results/figures
```

Chạy lại `export_eda(splits)` khi được phép dùng test thật để có thêm
`sentence_lengths.csv` (`split,length,count`). Min/mean/median/max trong `eda.csv`
không đủ để dựng histogram. Bảng mới lưu tần số câu theo từng độ dài, không lưu nội dung câu.
Script vẽ chỉ đọc CSV, không parse corpus hay tính lại OOV; không cần seaborn.

Ba PNG riêng, 200 dpi, nhãn tiếng Việt:
- `sentence_lengths.png`: histogram có khoảng rộng 1 token, ba split chồng màu.
- `upos_distribution.png`: phần trăm UPOS theo train/dev/test từ `percentage`.
- `oov_by_upos.png`: OOV theo UPOS của dev/test, đổi `oov_rate` từ 0–1 sang %.

Ô tỷ lệ trống hiện N/A, không thay bằng 0. Test toy xuất ảnh vào thư mục tạm:
`python -m pytest scripts/test_eda.py scripts/test_make_figures.py -q`.

### Sinh báo cáo audit từ metadata và EDA

```bash
python scripts/make_audit.py
# Tùy chọn: --manifest data/raw/MANIFEST.json --eda results/tables/eda.csv --output data_audit.md
python -m pytest scripts/test_make_audit.py -q
```

Báo cáo chép URL/ref/commit/SHA-256 và metadata sẵn có từ manifest, số câu/token
đã đo từ EDA; so số câu với mốc kế hoạch train/dev/test = 1400/1123/800.
Chênh lệch có dấu và nhãn CHÊNH LỆCH, không tự sửa split. Thiếu metadata ghi
`[KHÔNG CÓ TRONG DỮ LIỆU]`; giấy phép luôn để
`[ĐIỀN THỦ CÔNG sau khi đọc README repo]` để nhóm xác nhận nguồn.
Script không xác minh hash hay chứng minh hai file đầu vào cùng phiên bản corpus.
Không có đầu vào thật thì không sinh báo cáo thật. Chạy lại sẽ ghi đè báo cáo cũ,
nên giữ riêng phần bổ sung thủ công trước khi tái sinh.
