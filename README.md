# Vietnamese-POS-Tagging-HMM

Dự án gán nhãn UPOS tiếng Việt bằng baseline Most Frequent Tag và HMM Bigram, Trigram.
Pipeline gồm parser CoNLL-U, Viterbi NumPy, năm chỉ số đánh giá, grid search trên
dev và bootstrap theo câu cho đợt đánh giá đã khóa cấu hình.

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

[Mở notebook POS Tagging trên Google Colab](https://colab.research.google.com/github/PhamHoagn/Vietnamese-POS-Tagging-HMM/blob/main/notebooks/NLP_POS.ipynb).
Kết nối với runtime CPU và chạy hai ô mã đầu để mở demo trong notebook. Ô đầu
tự tải mã nguồn và cài thư viện. Ô thứ hai tải train/dev, kiểm tra mã băm và
mở giao diện nhập câu. Chọn **Runtime → Run all** để chạy cả thí nghiệm dev,
xem biểu đồ và các bảng Test đã khóa.

Smoke test chỉ dùng fixture tự tạo trong `tests/fixtures/tiny.conllu` với toy
train/dev; số in ra dùng để kiểm tra kết nối, không phải kết quả thực nghiệm.

- [API Contract và phân công](docs/PROJECT_CONTRACT.md)
- [Audit dữ liệu và phân tích lỗi dev](docs/TV1_DATA.md)
- [Giải tay Viterbi và demo](docs/TV2_GIAI_TAY_VITERBI.md)
- [Grid và kết quả dev](docs/TV3_EVALUATION_RUNBOOK.md)
- [Đăng ký trước đánh giá test](docs/pre_register_test.md)
- [Kết quả Test và ba khoảng tin cậy](docs/TV3_EVALUATION_RUNBOOK.md#đối-chiếu-kết-quả-test)
- [Checklist nghiệm thu và minh chứng](docs/PROJECT_CHECKLIST.md)

Không đưa corpus gốc vào Git trước khi xác nhận giấy phép. Thư mục `data/raw/`
được ignore, ngoại trừ `.gitkeep`.

## Tái lập dữ liệu và thực nghiệm dev

```sh
python -X utf8 -m scripts.prepare_tv1 --download
python -m scripts.evaluate_dev --data data/raw/ud-vtb-fa7130860b65
python -X utf8 -m scripts.prepare_tv1 --dev-predictions .venv/project-runs/dev/predictions_dev.csv
python -m scripts.plot_dev_grid
python -m scripts.demo_tv2
```

Grid so sánh trên dev và ghi bảng vào `results/tables/dev/`. Dự đoán chứa FORM
corpus lưu trong `.venv/project-runs/dev/`. Cách ghi xác nhận của nhóm và chạy
đánh giá cuối nằm trong [hồ sơ đăng ký](docs/pre_register_test.md).

## Demo trên trình duyệt

```sh
python -m scripts.demo_web
```

Mở [demo local](http://127.0.0.1:8765). Chọn **Nhập câu bình thường** để tự
tách từ, hoặc **Nhập token đã tách** và dùng dấu `|` để giữ ranh giới từ,
chẳng hạn `Tôi | đến | Hà Nội | .`. Hai câu mẫu đầu minh họa “bàn” trong ngữ
cảnh danh từ và động từ. Mặc định demo dùng corpus đồ chơi. Để dùng dữ liệu
huấn luyện UD VTB và cấu hình tốt nhất đã khóa của mỗi HMM, chạy thêm
`--train data/raw/ud-vtb-fa7130860b65/vi_vtb-ud-train.conllu`. Thêm
`--dev data/raw/ud-vtb-fa7130860b65/vi_vtb-ud-dev.conllu` để mở hai câu dev
có nhãn chuẩn. Menu **Ablation cùng tham số** dùng `α=0.1, cutoff=1` cho cả
hai HMM gốc. Accuracy trên web được tính theo câu có nhãn chuẩn đang hiển thị.
Cột **Ngữ cảnh** dùng bộ phân loại logistic học từ token và các từ liền kề.
Viterbi giải mã bằng chuyển tiếp Trigram và điểm log xác suất nhãn đã hiệu
chỉnh theo tần suất nhãn. Riêng bản demo này học thêm 26 câu tự soạn trong
[dữ liệu bổ sung](scripts/demo_training.txt). Cột B0, Bigram và Trigram dùng
tập train gốc. Bộ tách từ tự động ghép theo từ dài nhất đã gặp trong tập train
của demo và có thể tách sai cụm lạ. Khi biết ranh giới từ, chọn chế độ thủ công.
Xem [bảng dev](results/tables/dev/metrics_dev.csv) cho kết quả toàn split.

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

### Baseline MostFrequentTag và xuất dự đoán

```python
from src.baseline import MostFrequentTag, export_predictions

model = MostFrequentTag().fit(train)  # list[Sentence], chỉ dữ liệu train
pred = model.predict(["sinh viên", "học"])
export_predictions(model, dev, model.train_vocab_, "results/tables/baseline_dev.csv", split="dev")
```

Token đã thấy nhận nhãn nhiều nhất theo token; OOV nhận nhãn phổ biến nhất toàn
train. Cả hai trường hợp hòa dùng thứ tự 17 UPOS cố định. Predict/export không
cập nhật vocabulary. API cũ `MostFrequentTagger().fit(corpus_tuple)` được giữ qua adapter.
CSV có `split,sent_id,idx,token,gold,pred,is_oov`; idx bắt đầu từ 0 trong mỗi câu,
is_oov ghi True/False. Truyền `split` rõ ràng; nếu chỉ dùng bốn tham số gốc thì
cột split để trống, không suy đoán. sent_id thiếu cũng để trống. vocab_train phải
khớp tập FORM nguyên gốc model đã học, không dùng vocabulary sau cutoff.
Chỉ xuất dự đoán test sau khi khóa cấu hình. Fit chỉ nhận dữ liệu train do người
chạy cung cấp; tên split không thể được xác minh tự động từ nội dung Sentence.

### Mốc ngày 23 — test độc lập và phân tích lỗi dev

```bash
python -m pytest tests/test_bruteforce.py tests/test_normalization.py tests/test_edge_cases.py -q -rx
python -m src.error_script results/tables/baseline_dev.csv --split dev --limit 10
```

Các test normalization/edge-case gọi model qua `tests/adapters.py` đã nối với TV2.
Oracle giữ nguyên bảng xác suất và kỳ vọng độc lập; không gọi decoder để dựng
đáp án. Cơ chế XFAIL chờ API đã được bỏ để lỗi tích hợp không bị che.
Test normalization fit trên train toy và kiểm tra xác suất thực sự model dùng;
test leakage hash counts/vocabulary trước và sau khi thay đầu vào dev/test toy.
Test câu dài kiểm tra score đường đi, độ dài và tính lặp lại, không khẳng định
tối ưu bằng vét cạn. Test các HMM ngẫu nhiên dùng 200 bảng mỗi bậc, seed 42.

API `BigramHMMTagger.from_probabilities(...)` và Trigram tương ứng chỉ nạp bảng
xác suất cho kiểm thử, không fit hoặc smoothing lại. Decoder vẫn là cùng code
Viterbi dùng cho model train thật. Xem [biên bản tích hợp](docs/INTEGRATION_TV1_TV2.md).

Error script nhận CSV gộp `sent_id,idx,token,gold,pred_b0,pred_h1` và tùy chọn
`pred_h2,is_oov`. Khi CSV không có `is_oov`, dùng `--train-conllu` để dựng
vocabulary FORM nguyên gốc của train; thiếu cả hai nguồn thì script dừng thay vì
đoán OOV. Chạy:

```bash
python -m src.error_script results/tables/predictions_dev.csv --train-conllu data/raw/TRAIN.conllu --split dev
```

Script xuất `errors_dev.csv`, `top_error_pairs_dev.csv`, `error_examples_dev.md`
vào `results/tables/errors/`. Cặp nhầm tách theo OOV/IV và PUNCT/NON_PUNCT;
phần trăm lấy trong đúng nhóm model × OOV × PUNCT đó. Ví dụ chọn xác định: duyệt
các nhóm cặp theo count giảm dần rồi thứ tự khóa cố định, lấy vị trí đầu tiên chưa
chọn của mỗi nhóm cho đến tối đa 10 ví dụ. Cửa sổ giữ ±3 token; H2 vắng hiện N/A;
cột giải thích luôn có `[điền sau khi xem]`. Script không tính metric TV3, fit hay
tune model. Chỉ dùng `--split test` sau khi khóa cấu hình.

### Thống kê sparsity từ train

```python
from src.eda import export_sparsity

export_sparsity(splits["train"], splits["dev"])
```

Hàm ghi `results/tables/sparsity.csv` với một dòng số liệu. Một token type mơ hồ
là FORM đã xuất hiện với hơn một UPOS trong train; phần trăm tương ứng lấy số
lượt xuất hiện của các type đó chia tổng lượt token train. Bigram/trigram được
đếm trên chuỗi nhãn có `<s>` ở đầu và `</s>` ở cuối. Không gian khả dĩ lần lượt
là `(|S| + 1)^2` và `(|S| + 1)^3`, với `S` là nhãn xuất hiện trong train; một
chiều dùng `<s>`, chiều đích dùng `</s>`. Tỷ lệ bigram dev chưa thấy tính theo
lượt transition `<s> -> y1`, các chuyển tiếp nội bộ và `yT -> </s>`.

Các số liệu này chỉ mô tả sparsity, hỗ trợ giải thích vì sao Trigram có thể kém
Bigram; chúng không chứng minh quan hệ nhân quả và không được dùng để chọn tham
số theo test.
