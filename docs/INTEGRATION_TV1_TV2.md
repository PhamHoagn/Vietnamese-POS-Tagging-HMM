# Tích hợp TV1 và TV2 trên nhánh feature

## Phạm vi được người dùng duyệt

Ngày 2026-11-21, người dùng yêu cầu thực hiện tới bước 3: commit riêng giai đoạn
Trigram, ghép TV1, nối adapter/import và chạy oracle cùng toàn bộ test; chưa merge
vào main. Nhánh làm việc là `feature/hmm-viterbi`.

- Checkpoint Trigram và benchmark: `0ab2285`.
- Nguồn TV1: `5b554a18a3d301d339ce355bec0ea9b26cba99db`, nhánh
  `origin/feature/data-baseline`.
- Main cần giữ nguyên: `9e71502ae11b96963ad6075b6dd116b5cda321df`.
- Ghép bằng merge thông thường trên feature, giữ lịch sử commit của TV1;
  không có xung đột Git. Không rewrite lịch sử hoặc đưa thay đổi lên main.

## Các thay đổi tích hợp

1. Chuẩn hóa import trợ giúp thành `from tests...` vì tests là package của TV2.
2. Hoàn thiện `tests/adapters.py`: nạp bảng toy, gọi predict/log_joint của model,
   lấy hàng xác suất từ bảng log model thực sự dùng, và snapshot counts/vocabulary.
3. Thêm `src/hmm_tables.py` và `from_probabilities` để nạp xác suất cho trước.
   Không viết lại decoder và không dùng oracle để tạo dự đoán trong adapter.
4. Bỏ các nhánh XFAIL chờ API; lỗi decoder/adapter bây giờ làm test thất bại.
5. Bổ sung kiểm tra API tích hợp và cập nhật README/benchmark provenance.

`tests/oracle_bruteforce.py`, bảng xác suất tính tay, expected counts, công thức
normalization và các assertion toán học của TV1 được giữ nguyên. Chỉnh trong ba
file test oracle/normalization/edge chỉ gồm import và bỏ cơ chế chờ API.

## Nạp bảng xác suất và model train thật

```python
model = BigramHMMTagger.from_probabilities(labels, vocabulary, q1, emission, initial=pi)
model = TrigramHMMTagger.from_probabilities(labels, vocabulary, q2, emission)
```

- Bảng toy có thể dùng nhãn A/B/C và thứ tự nhãn cho trước; `fit(train_data)` vẫn
  chỉ nhận UPOS hợp lệ và sắp nhãn như contract.
- Bảng được kiểm tra miền xác suất/tổng hàng rồi sao chép, giữ nguyên số, không
  smoothing hoặc chuẩn hóa lại. Xác suất 0 thành log -inf; không có đường đi
  xác suất dương thì predict báo ValueError.
- Bigram toy giữ pi riêng trên S, không thêm xác suất EOS rồi chuẩn hóa lại pi.
  Trigram dùng q2(START,START), có EOS trong miền đích như bảng cho trước.
- Decoder gọi đúng `_viterbi` NumPy của model; `log_joint` của model bảng tính
  trực tiếp bằng bảng xác suất thường, không đọc cached log arrays của decoder.
  Model train vẫn chấm trực tiếp từ counts, không thay ước lượng Add-alpha.
- Toy không có UNK từ chối OOV; có khóa `<UNK>` thì dùng khóa đó. Riêng model
  train dùng sentinel phân biệt với FORM thật `<UNK>`.
- Câu rỗng Bigram toy bị từ chối; Trigram toy trả [] và score q2(EOS|START,START).
  Quy ước câu rỗng của model train giữ nguyên.
- Probability-only model không được lưu thành config train hoặc xuất snapshot
  counts giả. Có thể gọi fit để thay toàn bộ trạng thái bằng model học từ train.

## Snapshot chống leakage

Adapter sao chép counts thực từ model: transition (gồm start/end), emission sau
cutoff, tag counts và raw word counts. Không đếm lại train hoặc lấy expected
counts từ test. Khóa tuple mã hóa JSON, không ghép bằng dấu cách.

`train_vocab` giữ nguyên FORM; `emission_vocab` dùng chuỗi JSON `["FORM", word]`
và `["UNK"]` để hai ký hiệu không va chạm. Cả hai phản ánh vocabulary hiện tại,
không dùng dữ liệu dev/test để bổ sung. Hàng emission có khóa chuỗi không thể biểu
diễn đồng thời literal `<UNK>` và sentinel, nên adapter báo lỗi rõ trong trường
hợp đó; snapshot vẫn biểu diễn được cả hai bằng namespace.

## Kiểm tra và cách chạy

Tại repo root, cài requirements đã bổ sung matplotlib từ nhánh TV1, rồi chạy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.venv/tmp/integration-tests
.\.venv\Scripts\python.exe -m scripts.smoke_test
```

Kết quả oracle riêng: **579 passed in 0.93s** trước khi chạy toàn suite.
Các ca này gồm 200 HMM ngẫu nhiên mỗi bậc, toy giải tay, normalization từ train,
start/end, câu ngắn/dài, OOV, tính quyết định và hash chống leakage.
Các test tích hợp không đọc corpus test thật; mọi biến có tên test trong test
leakage là corpus toy tạo trong code. Benchmark chỉ dùng train/dev công khai.

Kết quả cuối: **875 passed, 3 subtests passed in 3.69s**, không XFAIL; smoke PASS.
Benchmark lại trên 100 câu dev: dự phóng 9 cấu hình Bigram 3.45 giây, Trigram
12.77 giây, tổng 16.22 giây, đã nhân 1.5 và dưới ngân sách 90 phút. CSV/JSON
được cập nhật hash decoder, helper nạp bảng và parser đã tích hợp. Runtime gate
độc lập với kiểm thử tính đúng, không thay thế oracle. Chưa push remote.

Chưa triển khai metric còn thiếu, grid search, bootstrap CI hoặc báo cáo của TV3.
Chưa có đánh giá chất lượng trên test và chưa chạy Colab thật.
