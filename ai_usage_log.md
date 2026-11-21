# Nhật ký sử dụng AI

## 2026-11-20 Giai đoạn 0 Đọc và đối chiếu

- Hỗ trợ: đọc code/contract và hai tài liệu Word người dùng cung cấp; đối chiếu
  phạm vi TV2, công thức, API, nhãn train-only và trách nhiệm kiểm chứng độc lập.
- Chưa thay đổi source trong giai đoạn 0; mục này được bổ sung khi người dùng
  xác nhận bắt đầu giai đoạn 1. Hai tài liệu Word không được sao chép vào repo.
- Người dùng cần tự xác nhận: API mới và API tương thích, câu rỗng, bộ nhãn HMM;
  thống nhất khác biệt metric/grid/bootstrap với TV3. Chưa có Lab/Module 6 mẫu
  trong repo để đối chiếu trực tiếp.
- Lỗi môi trường: pytest mặc định báo 7 setup errors do quyền thư mục temp Windows;
  chạy lại với `--basetemp` trong `.venv` được 55 passed, smoke test PASS.

## 2026-11-20 Giai đoạn 1 Bigram HMM

- AI hỗ trợ tạo logic counts/train-only UNK/cache trong `src/hmm_common.py`,
  triển khai `src/hmm_bigram.py`, gồm Viterbi NumPy, log_joint tính từ counts,
  API tương thích, batch prediction và save/load config JSON.
- Tạo `tests/__init__.py`, `tests/toy_corpus.py`, `tests/test_hmm_bigram.py` và
  `docs/TV2_BIGRAM.md`; cập nhật contract chung và smoke output để phản ánh trạng
  thái Bigram. Test placeholder Bigram trong `tests/test_contract.py` được thay
  bằng bộ kiểm tra hành vi Bigram mới; vẫn giữ kiểm tra placeholder Trigram.
- Không sửa parser, baseline, metrics, grid search hoặc báo cáo của TV1/TV3;
  không triển khai Trigram, không fit dev/test và không tạo số liệu thực nghiệm.
- Phạm vi test AI đề xuất: kỳ vọng tính tay theo phân số, chuẩn hóa, EOS, cutoff,
  OOV/literal UNK, vét cạn câu ngắn bằng log_joint, tie-break toàn chuỗi, cache,
  dữ liệu caller bị sửa, refit, config và câu dài. Đây là self-check TV2, không
  phải oracle nghiệm thu độc lập TV1.
- Điểm cần người dùng tự kiểm chứng: bảng phân số toy; count START/EOS và UNK;
  recurrence/backtracking; rank dùng xử lý hòa điểm; hash và vòng đời cache;
  save/load config cần refit, không lưu weights. TV1 cần tự viết oracle không
  lấy kỳ vọng từ hàm score/decoder này.
- Rà soát đã bổ sung: từ chối raw string thay vì âm thầm giải mã từng ký tự;
  tách literal FORM `<UNK>` khỏi sentinel; cảnh báo khi đổi config sau fit;
  sửa thông báo smoke cũ nói cả Bigram/Trigram đều chưa triển khai.
- Kết quả bước kiểm tra đầu: 107 passed cho Bigram + contract. Sau rà soát input
  raw string và cập nhật tài liệu, toàn bộ suite: **138 passed in 1.32s**, exit 0.
  Lệnh: `.\.venv\Scripts\python.exe -m pytest -q --basetemp=<thư mục mới trong .venv/tmp>`.
- `.\.venv\Scripts\python.exe -m scripts.smoke_test`: PASS, exit 0. Smoke vẫn chạy
  baseline, không tạo kết quả HMM giả. `git diff --check`: sạch. Chưa commit/push;
  dừng ở cuối giai đoạn 1 chờ người dùng duyệt trước khi triển khai Trigram.

## 2026-11-21 Giai đoạn 2 Trigram NumPy và benchmark

- Người dùng yêu cầu tiếp tục công việc TV2, chưa merge TV1. AI hỗ trợ triển khai
  `src/hmm_trigram.py`, mở rộng `src/hmm_common.py` để đếm hai bậc trong một lần
  duyệt train và cache log emission chung. Refactor hook preparation/config của
  `src/hmm_bigram.py` để tái sử dụng vòng đời model, giữ API và config Bigram cũ.
- Tạo `tests/test_hmm_trigram.py`, `tests/test_benchmark_hmm.py`,
  `scripts/benchmark_hmm.py`, `docs/TV2_TRIGRAM.md`; bổ sung toy counts/phân số trong
  `tests/toy_corpus.py`, thay test placeholder Trigram bằng kiểm tra dự đoán thật
  trong `tests/test_contract.py`; cập nhật contract và liên kết tài liệu Bigram.
- Không sửa source/test parser, baseline, EDA, metrics, grid search, báo cáo hay
  adapter của TV1/TV3; không merge nhánh nào. Các file benchmark CSV/JSON chỉ ghi
  số đo runtime và metadata, không chứa văn bản corpus hoặc điểm chất lượng.
- Người dùng chưa biết vị trí corpus thật. AI xác minh nguồn UD Vietnamese-VTB,
  ghim commit `fa7130860b65db9db342542c51dc57badd30fa9e`, tải riêng train/dev và
  README/LICENSE vào thư mục local được ignore `.venv/benchmark_data/`.
  SHA-256 của train/dev và mẫu câu được ghi trong `results/tables/bench_dev.json`.
  Không tải hoặc parse file test; không đưa corpus gốc vào Git.
- Self-check TV2: counts START/END tính tay, chuẩn hóa mọi lịch sử kể cả chưa thấy,
  uniform thay vì Bigram backoff, emission/UNK dùng chung, vét cạn câu ngắn qua
  log_joint, 60 corpus toy ngẫu nhiên seed 2026, câu dài, một nhãn, câu rỗng, EOS
  làm đổi nghiệm một token, tie-break, cache on/off và save/load config khác bậc.
- Người dùng cần tự kiểm chứng: tensor `[u,v,w]` và argmax theo u; khởi tạo hai
  START; backpointer lưu u và dịch cặp khi truy vết; mẫu số có EOS; rank prefix;
  log_joint tính từ counts. Các self-check này không thay oracle độc lập của TV1.
- Benchmark thực tế: train 1400 câu/20215 token; dev 1123 câu/26162 token;
  mẫu 100 câu/2357 token, dài 3–96. Mỗi bậc đo 9 cấu hình, 3 lần decode/config,
  median, cache chung. Dự phóng có hệ số 1.5: Bigram 3.4566 giây, Trigram 16.7678
  giây, cả hai 20.2244 giây; dưới ngân sách 90 phút trên Windows local này.
  Đây là dự phóng runtime, không phải full-grid quality experiment hoặc số đo Colab.
- Kết quả cuối: **209 passed in 1.44s**, smoke PASS; metadata hash code khớp.
  Pytest dùng basetemp mới trong `.venv/tmp` vì temp hệ thống đã từng lỗi quyền.
  Không có lỗi thuật toán nào được phát hiện trong lần chạy này; chưa có kiểm
  chứng oracle nghiệm thu của TV1. Runtime dao động giữa các lần đo; giữ lần đo
  cuối trong CSV/JSON thay vì chọn lần nhanh nhất để báo cáo.
- Lệnh chạy lại nằm trong `docs/TV2_TRIGRAM.md`, gồm lấy train/dev bằng commit/hash
  cố định, unit tests, smoke và benchmark. Chưa commit/push thay đổi giai đoạn 2;
  dừng sau bàn giao để người dùng review theo quy trình từng giai đoạn.
