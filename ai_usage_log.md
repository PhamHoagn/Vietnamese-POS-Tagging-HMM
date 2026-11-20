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
