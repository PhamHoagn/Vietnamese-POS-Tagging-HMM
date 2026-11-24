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

## 2026-11-21 Tích hợp TV1 và TV2 theo bước 1 đến 3

- Người dùng duyệt: commit riêng GĐ2, ghép TV1 và chạy toàn bộ kiểm thử; chưa
  merge vào main. Đã tạo checkpoint `0ab2285`, ghép commit TV1 `5b554a1` vào
  `feature/hmm-viterbi` bằng merge giữ lịch sử. Main giữ `9e71502`.
- AI hỗ trợ nối `tests/adapters.py`: gọi model thật cho decode/score và lấy các
  bảng log/count thật; không dùng oracle hoặc dữ liệu kỳ vọng thay kết quả.
- Thêm `src/hmm_tables.py`, API `from_probabilities` của HMM và nhánh score từ
  bảng thường cho toy. Cùng `_viterbi` dùng cho train thật, giữ pi Bigram riêng,
  hai START Trigram, EOS và các số 0; không huấn luyện/smoothing lại bảng toy.
- Nối import package `tests` ở ba file test độc lập; bỏ XFAIL chờ API. Giữ nguyên
  `tests/oracle_bruteforce.py`, bảng/count kỳ vọng và assertion công thức của TV1.
  Parser, baseline, EDA và error script không có chỉnh sửa ngoài bản TV1 ghép vào.
- Tạo `tests/test_adapters.py` kiểm tra copy, hàng xác suất độc lập, số 0, đường đi
  không thể, OOV, bảng lỗi, checkpoint và snapshot counts/vocab không va chạm UNK.
  Snapshot mã hóa namespace JSON cho emission vocabulary, không tái đếm train.
- Cập nhật README, contract, tài liệu Trigram, tạo `docs/INTEGRATION_TV1_TV2.md`;
  cập nhật benchmark script/provenance và đo lại CSV/JSON sau đổi code/parser.
- Kết quả oracle/normalization/edge riêng: **579 passed in 0.93s**. Cuối cùng toàn
  suite: **875 passed, 3 subtests passed in 3.69s**, không XFAIL; smoke PASS.
  Đã kiểm tra 200 HMM ngẫu nhiên mỗi bậc qua oracle độc lập và các ca leakage toy.
- Benchmark cuối trên 100 câu dev/2357 token: 9 Bigram dự phóng 3.4455 giây,
  9 Trigram 12.7725 giây, tổng 16.2180 giây, đã nhân an toàn 1.5; gate 90 phút PASS.
  Số đo local thay đổi theo tải máy; không phải số đo chất lượng hoặc chạy Colab.
- Người dùng cần review helper nạp bảng chỉ đổi biểu diễn; adapter không có
  decoder/oracle thứ hai; snapshot lấy counts thật; semantics toy/fit được phân
  biệt. Không phát hiện lỗi công thức khi chạy oracle; lỗi import và API thiếu
  trước tích hợp đã được xử lý, không che bằng skip/XFAIL.
- Chỉ hoàn tất merge commit trên feature sau khi kiểm thử đạt. Không merge main,
  không push; chưa dùng test corpus thật, chưa làm phần metrics/grid/CI của TV3.

## 2026-11-23 Nghiệm thu và bàn giao TV2

- Rà lại Bigram/Trigram, trường hợp 1 token, EOS, UNK, chuẩn hóa và cache qua toàn bộ
  suite của nhánh `feature/hmm-viterbi`. Kết quả **875 tests và 3 subtests đạt**,
  smoke test đạt. Không phát hiện lỗi decoder cần sửa trong lần nghiệm thu.
- Đo lại benchmark bằng `scripts/benchmark_hmm.py` trên 100 câu dev, 2.357 token,
  3 lần decode mỗi cấu hình. Dự phóng 9 cấu hình sau hệ số 1,5 lần là 3,62 giây
  (Bigram) và 13,75 giây (Trigram), đều dưới ngân sách 90 phút trên máy local.
  Bản ghi lần chạy lại lưu ở `.venv/tv2-final/bench_dev.csv` và JSON cùng tên.
- Tạo `report/TV2_Phuong_phap_HMM.docx`, bản nháp 2 trang có phương trình Word
  chỉnh sửa được, và `docs/TV2_GIAI_TAY_VITERBI.md` với diễn giải 3 token, kịch bản
  demo, phản biện và cách dùng model factory với cache. Đối chiếu bảng phân số
  với nhãn và `log_joint` từ code. Giữa hai lần xuất Word, bỏ đường kẻ mặc định
  ở Title và sửa cách diễn đạt xác suất đồng thời. Kiểm tra ảnh cả hai trang sau
  lần xuất cuối. Renderer của skill thiếu LibreOffice trên Windows, nên dùng
  Word để xuất PDF và chuyển PDF sang ảnh để kiểm tra bố cục.
- Tạo `scripts/demo_tv2.py`, đã chạy để kiểm chứng luồng đọc fixture, fit hai câu
  train và hiển thị Gold/B0/Bigram/Trigram/OOV cho câu giữ lại. Lệnh cài, chạy demo,
  kiểm thử và smoke nằm trong tài liệu giải tay, lệnh lấy train/dev và benchmark
  nằm trong `docs/TV2_TRIGRAM.md`.
- Phần thảo luận dùng kết quả dev cố định đã đo để minh họa chênh lệch H2−H1.
  Độ thưa lịch sử chỉ là giả thuyết cần đối chiếu. Kết quả test, CI, chọn cấu hình
  cuối và số liệu báo cáo chung thuộc đợt thực nghiệm do nhóm chốt.
- TV2 tự luyện giải ô `(V,V)` ở token thứ ba và truy ngược `N V V`, giải thích tại
  sao mẫu số gồm EOS, vì sao UNK có xác suất dương khi cutoff bằng 0 và cách phối
  hợp cache với grid của TV3. Đã xác minh trên Windows CPU, Colab chưa chạy trực tiếp.

## 2026-11-23 Điều phối tích hợp và thực nghiệm dev

- Manager tích hợp TV1 `8720c6c` và TV3 `7f003cd` vào `feature/hmm-viterbi` sau khi
  lưu Phương pháp và demo TV2. Bản báo cáo TV3 giữ nguyên nội dung của tác giả ở
  `7f003cd`.
- Agent TV1 tái lập audit, bảng và hình train/dev từ UD Vietnamese-VTB đã ghim,
  kiểm tra căn chỉnh 26.162 dòng dự đoán dev, diễn giải 10 ví dụ theo ID và kiểm tra
  toàn bộ notebook bằng một kernel mới. Corpus và dự đoán chứa FORM lưu local.
- Agent TV2 hoàn thiện slide HMM, lời dẫn và sửa lỗi mã hóa đầu ra demo khi Windows
  dùng cp1252. Slide trình bày cả truy vết toy, độ thưa và cách đo tốc độ.
- Agent TV3 chốt quy ước dev v4.1, chạy đủ 9+9 rồi mở rộng đến 12+12 khi Bigram
  chạm biên alpha thấp, xuất metric, per-tag, confusion matrix và bootstrap theo câu.
  Một lần chạy riêng tái tạo các bảng và dự đoán dev giống từng byte.
- Manager tạo CLI lập hồ sơ, khóa cấu hình và chạy test cuối với đúng các tham số
  đã chọn trên dev. Kiểm tra toy chứng minh hồ sơ chưa khóa bị từ chối, thay đổi
  tham số hoặc source làm hỏng khóa, các vai trò trùng cấu hình dùng chung dự đoán.
  Hồ sơ `results/tables/final_candidate.json` ghi commit đánh giá `e33563b`.
  QA đề nghị ghim cả các mô đun được nạp gián tiếp. Mã băm nguồn đã được mở rộng
  trước khi khóa và bảng cấu hình ứng viên giữ nguyên.
- QA tính lại toàn bộ điểm và năm ma trận nhầm lẫn từ dự đoán dev. Kiểm tra
  cp1252 bằng test toy phát hiện và xác minh bản sửa lỗi ghi kết quả trên Windows.
  Hai kiểu xuống dòng LF/CRLF cho cùng mã băm hồ sơ dev. Suite tích hợp
  đạt 905 bài kiểm thử và 3 bài kiểm thử con.
- Slide chung 12 trang có lời dẫn 10 phút và đã được kiểm tra trong PowerPoint.
  Biểu đồ dev dùng trực tiếp CSV lưới mở rộng. TV1 đã thử audit sau khóa trên toy.
  Cấu hình test vẫn chờ nhóm xác nhận.

## 2026-11-24 Kiểm tra notebook trên bản đóng gói

- Manager cập nhật notebook để sinh và hiển thị hình độ nhạy theo lưới đã dùng
  trong lần chạy dev, chỉnh README và soạn câu hỏi bảo vệ theo vai trò.
- Kiểm lại ZIP xuất từ commit `c81f0ec` trên kernel Jupyter mới tại Windows.
  Tải mới train/dev, tái tạo đúng nội dung 14 CSV dev, cấu hình và quy tắc chọn,
  đồng thời hiển thị đủ bốn hình. Nhóm cần diễn tập và kiểm tra quyền truy cập
  trên bản chia sẻ cuối.

## 2026-11-24 Đánh giá Test đã khóa

- Manager ghi xác nhận của người dùng, khóa đúng candidate đã chốt trên dev rồi
  chạy Test một đợt. Bốn cấu hình duy nhất phục vụ năm vai trò, gồm ablation.
- Agent QA tính lại từ dự đoán gốc 25 ô chỉ số, 85 dòng theo UPOS, 5 ma trận nhầm
  lẫn, 3 khoảng tin cậy và 389 cặp lỗi. Các số đo và mã băm khớp đầu ra đã lưu.
- Agent TV1 kiểm kê ba split, cập nhật ba hình và diễn giải 10 lỗi Test theo
  ID câu, vị trí và nhãn. Dữ liệu gốc và dự đoán chi tiết được giữ local.
