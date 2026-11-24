# Đăng ký trước đánh giá test

**Đã khóa lúc 10:16:21 ngày 24/11/2026 (UTC+7).** Hồ sơ lưu xác nhận qua cuộc trò chuyện rằng TV1 và TV3 đã duyệt chạy Test với ablation. Bản đăng ký dựa trên dev đã được lưu trước khi mở Test. [Hồ sơ đã khóa](../results/tables/final_lock.json)

- [x] TV1 xác nhận corpus, giấy phép và phạm vi đánh giá.
- [x] TV2 đối chiếu hai bộ giải mã, kiểm thử oracle và ngân sách CPU.
- [x] TV3 xác nhận metric, cấu hình và bootstrap.
- [x] Ghi nhận xác nhận qua cuộc trò chuyện và niêm phong hồ sơ trước đợt chạy Test.

## Dữ liệu

UD Vietnamese VTB, CC BY-SA 4.0, commit `fa7130860b65db9db342542c51dc57badd30fa9e`. Giữ nguyên FORM (không chuẩn hóa thêm), nhãn UPOS và các split gốc.

| Split | Bằng chứng định danh |
| --- | --- |
| Train | SHA-256 `1df180722db99b724f7368aca50597133bee28606de3394cd8d12a57da89978d` |
| Dev | SHA-256 `0b779257b9225575b3b30e15d64cc4ef9945a77f8cceb9f92d794b483a2752dd` |
| Test | Git blob SHA-1 `29cfccf7bc782829f77d039db5d906157a747ae6`, 538.633 byte theo Git tree tại commit trên |

Hồ sơ dùng định danh Git của test để ghim nội dung trước khi lấy file. Lệnh chạy test kiểm tra bytes đúng Git blob và ghi SHA-256 trước khi parse. Tất cả mô hình học tham số và từ vựng từ train. Dev chọn cấu hình. Khi dự đoán, mỗi mô hình chỉ nhận token.

## Quy tắc lựa chọn trên dev

Grid khởi đầu cho cả hai HMM gồm α ∈ {0,01, 0,1, 1,0} và cutoff ∈ {0, 1, 2}. Xếp theo Token Accuracy giảm dần, sau đó Macro-F1 giảm dần (bằng nhau trong dung sai tuyệt đối 10⁻¹²), cutoff tăng dần và α giảm dần. Lần mở rộng chung thêm α=0,001 vì Bigram đứng đầu vòng đầu ở α=0,01. Mỗi bậc có 12 cấu hình sau mở rộng.

| Vai trò trên test | Cấu hình | Token Accuracy dev | Macro-F1 dev |
| --- | --- | ---: | ---: |
| B0 | Most Frequent Tag | 81,2438% | 64,7742% |
| H1_best | Bigram α=0,01, cutoff=1 | 78,9389% | 67,3225% |
| H2_best | Trigram α=0,1, cutoff=1 | 78,5949% | 65,9711% |
| H1_fixed | Bigram α=0,1, cutoff=1 | 78,8854% | 66,4386% |
| H2_fixed | Trigram α=0,1, cutoff=1 | 78,5949% | 65,9711% |

H2_best và H2_fixed trùng tham số và dùng chung dự đoán. Hai mô hình fixed cùng tập nhãn train, từ vựng, cutoff, alpha và phát xạ. Lưới, điểm và phiên bản môi trường của lần chạy nằm trong [hồ sơ lựa chọn dev](../results/tables/dev/selection.json).

Trigram đạt kiểm thử oracle, chuẩn hóa và ca biên của TV1, cùng ngưỡng benchmark CPU trên máy local. Đây là cơ sở để đưa cả H2_best và H2_fixed vào đợt test.

## Chỉ số và khoảng tin cậy

- Chỉ số chính là Token Accuracy theo token có tính PUNCT.
- Macro-F1 dùng các nhãn UPOS có gold support trong split đang đánh giá, `zero_division=0`. Bảng per-tag và confusion matrix luôn theo thứ tự đủ 17 UPOS.
- OOV được xác định từ FORM vắng trong từ vựng gốc của train. Accuracy không tính dấu câu lọc bằng nhãn gold PUNCT. Metric có mẫu số bằng 0 trả 0,0.
- Paired sentence bootstrap cho **H1_best − B0**, **H2_best − H1_best** và **H2_fixed − H1_fixed**. Dùng các chỉ số câu giống nhau cho mọi cấu hình trong mỗi lần lấy mẫu, lấy lại N câu có hoàn lại, rồi cộng số token đúng và tổng token trước khi chia. Dùng 2.000 lần lấy mẫu, seed 42, phân vị 2,5% và 97,5% của chênh lệch Token Accuracy.
- Hồ sơ test giữ hash dự đoán, hash các bảng tổng hợp và phiên bản môi trường. Dữ liệu gốc, dự đoán từng token và văn bản ví dụ lỗi nằm trong thư mục local.
- SHA-256 của hồ sơ dev trong ứng viên dùng văn bản UTF-8 với xuống dòng LF. Các hash file dữ liệu và CSV Test dùng byte nguyên gốc của lần chạy.

## Trình tự thực hiện

1. `python -m scripts.final_experiment prepare --selection results/tables/dev/selection.json --output results/tables/final_candidate.json` sau khi commit code thực nghiệm. Hồ sơ chứa commit code, hash script và định danh dữ liệu.
2. TV1 và TV3 duyệt bảng cấu hình và ghi xác nhận khóa. Sau xác nhận, chạy `python -m scripts.final_experiment seal --candidate results/tables/final_candidate.json --confirmation "THAM_CHIEU_XAC_NHAN_CUA_NHOM" --output results/tables/final_lock.json` và thay tham chiếu bằng xác nhận thực tế.
3. Chạy `python -m scripts.final_experiment run --lock results/tables/final_lock.json --data-dir data/raw/ud-vtb-fa7130860b65 --private-dir .venv/project-runs/test --summary-dir results/tables/test` từ commit đã duyệt. Thay `--data-dir` theo nơi lưu train/dev đã xác minh trên máy đang chạy.

Mọi thay đổi protocol sau khi xem kết quả test cần ghi rõ thời điểm và lý do, và được tách khỏi phân tích đã đăng ký.

## Lần chạy đã khóa

Lần chạy hoàn tất lúc 10:16:25 ngày 24/11/2026 (UTC+7), trên 800 câu và 11.692 token Test. Trigram best và fixed dùng cùng một kết quả dự đoán, nên năm vai trò dùng bốn cấu hình duy nhất. [Bảng kết quả](../results/tables/test/metrics_test.csv) · [Ba khoảng tin cậy](../results/tables/test/paired_bootstrap_test.csv) · [Mã băm và phiên bản môi trường](../results/tables/test/run_metadata.json)
