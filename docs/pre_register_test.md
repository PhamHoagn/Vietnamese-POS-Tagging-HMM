# DRAFT — đăng ký trước đánh giá test

**CHƯA ĐƯỢC KHÓA.** Cả ba thành viên phải xác nhận và khóa bản này trước khi xem
bất kỳ kết quả test nào. Không điền kết quả giả vào tài liệu. Các lựa chọn dưới đây
là đề xuất cần cả nhóm duyệt, không phải những thí nghiệm đã chạy.

## Cổng xác nhận

- [ ] TV1 @angqun xác nhận corpus, quyền phân phối, tagset, split/checksum.
- [ ] TV2 @PhamHoagn xác nhận smoothing, cutoff, UNK, BOS/EOS và brute-force tests.
- [ ] TV3 @PhuongNamkk5 xác nhận metric, grid, bootstrap và quy tắc báo cáo.
- [ ] Ghi ngày khóa, commit tài liệu, phiên bản môi trường và người xác nhận.

Corpus/phiên bản/giấy phép: **TBD**. Train/dev/test IDs và checksum: **TBD**.
Thời điểm khóa và commit: **TBD**. Chưa được chạy test cuối khi còn TBD quan trọng.

## Câu hỏi và các phép so sánh dự kiến

1. Most Frequent Tag so với Bigram HMM được chọn bằng dev.
2. Most Frequent Tag so với Trigram HMM được chọn bằng dev.
3. Bigram so với Trigram sau lựa chọn dev độc lập, dùng cùng split và tagset.
4. **Ablation cố định:** Bigram `alpha=0.1, cutoff=1` so với Trigram
   `alpha=0.1, cutoff=1`; chỉ thay bậc Markov, giữ cùng định nghĩa emission,
   preprocessing, UNK, boundary và smoothing. Không thay cặp này sau khi xem test.
5. Phân tích phụ OOV và loại PUNCT sau khi các metric tương ứng được triển khai
   và kiểm thử. Không coi giá trị `None` hiện tại là kết quả số.

## Lựa chọn bằng dev

Đề xuất grid `alpha in [0.01, 0.1, 1.0]`, `cutoff in [0, 1, 2]`; nhóm phải xác
nhận chi phí CPU trước khi khóa. Huấn luyện từng cấu hình từ train, chọn bằng
dev Token Accuracy; hòa chọn `(alpha, cutoff)` nhỏ nhất theo thứ tự từ điển.
Không xem test để chọn grid, cấu hình, seed hoặc thay cách UNK. Nếu cấu hình lỗi,
ghi nguyên nhân và xử lý trước khi khóa, không âm thầm bỏ sau khi xem test.

## Đánh giá cuối

- Primary: Token Accuracy gộp mọi token, bao gồm PUNCT.
- Secondary: Macro-F1 trên 17 `UPOS_TAGS`, kể cả nhãn vắng, `zero_division=0`.
- Phụ: OOV Accuracy/Rate dùng raw train vocabulary, accuracy_no_punct lọc theo
  gold PUNCT. Báo số token tổng/OOV/không-PUNCT và quy ước mẫu số bằng 0.
- Giữ train/dev tách biệt; model cuối vẫn fit train, không tự gộp train+dev.
- Lưu predictions trước khi tính metric; predict chỉ nhận tokens.
- Không dùng token của test để mở rộng vocabulary hoặc thay rare cutoff.

## CI đề xuất, chưa triển khai

Paired bootstrap theo **câu**, lấy lại N câu có hoàn lại từ N câu test, cùng chỉ
số lấy mẫu cho cả hai model. Đề xuất 10.000 mẫu, seed 2026, percentile CI 95% cho
chênh lệch Token Accuracy và Macro-F1; tính lại metric trên token gộp trong mỗi
mẫu, giữ nguyên 17 labels. Không bootstrap token độc lập. TV3 xác nhận thời gian
CPU và số mẫu trước khi khóa. Các CI nhiều cặp mang tính mô tả; không tuyên bố
kiểm định đã hiệu chỉnh nhiều so sánh nếu chưa đăng ký phương pháp cụ thể.

## Tái lập và thay đổi protocol

Lưu cấu hình, seed, thời gian CPU, package versions, split checksum và Git commit
cho từng lần chạy. Không có bảng kết quả thực nghiệm trong bản nháp này.
Nếu cần sửa sau khi khóa, ghi ngày, lý do, người duyệt và đã xem test hay chưa;
phân tích phát sinh sau khi xem test phải được ghi là exploratory.
