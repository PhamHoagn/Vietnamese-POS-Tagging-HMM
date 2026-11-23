# Checklist nghiệm thu bài tập lớn

Trạng thái ngày 24/11/2026, theo các mã việc trong kế hoạch v4.1.

| Mã việc | Chủ trì | Trạng thái và minh chứng | Kiểm tra |
| --- | --- | --- | --- |
| W01–W04 | TV1 | Đạt trên train/dev. Parser, B0, audit nguồn, EDA và hình. Audit split Test chờ W15. [Kiểm kê và phân tích dữ liệu](TV1_DATA.md) | TV1, QA |
| W05–W09, W08a | TV2 và TV1 | Đạt. Bigram, Trigram NumPy, benchmark CPU và bài giải tay đã đối chiếu với oracle độc lập. [Trigram và benchmark](TV2_TRIGRAM.md), [giải tay và demo](TV2_GIAI_TAY_VITERBI.md) | TV1, manager |
| W10–W12, W16 | TV3 cùng TV1 và TV2 | Đạt trên dev. Grid 9+9, mở rộng 12+12, metric, per-tag, confusion và lỗi dev. [Bảng dev](../results/tables/dev/metrics_dev.csv), [biểu đồ lưới](../results/figures/dev/grid_sensitivity.png), [10 lỗi dev](TV1_DATA.md#phân-tích-10-vị-trí-trên-dev) | QA tính lại điểm và ma trận từ dự đoán |
| W13–W15 | TV1, TV2, TV3 | Ablation dev đã chạy. Hồ sơ chờ TV1 và TV3 xác nhận theo Phụ lục A, sau đó khóa và chạy một đợt đủ 5 vai trò, xuất đủ 3 CI. [Biên bản](pre_register_test.md), [hồ sơ ứng viên](../results/tables/final_candidate.json) | Manager kiểm thử khóa và bootstrap với dữ liệu giả lập |
| W17 | TV1, TV2, TV3 | Chờ kết quả chính thức. Audit Test, kiểm 5–10 lỗi, đối chiếu OOV, no-PUNCT và phân tích uniform, sparsity cùng CI. [Lệnh audit](TV1_DATA.md#chạy-lại) | TV1 kiểm tra căn chỉnh và hash, TV2 và TV3 đọc chéo |
| W18–W19 | Nhóm | Phần viết báo cáo TV3 được hoãn theo phạm vi hiện tại. [Bản báo cáo của TV3](../report/report_nlp.docx) | Nhóm |
| W20 | TV1 | README và notebook đã chạy từ kernel Jupyter mới. Kiểm tra liên kết và quyền trên môi trường chia sẻ sau khi đưa bản chốt lên nhánh nộp. [README](../README.md), [notebook](../notebooks/NLP_POS.ipynb) | TV1, manager |
| W21 | Nhóm | Slide 12 trang và demo đã kiểm tra. Cập nhật bảng kết quả Test, CI, ví dụ lỗi và kết luận, rồi diễn tập 10 phút và 5 phút hỏi đáp. [Slide nhóm](../slides/NLP_POS_NHOM.pptx), [câu hỏi theo vai trò](DEFENSE_QA.md) | TV2 đã mở và rà bản dev trong PowerPoint |
| W22 | Nhóm | Soát số liệu và công thức sau Test, kiểm README/slide/notebook và quyền xem. Rà bản PDF, chốt nhánh và nộp khi nhóm xử lý phần báo cáo. | Nhóm |
