# Checklist nghiệm thu bài tập lớn

Trạng thái ngày 24/11/2026, theo các mã việc trong kế hoạch v4.1.

| Mã việc | Chủ trì | Trạng thái và minh chứng | Kiểm tra |
| --- | --- | --- | --- |
| W01–W04 | TV1 | Đạt trên cả ba split. Parser, B0, audit nguồn, EDA và 3 biểu đồ. [Kiểm kê và phân tích dữ liệu](TV1_DATA.md) | TV1, QA |
| W05–W09, W08a | TV2 và TV1 | Đạt. Bigram, Trigram NumPy, benchmark CPU và bài giải tay đã đối chiếu với oracle độc lập. [Trigram và benchmark](TV2_TRIGRAM.md), [giải tay và demo](TV2_GIAI_TAY_VITERBI.md) | TV1, manager |
| W10–W12, W16 | TV3 cùng TV1 và TV2 | Đạt trên dev. Grid 9+9, mở rộng 12+12, metric, per-tag, confusion và lỗi dev. [Bảng dev](../results/tables/dev/metrics_dev.csv), [biểu đồ lưới](../results/figures/dev/grid_sensitivity.png), [10 lỗi dev](TV1_DATA.md#phân-tích-10-vị-trí-trên-dev) | QA tính lại điểm và ma trận từ dự đoán |
| W13–W15 | TV1, TV2, TV3 | Đã khóa và chạy Test một đợt đủ 5 vai trò, gồm cặp ablation. Đã xuất 3 CI. [Biên bản](pre_register_test.md), [metadata](../results/tables/test/run_metadata.json) | QA tính lại 25 ô chỉ số, 5 ma trận nhầm lẫn và 3 CI từ dự đoán đã lưu |
| W17 | TV1, TV2, TV3 | Đã audit Test, diễn giải 10 lỗi và đối chiếu OOV, no-PUNCT, độ thưa và CI. [Audit và ví dụ](TV1_DATA.md), [đối chiếu kết quả](TV3_EVALUATION_RUNBOOK.md#đối-chiếu-kết-quả-test) | TV1 đối chiếu 11.692 dòng dự đoán với gold, QA đối chiếu 389 cặp lỗi |
| W18–W19 | Nhóm | Phần viết báo cáo TV3 được hoãn theo phạm vi hiện tại. [Bản báo cáo của TV3](../report/report_nlp.docx) | Nhóm |
| W20 | TV1 | Đã chạy cả 5 cell code từ ZIP commit `16579f8` trên kernel Jupyter mới. Tái tạo 14 CSV dev, hiển thị 4 hình và 2 bảng Test đã xác minh hash. [README](../README.md), [notebook](../notebooks/NLP_POS.ipynb) | TV1, manager |
| W21 | Nhóm | Deck 14 slide, lời dẫn 10 phút, demo và bộ câu hỏi đã sẵn sàng cho buổi diễn tập. Kết quả Test, 3 CI và ví dụ lỗi đã lên slide. [Slide nhóm](../slides/NLP_POS_NHOM.pptx), [câu hỏi theo vai trò](DEFENSE_QA.md) | TV2 đối chiếu số liệu và render đủ 14 slide trong PowerPoint |
| W22 | Nhóm | Đã soát số liệu, công thức, README, slide và notebook. Buổi diễn tập, quyền xem trên bản chia sẻ, rà PDF và nộp thuộc đợt bàn giao của nhóm. | TV1, TV2, TV3, QA |
