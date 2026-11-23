# Chạy đánh giá mô hình

Pipeline dev dùng UD Vietnamese-VTB tại commit
`fa7130860b65db9db342542c51dc57badd30fa9e`. Cần hai tệp train và dev đúng SHA-256 trong
`scripts/evaluate_dev.py`. Quy trình kiểm tra mã băm trước khi parse. Mỗi mô hình
fit trên train, dự đoán từ token dev, và dùng từ vựng train trước cutoff để tính OOV.

```powershell
python -m scripts.evaluate_dev --data .venv/benchmark_data/ud-vtb-fa7130860b65
```

Kết quả tổng hợp lưu ở `results/tables/dev/`. Dùng `--output` để đổi thư mục.
CSV dự đoán chứa FORM lưu ở `.venv/project-runs/dev/predictions_dev.csv`.
Dùng `--predictions` để đổi đường dẫn trong vùng local được bỏ qua bởi Git.
Cột `idx` bắt đầu từ 0 trong mỗi câu và `sent_id` giữ nguyên từ CoNLL-U.
`pred_b0`, `pred_h1`, `pred_h2` là nhãn baseline và hai cấu hình tốt nhất.
`pred_h1_fixed`, `pred_h2_fixed` là cặp dùng cho ablation.

| Tệp | Nội dung |
| --- | --- |
| `grid_initial_dev.csv` | 9 cấu hình mỗi bậc, xếp hạng riêng cho từng mô hình |
| `grid_expanded_dev.csv` | Lưới chung mở rộng một vòng khi cấu hình dẫn đầu chạm biên |
| `selection.json` | Phiên bản dữ liệu, mã băm, quy tắc xếp hạng và cấu hình cho năm vai trò đánh giá |
| `metrics_dev.csv` | Năm metric của baseline, hai cấu hình tốt nhất và cặp ablation |
| `ablation_fixed_dev.csv` | Cặp cố định α = 0,1, cutoff = 1 và hiệu H2 − H1 |
| `per_tag_*.csv`, `confusion_*.csv` | Precision, recall, F1, support và ma trận theo 17 nhãn cho từng vai trò. Hàng là gold, cột là dự đoán. |

Xếp hạng lần lượt theo Token Accuracy giảm dần, Macro-F1 giảm dần, cutoff tăng dần
và α giảm dần. Khi Token Accuracy bằng nhau, Macro-F1 được coi là hòa trong
sai số 1e-12 so với giá trị tốt nhất của nhóm đang xét. Macro-F1 trong pipeline này tính trên
nhãn có gold support trong split. Nhãn chỉ xuất hiện trong dự đoán vẫn ảnh hưởng
đến các lỗi của nhãn gold.
Lưới ban đầu là α ∈ {0,01; 0,1; 1} và cutoff ∈ {0, 1, 2}. Nếu cấu hình dẫn đầu
của một trong hai mô hình chạm α thấp, α cao hoặc cutoff cao thì thêm lần lượt
0,001, 3 hoặc cutoff 3. Cả hai mô hình dùng cùng lưới mở rộng.

[Biểu đồ độ nhạy](../results/figures/dev/grid_sensitivity.png) dùng lưới mở rộng
12 cấu hình mỗi bậc. Bigram chạm biên α thấp ở vòng đầu. Sau khi thêm α=0,001
cho cả hai bậc, Bigram vẫn chọn α=0,01, cutoff=1 và Trigram chọn α=0,1,
cutoff=1. Vòng tròn đánh dấu các cấu hình được chọn.

```sh
python -m scripts.plot_dev_grid
```

Nếu lần chạy dev chỉ có lưới khởi đầu, thêm
`--grid results/tables/dev/grid_initial_dev.csv`.

Bootstrap trong `src/bootstrap.py` dùng cùng mẫu câu có hoàn lại cho mỗi cặp,
gom token rồi tính chênh lệch accuracy và lấy phân vị 2,5% và 97,5% trên 2.000
lần lấy mẫu với seed 42. Ba chênh lệch đã định nghĩa là Bigram tốt nhất trừ
baseline, Trigram tốt nhất trừ Bigram tốt nhất, và Trigram cố định trừ Bigram
cố định. Việc chạy đánh giá cuối theo cấu hình đã khóa nằm ở quy trình tích hợp.
