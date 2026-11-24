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
| [`runtime.json`](../results/tables/dev/runtime.json) | Lần đo lại toàn bộ lệnh dev từ tải file local đến xuất bảng kết quả. Grid 9→12 cấu hình mỗi bậc chạy trong 31,16 giây trên Windows. |

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
cố định. Lần chạy chính thức dùng [biên bản khóa](pre_register_test.md).

## Đối chiếu kết quả Test

Test có 800 câu, 11.692 token và 1.747 token OOV (14,94%). Ma trận nhầm lẫn và
bảng theo nhãn của cả năm vai trò dùng cùng tập gold, đủ 17 UPOS.
[Bảng chỉ số đầy đủ](../results/tables/test/metrics_test.csv) · [Metadata](../results/tables/test/run_metadata.json)

| Cấu hình | Accuracy | Macro-F1 | OOV Accuracy | Accuracy bỏ PUNCT |
| --- | ---: | ---: | ---: | ---: |
| B0 | 81,79% | 71,74% | 34,29% | 78,69% |
| Bigram best | 81,72% | 71,11% | 38,47% | 78,67% |
| Trigram best | 81,73% | 69,19% | 42,47% | 78,66% |
| Bigram fixed | 81,54% | 70,30% | 38,12% | 78,44% |
| Trigram fixed | 81,73% | 69,19% | 42,47% | 78,66% |

| Chênh lệch Accuracy | Ước lượng (điểm %) | CI 95% (điểm %) |
| --- | ---: | ---: |
| Bigram best − B0 | −0,068 | [−0,652; +0,531] |
| Trigram best − Bigram best | +0,009 | [−0,511; +0,511] |
| Trigram fixed − Bigram fixed | +0,188 | [−0,330; +0,687] |

Cả ba CI chứa 0. Trigram best đúng thêm 70 token OOV so với Bigram best
(742 so với 672) và đúng ít hơn 69 token đã thấy trong train (8.814 so với
8.883). Hai thay đổi gần như triệt tiêu ở Accuracy chung. B0 có Macro-F1 cao nhất
trong các cấu hình này. Khi bỏ PUNCT, Accuracy của B0 và hai HMM best nằm trong
khoảng 78,66–78,69%. [CI gốc](../results/tables/test/paired_bootstrap_test.csv)

Ở cặp cố định, Trigram − Bigram là −0,290 điểm % trên dev và +0,188 điểm %
trên Test. CI Test chứa 0. [Bảng ablation dev](../results/tables/dev/ablation_fixed_dev.csv)

Kiểm tra mô tả bằng chuỗi nhãn gold cho thấy 21/12.492 chuyển tiếp Bigram
và 359/12.492 chuyển tiếp Trigram trên Test chưa từng xuất hiện trong train,
tính cả EOS. Với Trigram, 21 bước có cả lịch sử hai nhãn chưa từng thấy, khi đó
Add-α cho xác suất đều trên các nhãn đích và EOS. Đây là thống kê mô tả độ thưa
trên chuỗi gold. Cần truy vết riêng trên đường giải mã để gắn một lỗi cụ thể
với lịch sử của decoder. [Audit dữ liệu và lỗi](TV1_DATA.md)
