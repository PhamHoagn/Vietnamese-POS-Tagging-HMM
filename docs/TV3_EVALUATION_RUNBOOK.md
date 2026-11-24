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

### Ví dụ từ đa nghĩa trên dev

FORM `bàn` xuất hiện 3 lần trong train và cả 3 mang nhãn VERB. Trên dev,
cả 7 lần đều mang nhãn NOUN. B0 và Bigram best đúng 0/7. Trigram best,
Bigram fixed và Trigram fixed đều đúng 2/7 ở `dev-16` và `dev-900`.
Demo cho mở `dev-804` (cả ba mô hình best đều sai ở `bàn`) và
`dev-900` (Trigram best đúng ở `bàn`), với nhãn đọc từ file dev đã kiểm
mã băm. Đây là hai ví dụ được chọn để minh họa, còn các tỷ lệ 0/7 và 2/7
đếm trên toàn bộ lần xuất hiện của FORM này trong dev.

Bootstrap trong `src/bootstrap.py` dùng cùng mẫu câu có hoàn lại cho mỗi cặp,
gom token rồi tính chênh lệch accuracy và lấy phân vị 2,5% và 97,5% trên 2.000
lần lấy mẫu với seed 42. Ba chênh lệch đã định nghĩa là Bigram tốt nhất trừ
baseline, Trigram tốt nhất trừ Bigram tốt nhất, và Trigram cố định trừ Bigram
cố định. Lần chạy chính thức dùng [biên bản khóa](pre_register_test.md).

## Vì sao Bigram thấp hơn B0 trên dev?

Chia 26.162 token dev theo số lần FORM xuất hiện trong train. B0 chọn nhãn
phổ biến nhất của từng FORM. Bigram dùng cấu hình đã chọn trên dev (α = 0,01,
cutoff = 1). Cột cuối lấy chênh lệch số token đúng của nhóm chia cho **toàn bộ**
26.162 token dev và nhân 100.

| Tần suất trong train | Token dev | B0 đúng | Bigram đúng | Chênh lệch số đúng | Đóng góp vào Accuracy chung (điểm %) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ít nhất 2 lần | 19.737 | 89,72% | 90,51% | +155 | +0,592 |
| Đúng 1 lần | 1.596 | 89,85% | 50,56% | −627 | −2,397 |
| Chưa gặp (OOV) | 4.829 | 43,74% | 41,02% | −131 | −0,501 |
| **Tổng** | **26.162** | **81,24%** | **78,94%** | **−603** | **−2,305** |

Nhóm xuất hiện đúng một lần chiếm 6,10% token dev nhưng tạo ra phần hụt lớn
nhất. Với cutoff = 1, Bigram thay cả những từ này bằng `<UNK>` và mất thông tin
về chính từ đã thấy. Với cutoff = 0 ở cùng α, Bigram đúng 89,29% trên nhóm này
nhưng chỉ đúng 13,17% trên OOV. Khi đó `<UNK>` có số đếm bằng 0 cho mọi nhãn.
Sau Add-α, mẫu số nhỏ của nhãn ít gặp có thể khiến nhãn đó được ưu tiên.
Đây là phân rã kết quả dự đoán. Tính đúng của Viterbi được kiểm tra riêng bằng
oracle vét cạn.

### Thử nghiệm bổ sung chỉ trên dev

Giữ mọi FORM đã thấy, giữ nguyên số đếm chuyển tiếp và số đếm từ–nhãn. Với mỗi
nhãn, thêm vào cột `<UNK>` một lượt đếm cho mỗi FORM chỉ xuất hiện một lần với
nhãn đó trong train. Sau đó cộng α và chuẩn hóa lại **toàn bộ hàng phát xạ**,
kể cả cột `<UNK>`. Chỉ FORM chưa gặp mới đi qua cột này lúc giải mã.

| Mô hình | α | Accuracy dev gốc | Accuracy dev với phát xạ mới | OOV Accuracy với phát xạ mới |
| --- | ---: | ---: | ---: | ---: |
| Bigram | 0,01 | 78,94% | 81,2361% | 41,00% |
| Trigram | 0,1 | 78,59% | 80,7201% | 43,26% |

Biến thể Bigram đúng 21.253 token, B0 đúng 21.255. So với Bigram gốc, biến thể
đúng thêm 625 token ở nhóm xuất hiện một lần, đúng ít hơn 23 token ở nhóm xuất
hiện ít nhất hai lần và ít hơn 1 token OOV. Phần cải thiện đến chủ yếu từ việc
giữ lại từ đã thấy. Kết quả này là phân tích bổ sung sau khi khóa cấu hình.

Tái lập từ train/dev bằng lệnh dưới đây. CSV chỉ chứa số liệu tổng hợp.
[Bảng số đếm và đóng góp của từng nhóm](../results/tables/diagnostics_dev_unk.csv)

```sh
python -m scripts.diagnose_dev_unk --data data/raw/ud-vtb-fa7130860b65
```

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
