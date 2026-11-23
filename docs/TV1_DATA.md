# Dữ liệu và phân tích dev của TV1

Các bảng và hình được tạo từ UD Vietnamese-VTB tại commit `fa7130860b65db9db342542c51dc57badd30fa9e`. Corpus mang giấy phép CC BY-SA 4.0. Pipeline giữ nguyên FORM, phân biệt hoa thường, giữ khoảng trắng trong token và dùng nhãn UPOS từ CoNLL-U. Bản kê ghi URL, kích thước và SHA-256 đã kiểm tra của train/dev. Mục test chỉ ghi metadata Git của tệp đã ghim để đối chiếu sau khi khóa thí nghiệm. [Nguồn dữ liệu](https://github.com/UniversalDependencies/UD_Vietnamese-VTB/tree/fa7130860b65db9db342542c51dc57badd30fa9e), [giấy phép](https://github.com/UniversalDependencies/UD_Vietnamese-VTB/blob/fa7130860b65db9db342542c51dc57badd30fa9e/LICENSE.txt), [bản kê audit](../results/tables/tv1/audit.json).

## Chạy lại

Từ thư mục gốc dự án, có thể tải hai tệp đã ghim vào `data/raw/ud-vtb-fa7130860b65` và xuất bảng/hình bằng các lệnh sau.

```sh
python -m pip install -r requirements.txt
python -X utf8 -m scripts.prepare_tv1 --download
```

Nếu đã có dữ liệu local, truyền `--data-dir` để chạy ngoại tuyến. Sau khi có CSV dự đoán dev tổng hợp, dùng cùng lệnh và thêm `--dev-predictions`. Script kiểm đủ từng dòng qua `sent_id`, `idx`, FORM và nhãn vàng trước khi đếm lỗi. CSV dự đoán chứa token corpus nên lưu trong thư mục Git bỏ qua, ví dụ `.venv/project-runs/dev`.

```sh
python -X utf8 -m scripts.prepare_tv1 --data-dir data/raw/ud-vtb-fa7130860b65 --dev-predictions .venv/project-runs/dev/predictions_dev.csv
```

Notebook [NLP_POS.ipynb](../notebooks/NLP_POS.ipynb) chạy demo, grid dev, EDA, kiểm tra căn chỉnh dự đoán và hiển thị kết quả. Cell markdown cuối mô tả cách chạy đánh giá test sau khi nhóm khóa cấu hình.

Sau khi nhóm khóa cấu hình và `scripts.final_experiment run` hoàn tất, audit test dùng cùng `lock.json`, `run_metadata.json`, thư mục dữ liệu và CSV dự đoán nguyên gốc của lần chạy. Audit đối chiếu Git blob, SHA-256 và từng hàng dự đoán trước khi tạo bảng và hình tổng hợp cho cả ba split. Ví dụ lỗi chi tiết có FORM nằm trong thư mục riêng do `scripts.final_experiment run` tạo.

```sh
python -X utf8 -m scripts.audit_final_tv1 --lock results/tables/final_lock.json --run-metadata results/tables/test/run_metadata.json --data-dir data/raw/ud-vtb-fa7130860b65 --predictions .venv/project-runs/test/predictions_test.csv
```

Thay các đường dẫn ví dụ bằng đường dẫn của lần chạy đã khóa. Bảng tổng hợp mặc định ở `results/tables/tv1/final` và hình ở `results/figures/tv1/final`.

## Kiểm tra dữ liệu train và dev

| Split | Số câu | Token | FORM khác nhau | Độ dài trung bình | Độ dài tối đa |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1.400 | 20.215 | 3.870 | 14,44 | 25 |
| Dev | 1.123 | 26.162 | 4.818 | 23,30 | 96 |

Kiểm tra parser cho thấy số token khớp số hàng từ gốc ở cả hai split. Có đủ ID câu và mọi nhãn đọc được thuộc 17 UPOS. Cả hai tập không có dòng multiword, empty node hoặc khối chỉ chứa metadata trong phiên bản này. So trùng theo toàn bộ chuỗi FORM cho thấy 48 cặp nội bộ train, 2 cặp nội bộ dev và 2 cặp giao train/dev. Giữ nguyên split chuẩn và ghi các vị trí theo ID, chỉ số câu để nhóm tái kiểm tra. [Thống kê](../results/tables/tv1/eda.csv), [chỉ mục câu trùng](../results/tables/tv1/duplicates.csv).

Theo từ vựng train trước cutoff, **4.829/26.162 token dev là OOV (18,46%)**. Có 1.596 lượt token dev thuộc các FORM xuất hiện đúng một lần trong train và 2.664 lượt thuộc các FORM xuất hiện một hoặc hai lần. Các tỷ lệ OOV theo nhãn có mẫu số là số token dev của nhãn vàng tương ứng, phù hợp khi phân tích nhóm nhãn hiếm. [Bảng OOV](../results/tables/tv1/oov_stats.csv), [phân bố UPOS](../results/tables/tv1/upos_distribution.csv).

Ba biểu đồ dùng trực tiếp các bảng CSV. [Độ dài câu](../results/figures/tv1/sentence_lengths.png), [phân bố UPOS](../results/figures/tv1/upos_distribution.png), [OOV theo UPOS](../results/figures/tv1/oov_by_upos.png).

## Độ thưa của chuyển tiếp

Train có 310 FORM nhận từ hai nhãn trở lên, chiếm 5.437/20.215 lượt token (26,90%). Bigram quan sát được 234/324 ô của ma trận chuyển tiếp (72,22%). Trigram quan sát được 1.543/5.832 ô của tensor (26,46%). Mẫu số là số ô được cấp trong mảng theo cách mã hóa START/EOS của mô hình, có cả các ô lịch sử không đi qua trong giải mã thông thường.

Trên 27.285 bước chuyển của các chuỗi nhãn vàng dev, gồm START và EOS, có 41 chuyển tiếp Bigram mới (0,15%) và 643 chuyển tiếp Trigram mới (2,36%). Có 36 bước Trigram mang lịch sử hai nhãn chưa xuất hiện trong train (0,13%). Add-α cho phân phối đều trên nhãn đích và EOS tại các lịch sử chưa thấy. Các số này mô tả đường nhãn vàng dev, nên chỉ hỗ trợ việc đọc kết quả dự đoán và hình thành giả thuyết về sparsity. [Bảng sparsity](../results/tables/tv1/sparsity.csv).

## Phân tích 10 vị trí trên dev

So sánh B0 với H1_best (`α=0,01`, `k=1`) và H2_best (`α=0,1`, `k=1`). CSV dự đoán khớp toàn bộ 26.162 hàng dev theo ID, chỉ số token từ 0, FORM và nhãn vàng. Có 16.017 lỗi tính theo **cặp mô hình và token**, vì một token có thể sai ở nhiều mô hình. Danh sách dưới chọn theo nhóm cặp nhãn nhầm phổ biến, vì vậy phù hợp để minh họa mẫu lỗi. Dùng [bảng tổng hợp](../results/tables/tv1/dev_error_pairs.csv) khi cần tần suất và [SHA-256 dự đoán](../results/tables/tv1/dev_error_provenance.json) để tra bản chạy.

| Vị trí dev (chỉ số từ 0) | Gold | B0 | H1 | H2 | OOV | Giải thích theo ngữ cảnh |
| --- | --- | --- | --- | --- | --- | --- |
| `dev-1 / 0` | VERB | NOUN | NOUN | NOUN | Có | Động từ mở đầu một cụm chỉ hành động, có đối tượng theo sau. FORM đầu câu nằm ngoài từ vựng train và cả ba mô hình dồn về nhãn danh từ. |
| `dev-1 / 4` | PROPN | NOUN | NOUN | VERB | Có | Tên riêng đi sau đại từ xưng hô và làm chủ thể của động từ nói. B0/H1 nhận danh từ chung. H2 đưa nhầm động từ lên vị trí tên. |
| `dev-2 / 26` | NOUN | NOUN | VERB | VERB | Có | Token giữ vai trò danh từ trong cụm chỉ số lượng sau động từ chính. Cả hai HMM gán VERB dù cụm định lượng trước đó là tín hiệu cho cách dùng danh từ. |
| `dev-4 / 27` | NOUN | NOUN | NOUN | VERB | Có | Token là thành tố của cụm danh từ chỉ một công trình, đứng trước mã định danh. Chuyển sang VERB ở H2 làm gãy cụm này. |
| `dev-8 / 4` | ADJ | NOUN | ADJ | NOUN | Có | Tính từ đứng sau danh từ chỉ khoảng thời gian và bổ nghĩa độ dài. H1 khớp nhãn vàng trong ngữ cảnh này. |
| `dev-6 / 0` | VERB | NOUN | NOUN | NOUN | Có | Cùng chuỗi token với `dev-1`. Kết quả lặp lại cho thấy lỗi ổn định trên hai vị trí trùng trong dev, nên tính đây là một kiểu lỗi khi thảo luận. |
| `dev-12 / 0` | VERB | NOUN | NOUN | NOUN | Có | Động từ chỉ tư thế đứng đầu câu, trước cụm chỉ vị trí. Các mô hình đều ưu tiên NOUN cho FORM chưa gặp. |
| `dev-3 / 22` | PROPN | NOUN | VERB | NOUN | Có | Tên viết tắt trong cụm nói về người theo quốc tịch. Vị trí ngay trước cụm động từ kế tiếp khiến nhãn VERB của H1 dễ lẫn với hành động của chủ thể. |
| `dev-2 / 2` | PROPN | NOUN | NOUN | ADJ | Có | Tên đường theo sau danh từ chỉ loại công trình. B0/H1 giữ nhãn danh từ chung, còn ADJ của H2 đánh mất chức năng định danh của tên. |
| `dev-2 / 6` | NOUN | NOUN | NOUN | VERB | Không | Viết tắt chỉ địa danh theo nhãn vàng của corpus, FORM đã thấy trong train. Đây là ca H2 sai dù có từ vựng, cần xem cùng các chuyển tiếp quanh vị trí đó. |

Các diễn giải mô tả vai trò cú pháp được quan sát trong ngữ cảnh. Để quy một lỗi cụ thể cho emission hoặc chuyển tiếp, cần xem thêm điểm của các đường Viterbi cạnh tranh.
