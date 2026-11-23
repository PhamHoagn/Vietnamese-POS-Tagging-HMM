# Giải tay Viterbi và chuẩn bị bảo vệ TV2

## Ví dụ tính tay

Train toy gồm bốn câu có nhãn, lấy từ [`tests/toy_corpus.py`](../tests/toy_corpus.py). Chọn `α = 1`, `cutoff = 1` để các phép tính ra phân số đơn giản. Ký hiệu `N = NOUN`, `V = VERB`, `s = <s>` và `E = </s>`.

| Câu train | Nhãn |
| --- | --- |
| `a b` (hai lần) | `N V` |
| `c` | `N` |
| `b` | `V` |

`a` xuất hiện 2 lần, `b` 3 lần, `c` 1 lần. Vì vậy `c` ánh xạ sang UNK và miền phát xạ là `{a, b, UNK}`. Mỗi nhãn có 3 token train nên mẫu số của mỗi hàng phát xạ là `3 + 1 × 3 = 6`. Chẳng hạn `e(a|N) = (2+1)/6 = 1/2` và `e(b|V) = (3+1)/6 = 2/3`.

| Nhãn | `e(a)` | `e(b)` | `e(UNK)` |
| --- | ---: | ---: | ---: |
| N | 1/2 | 1/6 | 1/3 |
| V | 1/6 | 2/3 | 1/6 |

Tập đích chuyển tiếp là `{N, V, E}`. Mỗi hàng thêm 1 vào ba ô rồi chia cho `tổng count + 3`:

| Lịch sử Bigram | Đến N | Đến V | Đến E |
| --- | ---: | ---: | ---: |
| `s` | 4/7 | 2/7 | 1/7 |
| `N` | 1/6 | 1/2 | 1/3 |
| `V` | 1/6 | 1/6 | 2/3 |

| Lịch sử Trigram | Đến N | Đến V | Đến E |
| --- | ---: | ---: | ---: |
| `(s,s)` | 4/7 | 2/7 | 1/7 |
| `(s,N)` | 1/6 | 1/2 | 1/3 |
| `(s,V)` | 1/4 | 1/4 | 1/2 |
| `(N,V)` | 1/5 | 1/5 | 3/5 |
| Cặp nhãn chưa thấy `(N,N)`, `(V,N)`, `(V,V)` | 1/3 | 1/3 | 1/3 |

### Giải mã câu a b b

Các bảng dưới dùng miền xác suất để dễ tính tay. Thứ tự đường đi được giữ khi chuyển sang log. Chữ sau mũi tên là nhãn tiền nhiệm cần lưu trong backpointer.

| Bước Bigram | Kết thúc tại N | Kết thúc tại V |
| --- | ---: | ---: |
| 1 · `a` | 2/7 | 1/21 |
| 2 · `b` | 1/126 ← N | 2/21 ← N |
| 3 · `b` | 1/378 ← V | 2/189 ← V |
| Sau khi nhân xác suất đến E | 1/1134 | **4/567** |

Bigram chọn V ở bước 3. Truy ngược được V ở bước 2 và N ở bước 1, nên kết quả là `N V V`. Ví dụ ô `D₃(V) = max((1/126)·(1/2), (2/21)·(1/6))·(2/3) = 2/189`.

Trigram khởi tạo bước 1 tại `(s,N)` và `(s,V)` với điểm `2/7` và `1/21`. Bước 2 tạo đủ bốn cặp nhãn thực. Bước 3, mỗi cặp `(v,w)` chọn nhãn `u` tối ưu từ cặp trước `(u,v)`.

| Cặp cuối | Bước 2 · `b` | Bước 3 · `b` | Sau khi nhân xác suất đến E |
| --- | ---: | ---: | ---: |
| `(N,N)` | 1/126 | 1/2268 ← N | 1/6804 |
| `(N,V)` | 2/21 | 1/567 ← N | 1/945 |
| `(V,N)` | 1/504 | 1/315 ← N | 1/945 |
| `(V,V)` | 1/126 | 4/315 ← N | **4/945** |

Ở ô `(V,V)` bước 3, so `(2/21)·(1/5)` từ `(N,V)` với `(1/126)·(1/3)` từ `(V,V)`, rồi nhân `e(b|V) = 2/3`. Cặp thắng ở bước cuối là `(V,V)`, backpointer cho `u = N`, nên Trigram cũng trả `N V V`. Điểm log đồng thời có EOS của Bigram là `ln(4/567) ≈ −4,954065` và của Trigram là `ln(4/945) ≈ −5,464891`. Hai điểm này thuộc hai phân phối khác nhau. So sánh chất lượng mô hình bằng các dự đoán trên cùng tập dev.

| Ca một token `a` có nhãn N | Công thức có EOS | Điểm |
| --- | --- | ---: |
| Bigram | `q₁(N|s) · e(a|N) · q₁(E|N)` | 2/21 |
| Trigram | `q₂(N|s,s) · e(a|N) · q₂(E|s,N)` | 2/21 |

## Demo và cách chạy lại

Từ thư mục gốc repo, với Python 3.11 trở lên:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.demo_tv2
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.venv/tmp/tv2-review
.\.venv\Scripts\python.exe -m scripts.smoke_test
```

Demo lấy hai câu train toy và in Gold/B0/Bigram/Trigram trên câu giữ lại. Khi trình bày, chỉ vào token `học` chưa xuất hiện trong train. Gold dùng để đối chiếu sau khi dự đoán. Trong Colab, dùng `python -m scripts.demo_tv2` từ thư mục gốc của checkout sau khi cài `requirements.txt`. Lệnh lấy train/dev đúng commit, kiểm tra SHA-256 và chạy benchmark nằm trong [hướng dẫn Trigram](TV2_TRIGRAM.md).

Phần [Phương pháp dạng Word](../report/TV2_Phuong_phap_HMM.docx) có công thức sửa được bằng Word. Lần nghiệm thu ngày 23/11 trên Windows gồm 875 tests, 3 subtests và smoke test đều đạt. Đo 100 câu dev với hệ số an toàn 1,5 cho thời gian dự phóng 3,62 giây với 9 cấu hình Bigram và 13,75 giây với 9 cấu hình Trigram. Đây là phép đo tốc độ decoder trên CPU local.

Để nối với thực nghiệm, TV3 có thể tạo một `HMMTrainingCache` rồi truyền cùng đối tượng đó vào các lần khởi tạo `BigramHMMTagger(alpha=..., cutoff=..., cache=cache)` và `TrigramHMMTagger(alpha=..., cutoff=..., cache=cache)`. Mỗi model gọi `fit(train_data)` và dùng `predict_corpus` trên danh sách token dev. `train_vocab_` là nguồn dùng cho metric OOV.

## Kịch bản trình bày trong ba phút

1. **40 giây.** Giới thiệu đầu vào là câu đã tách token. B0 tra theo FORM, còn HMM tìm cả chuỗi nhãn. Vẽ `s → y₁ → … → yₜ → E` và thêm một `s` cho Trigram.
2. **70 giây.** Trình bày `e(b|V) = 2/3`, một hàng chuyển tiếp và ý nghĩa của EOS trong mẫu số. Giải ô `(V,V)` trong bảng rồi lần backpointer để lấy `N V V`.
3. **45 giây.** Chạy demo, đối chiếu các nhãn với Gold. Giải thích OOV được tính theo FORM trước cutoff, còn từ hiếm đã thấy vẫn có thể đi vào ô phát xạ UNK.
4. **25 giây.** Nêu bằng chứng kiểm thử tối ưu bằng oracle ngắn, rồi tách riêng bằng chứng tốc độ. Dùng báo cáo kết quả của nhóm cho các kết luận chất lượng cuối.

## Câu hỏi phản biện

| Câu hỏi | Câu trả lời ngắn |
| --- | --- |
| Vì sao thêm EOS có thể làm đổi nhãn cuối? | Tổng điểm cuối cộng thêm log xác suất kết thúc của từng trạng thái. Một trạng thái có điểm prefix cao hơn vẫn có thể thua sau khi tính EOS. Repo có test riêng cho ca một token. |
| Vì sao Trigram dùng hai START và backpointer lưu một nhãn? | Hai START cung cấp đủ lịch sử cho token đầu. Khi biết cặp cuối `(v,w)` và tiền nhiệm `u`, ta lùi được về `(u,v)`. |
| Vì sao Trigram tốn `O(TS³)`? | Mỗi vị trí có `S²` cặp đích, mỗi cặp xét `S` nhãn tiền nhiệm. NumPy xử lý đồng thời tensor `score[u,v,w]` trên trục `u`. |
| Khi ngữ cảnh chưa xuất hiện, Add-α cho kết quả gì? | Mỗi đích trong `S ∪ {EOS}` có `α / (α · (|S|+1))`. Hàng đó là phân phối đều và vẫn gồm EOS. |
| `cutoff=0` còn dự đoán được từ mới không? | Có. Ô UNK vẫn nằm trong miền phát xạ. Count UNK trên train bằng 0 và Add-α cấp xác suất dương. |
| Vì sao tối đa `log_joint` tìm đúng chuỗi nhãn? | Với câu quan sát cố định, mẫu số của xác suất hậu nghiệm không đổi theo chuỗi nhãn. Log bảo toàn thứ tự và đổi phép nhân thành phép cộng. |
| Kiểm chứng tính đúng và chống rò rỉ thế nào? | TV1 vét cạn chuỗi nhãn ngắn từ bảng xác suất riêng. Test chuẩn hóa đối chiếu counts tính tay. Test leakage kiểm tra counts và từ vựng trước và sau dự đoán trên tập toy. |
| Vì sao Trigram có thể kém Bigram? | Nhiều lịch sử cặp nhãn hơn làm count trên mỗi hàng thưa hơn. Lần chạy dev với `α=0,1`, `cutoff=1` cho Trigram thấp hơn Bigram 0,29 điểm phần trăm Token Accuracy. Độ thưa là một giả thuyết cần kiểm tra bằng số liệu sparsity và đối chứng của nhóm. |
