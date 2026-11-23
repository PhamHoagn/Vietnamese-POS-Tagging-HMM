# TV2 Giai đoạn 1 Bigram HMM

Tài liệu này ghi bàn giao giai đoạn 1. Trạng thái triển khai mới hơn và benchmark
nằm trong [TV2_TRIGRAM.md](TV2_TRIGRAM.md).

## Phạm vi và dữ liệu

Bigram đã triển khai counts, Add-alpha transition/emission, UNK, Viterbi log-space,
`log_joint`, cache và lưu/đọc cấu hình JSON. Trigram và benchmark trên dev thật
thuộc giai đoạn sau. Các tests của TV2 là self-check, không thay oracle độc lập
và normalization tests nghiệm thu do TV1 viết.

Tiếp tục dùng parser và kiểu dữ liệu của repo:

```python
Sentence = tuple[list[str], list[str]]  # (tokens, UPOS tags)
Corpus = list[Sentence]
```

Không đổi sang `list[tuple[word, tag]]`, vì repo đã có parser của TV1. Tokens
đã được tách sẵn; FORM có dấu cách vẫn là một token. Không đổi parser, baseline,
metrics hoặc grid search của thành viên khác. HMM chỉ fit trên corpus được truyền
vào `fit`; không có đường dẫn tải hoặc tự đọc dev/test trong model.

## API và tương thích

```python
from src.hmm_bigram import BigramHMMTagger, BigramHMM
from src.hmm_common import START, END, UNK, HMMTrainingCache

# API cũ cho pipeline và model factory hiện tại:
tagger = BigramHMMTagger(alpha=0.1, cutoff=1)
tagger.fit(train_data)
tags = tagger.predict(tokens)

# API mới theo yêu cầu TV2, vẫn nhận Corpus của parser hiện có:
model = BigramHMM()
model.fit(train_data, alpha=0.1, cutoff=1)
predictions = model.predict_corpus(word_lists)
score = model.log_joint(tokens, supplied_tags)
```

`fit` trả chính model, refit thay trạng thái cũ; fit lỗi giữ nguyên model cũ.
`predict` và `log_joint` trước fit gây `RuntimeError`. Đổi alpha/cutoff trực tiếp
sau fit buộc fit lại, tránh config khác bảng xác suất đang dùng.

`BigramHMMTagger.predict([])` trả `[]` để tương thích contract; lớp mới
`BigramHMM.predict([])` gây `ValueError`. `log_joint([], [])` gây `ValueError`
ở cả hai lớp vì không định nghĩa score câu rỗng. Câu rỗng trong train bị bỏ;
train không có token gây `ValueError`. Batch dự đoán rỗng trả `[]` sau fit.
Đầu vào predict phải là `list[str]`, không nhận văn bản thô để tự tách từ.

`model.tags` là tuple tên UPOS xuất hiện trong train, sắp xếp tăng dần. Đây là
quy ước mới đã được người dùng xác nhận: tập giải mã S có thể ít hơn 17 nhãn;
evaluation vẫn dùng labels do TV3 quy định. START `<s>` và END `</s>` không phải
nhãn dự đoán. Các thuộc tính dùng hậu tố `_` là trạng thái đã fit.

## Counts và xác suất

Với `S = model.tags`, `K = len(S) + 1`:

```text
q(v | u) = (C(u,v) + alpha) / (sum_z C(u,z) + alpha*K)
z, v thuộc S + END; u thuộc S + START.
e(x | y) = (C_cutoff(y, map(x)) + alpha)
           / (C(y) + alpha*(len(V_cutoff) + 1))
V_cutoff = {word: frequency_train(word) > cutoff}
```

Một alpha hữu hạn > 0 dùng chung cho cả transition và emission. Cutoff là số
nguyên >= 0; count **<= cutoff** chuyển sang UNK. Cutoff=0 không có mẫu UNK train
nhưng vẫn dành một ô UNK và smoothing cho từ chưa thấy. Alpha 0.001/3.0 và cutoff=3
cũng được hỗ trợ, không hardcode lưới. Log-prob tính bằng float64 và `logaddexp`
để tránh tràn số với alpha rất lớn hoặc quá nhỏ.

Mỗi câu reset START; mỗi câu không rỗng có đúng một transition đến END. Các hàng
transition được chuẩn hóa trên S + END, kể cả START. Phương thức
`transition_probability(previous, following)` cũng trả uniform `1/K` nếu context
chưa thấy; không nhận END làm context hoặc nhãn ngoài S + END làm đích.
Không có smoothing backoff/interpolation sang một mô hình khác.

Các thuộc tính để TV1 kiểm tra:

| Thuộc tính | Ý nghĩa |
| --- | --- |
| `tags`, `tag_to_index_` | Thứ tự/chỉ số nhãn cố định |
| `word_counts_`, `tag_counts_` | Count từ raw train và count từng nhãn |
| `transition_counts_` | Hàng S + START, cột S + END; shape `(K,K)` |
| `emission_counts_` | Hàng S, cột các từ được giữ theo thứ tự tên rồi UNK |
| `transition_log_probs_`, `emission_log_probs_` | Bảng log-prob float64 tính trong fit |
| `train_vocab_` | Raw FORM train, dùng định nghĩa OOV, trước cutoff |
| `retained_vocab_` | Các từ có count > cutoff, chưa gồm UNK |
| `vocab_`, `word_to_index_` | Miền emission, gồm sentinel UNK và chỉ số tương ứng |
| `unk_map_`, `map_token(word)` | Mapping từ train và mapping khi suy luận |
| `training_fingerprint_` | SHA-256 corpus train theo thứ tự, bỏ câu rỗng |

`UNK` là sentinel Enum, không phải chuỗi `"<UNK>"`; FORM thật có tên `<UNK>` vẫn
được phân biệt. Dùng `UNK` import từ module khi kiểm tra emission hoặc mapping.
`emission_probability(word, tag)` nhận FORM hoặc sentinel UNK. Cached arrays và
mappings chỉ đọc để tránh một model vô tình sửa trạng thái của model khác.

## Viterbi và log_joint

```text
D_1(v) = log q(v | START) + log e(x_1 | v)
D_i(v) = log e(x_i | v) + max_u [D_(i-1)(u) + log q(v | u)]
Final = max_v [D_T(v) + log q(END | v)]
```

Vòng theo thời gian dùng ma trận NumPy `score[u,v]`, không tính log trong vòng
lặp. Emission của toàn câu được gather trước vòng lặp. Backpointer lưu u để
truy vết đúng T nhãn; time O(T*|S|^2), bộ nhớ backpointer O(T*|S|).

Hòa điểm chính xác ở float64 chọn toàn bộ chuỗi nhãn nhỏ nhất theo thứ tự từ
điển, đúng contract ban đầu. Chỉ chọn chỉ số nhãn cuối nhỏ nhất không đủ: hai
đường NOUN,VERB và VERB,NOUN có thể hòa. Decoder giữ thêm rank của prefix tối ưu
để xử lý đúng mà không sao chép cả prefix ở mỗi bước.

`log_joint(words, tags)` tính score từ counts và công thức Add-alpha ở trên,
cộng cả START và END. Hàm không gọi Viterbi hoặc dùng bảng log-prob của decoder.
Việc vét cạn các tags rồi so score với decoder là self-check hữu ích, nhưng
vẫn cần oracle TV1 dùng bảng xác suất/kỳ vọng độc lập.

## Cache cho TV3 tái sử dụng

```python
shared_cache = HMMTrainingCache()
first = BigramHMMTagger(0.1, 1, cache=shared_cache).fit(train_data)
second = BigramHMMTagger(1.0, 1, cache=shared_cache).fit(train_data)
repeat = BigramHMMTagger(0.1, 1, cache=shared_cache).fit(train_data)
reference = BigramHMMTagger(0.1, 1, use_cache=False).fit(train_data)
```

Raw counts dùng key hash train; emission counts/mapping dùng `(hash, cutoff)`;
log-prob Bigram dùng `(hash, alpha, cutoff)`. Cùng config chỉ chuẩn bị log-prob
một lần khi dùng cùng cache. Mỗi lần fit vẫn xác nhận dữ liệu đầu vào và tính hash
để phát hiện người gọi sửa corpus. Đổi corpus tạo entry riêng, không dùng lẫn
vocabulary. Cache mặc định thuộc từng model; chia sẻ tường minh như ví dụ trên
khi muốn tái sử dụng giữa các model. `use_cache=False` tính lại, không đọc/ghi
cache. Cache giữ trong RAM, không dùng global hoặc tự lưu ra đĩa; giữ lâu nhiều
corpus sẽ tốn bộ nhớ, nên giới hạn vòng đời cache theo một đợt thí nghiệm.

## Lưu và đọc cấu hình

```python
model.save_config("bigram_config.json")
restored = BigramHMM.load_config("bigram_config.json")
restored.fit(train_data)
```

JSON gồm schema, model, alpha, cutoff, thứ tự nhãn, version code `tv2-hmm-1` và
hash train. Đây là **config, không phải checkpoint chứa weights**: model load
chưa fit và không được predict ngay. Refit kiểm tra hash/train, tags và tham số
khớp config đã khóa; config/schema/version sai gây lỗi. Hash bao gồm thứ tự câu,
nên cùng nội dung nhưng đảo thứ tự cũng không được coi là đúng bản train đã khóa.
Các package versions/Git commit đầy đủ vẫn cần được lưu khi chạy thực nghiệm thật.

## Chạy lại trên Windows hoặc Colab CPU

Tại repo root, dùng Python của môi trường đã cài `requirements.txt`:

```sh
python -m pytest -q tests/test_hmm_bigram.py tests/test_contract.py
python -m pytest -q
python -m scripts.smoke_test
```

Trên Windows của phiên làm việc này, dùng `.\.venv\Scripts\python.exe` thay
`python`. Nếu thư mục temp hệ thống bị lỗi quyền, thêm
`--basetemp=.venv/tmp/tv2-tests` vào lệnh pytest; thư mục này chỉ chứa file tạm.

Toy corpus trong `tests/toy_corpus.py` có 4 câu, 2 nhãn, bảng count và xác suất
kỳ vọng bằng phân số. Ví dụ alpha=1, cutoff=1, câu `[a,b]` tối ưu `[NOUN,VERB]`,
joint probability gồm EOS là `4/63`. Đây là fixture kiểm tra kỹ thuật, không phải
kết quả thực nghiệm hoặc cấu hình được chọn trên dev.

Kết quả local ngày 2026-11-20: Bigram + contract **107 passed**; sau rà soát input
và cập nhật tài liệu, toàn bộ suite **138 passed in 1.32s** (exit 0), smoke test
**PASS** (exit 0), `git diff --check` sạch. Môi trường Python 3.14.7, NumPy 2.5.5;
pytest dùng basetemp mới trong `.venv/tmp` để tránh lỗi quyền temp hệ thống.
Smoke chỉ kiểm tra baseline/integration, còn kiểm thử Bigram nằm trong suite riêng.

## Trạng thái sau tích hợp

- Oracle độc lập của TV1, kiểm thử chuẩn hóa và các ca biên chạy trong suite
  của dự án. [Ghi chú tích hợp](INTEGRATION_TV1_TV2.md)
- Trigram NumPy đã được tích hợp. Các phép đo CPU trên 100 câu dev và mức
  dự trù cho 9 cấu hình nằm trong [tài liệu Trigram](TV2_TRIGRAM.md).
- Metric, tie-break, lưới dev và bootstrap đã đồng bộ với kế hoạch v4.1.
  [Quy trình đánh giá](TV3_EVALUATION_RUNBOOK.md)
- Notebook đã chạy với kernel Jupyter sạch trên Windows. Test chính thức
  chờ xác nhận khóa. [Biên bản](pre_register_test.md)
