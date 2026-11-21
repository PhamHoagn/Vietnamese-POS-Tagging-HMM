# TV2 Giai đoạn 2 Trigram HMM và benchmark CPU

Sau checkpoint giai đoạn 2 `0ab2285`, TV1 đã được tích hợp trên nhánh feature;
xem [biên bản tích hợp](INTEGRATION_TV1_TV2.md). Main chưa được merge.

## Trạng thái và phạm vi

Trigram đã có supervised counts, Add-alpha, Viterbi NumPy theo cặp nhãn,
`log_joint` tính từ counts, batch predict, save/load config và cache. Parser,
baseline/EDA nhận nguyên bản từ TV1 khi tích hợp; metrics và grid search giữ nguyên.
Benchmark chỉ đo tốc độ trên train/dev; không dùng test, không chấm chất lượng
hoặc chọn alpha/cutoff. Đây chưa phải nghiệm thu oracle độc lập của TV1.

## API dùng cùng định dạng Corpus

```python
from src.hmm_trigram import TrigramHMMTagger, TrigramHMM
from src.hmm_common import HMMTrainingCache, START, END, UNK

shared = HMMTrainingCache()
tagger = TrigramHMMTagger(alpha=0.1, cutoff=1, cache=shared).fit(train_data)
tags = tagger.predict(tokens)
predictions = tagger.predict_corpus(word_lists)
score = tagger.log_joint(tokens, supplied_tags)

model = TrigramHMM()
model.fit(train_data, alpha=0.1, cutoff=1)
```

`train_data` vẫn là `list[tuple[list[str], list[str]]]`, không phải dataclass
Sentence mới bên nhánh TV1. Dùng `load_conllu()` hiện có; khi tích hợp `read_conllu`
thì chuyển `(sentence.tokens, sentence.tags)` ở lớp kết nối.

`fit` trả self, chỉ học dữ liệu được truyền vào. Tập nhãn S là UPOS có trong train,
sắp theo tên; cùng raw vocabulary, cutoff và emission với Bigram.
`TrigramHMMTagger.predict([])` trả `[]`; lớp mới `TrigramHMM.predict([])` báo
`ValueError`. `log_joint([], [])` báo `ValueError`. Không dự đoán/chấm trước fit.
Đổi config sau fit buộc fit lại. Refit lỗi giữ model cũ.

## Counts và smoothing bậc hai

```text
q2(w | u,v) = (C(u,v,w) + alpha) / (sum_z C(u,v,z) + alpha*(|S|+1))
z,w thuộc S + END
e(x | y) = (C_cutoff(y,map(x)) + alpha) / (C(y) + alpha*(|V_cutoff|+1))
```

Mỗi câu bắt đầu bằng lịch sử `(START,START)`. Kết thúc đúng một lần bằng END;
câu một token kết thúc ở lịch sử `(START,y1)`. `transition_counts_` có shape
`(|S|+1, |S|+1, |S|+1)`: hai trục đầu là S + START, trục cuối là S + END.
START và END dùng chung chỉ số biên nhưng khác vai trò theo trục; không sinh
START hoặc END làm nhãn token.

Mẫu số `C(u,v)` tính cả transition END. Với mọi lịch sử chưa thấy, hàng là
uniform `1/(|S|+1)`; không lùi về Bigram. Một alpha hữu hạn > 0 dùng chung với
emission, cutoff giữ từ có count train > k, và luôn có đúng một ô UNK.
Sentinel UNK không va chạm FORM thật `<UNK>`.

`transition_probability(earlier, previous, following)` nhận ba nhãn/context;
khác Bigram nhận hai. Hàng chưa thấy có xác suất uniform. Các thuộc tính counts,
emission/vocabulary, `map_token` và config giống Bigram, chỉ khác tensor chuyển.

## Tensor Viterbi và backtracking

```text
D1(v) = log q2(v | START,START) + log e(x1 | v)
D2(v,w) = D1(v) + log q2(w | START,v) + log e(x2 | w)
Di(v,w) = log e(xi | w) + max_u [D(i-1)(u,v) + log q2(w | u,v)]
Final = max_(v,w) [DT(v,w) + log q2(END | v,w)]
```

Câu một token dùng riêng `max_v D1(v) + log q2(END | START,v)`.
Với câu dài, tạo tensor `score[u,v,w]`, lấy max/argmax theo trục u bằng NumPy.
Backpointer `[i,v,w]` lưu u; truy vết lùi bằng cách chuyển `(v,w)` thành `(u,v)`.
Không có ba vòng for Python lồng nhau, không gọi log trong vòng theo thời gian;
log transition được cache lúc fit, log emission cả câu được gather trước decode.

Thời gian O(T*|S|^3), backpointer O(T*|S|^2), tensor tạm O(|S|^3). Rank prefix
đảm bảo hòa điểm chính xác ở float64 chọn toàn chuỗi nhỏ nhất theo từ điển, nhất
quán Bigram. `log_joint` tính trực tiếp từ counts và công thức, không gọi decoder
hoặc đọc cached log tables của decoder.

Trigram kế thừa vòng đời fit, xử lý emission, vocabulary và config từ lớp Bigram;
override preparation, transition, Viterbi và log_joint. Việc tái sử dụng code này
không có nghĩa dùng backoff Bigram trong phân phối Trigram.

## Cache và config

`HMMTrainingCache` đếm cả Bigram/Trigram trong một lần duyệt train. Cùng cutoff
tái sử dụng emission counts/mapping; cùng alpha/cutoff tái sử dụng chính bảng log
emission giữa hai bậc. Trigram log transition có cache riêng theo hash train và
config; `use_cache=False` cho kết quả giống hệt nhưng tính lại. Không dùng global
cache hoặc thêm vocabulary lúc predict.

```python
model.save_config("trigram_config.json")
restored = TrigramHMM.load_config("trigram_config.json").fit(train_data)
```

Config có `model="trigram"`, version `tv2-trigram-1`, alpha/cutoff, tags và SHA-256
train. Không chứa weights, nên cần fit lại; từ chối load config khác bậc hoặc
refit bằng corpus khác. Bigram giữ version config cũ `tv2-hmm-1`.

## Benchmark trên train và dev thật

Nguồn: [UD Vietnamese-VTB](https://github.com/UniversalDependencies/UD_Vietnamese-VTB),
ghim commit `fa7130860b65db9db342542c51dc57badd30fa9e`.
[README tại commit](https://github.com/UniversalDependencies/UD_Vietnamese-VTB/blob/fa7130860b65db9db342542c51dc57badd30fa9e/README.md)
ghi giấy phép CC BY-SA 4.0. Bản sao local train/dev và README/LICENSE nằm trong
`.venv/benchmark_data/ud-vtb-fa7130860b65/`, được Git ignore. Không đưa corpus gốc
hoặc văn bản câu mẫu vào commit. Không tải/parse file test.

| Split | File | Số câu đo được | Số token đo được | SHA-256 |
| --- | --- | ---: | ---: | --- |
| train | vi_vtb-ud-train.conllu | 1400 | 20215 | 1df180722db99b724f7368aca50597133bee28606de3394cd8d12a57da89978d |
| dev | vi_vtb-ud-dev.conllu | 1123 | 26162 | 0b779257b9225575b3b30e15d64cc4ef9945a77f8cceb9f92d794b483a2752dd |

Lấy 100 câu dev bằng các rank cách đều sau khi sắp theo (độ dài, vị trí gốc), gồm
câu ngắn nhất/dài nhất. Mẫu có 2357 token, độ dài 3–96, tập nhãn train đủ 17 UPOS.
Đo cả hai bậc, mỗi bậc 9 cấu hình alpha=(.01,.1,1), cutoff=(0,1,2); mỗi cấu hình
decode 3 lần và lấy median, warm-up riêng ngoài phép đo. Fit dùng cache chung.

CSV `results/tables/bench_dev.csv` ghi fit_seconds, decode median và seconds/token
thực tế. JSON cùng tên ghi môi trường, hash dữ liệu/code, chỉ số mẫu, config và
phép dự phóng:

```text
full_dev_seconds(config) = fit_seconds + seconds_per_token * total_dev_tokens
grid_seconds = 1.5 * sum(full_dev_seconds(config))
```

Đây là ước lượng trên máy Windows local, không phải thời gian đo full grid trên
Colab. Chưa gồm load dữ liệu, tính metric, xuất predictions CSV hoặc bootstrap CI.
Phải đo lại trên máy dùng chạy thực nghiệm nếu môi trường thay đổi; runtime gate
không thay gate tính đúng qua oracle độc lập.

Kết quả lần đo sau tích hợp ngày 2026-11-21, Python 3.14.7, NumPy 2.5.5, Windows 11:

| Phạm vi dự phóng trên toàn dev | Thời gian, đã nhân 1.5 |
| --- | ---: |
| 9 cấu hình Bigram | 3.45 giây |
| 9 cấu hình Trigram | 12.77 giây |
| Cả 18 cấu hình | 16.22 giây |

Runtime gate PASS so với 90 phút. Số chính xác theo từng cấu hình và hash code
đã chạy nằm trong CSV/JSON, không phải số giả định của kế hoạch.

Để chuẩn bị lại dữ liệu từ đầu trên Windows PowerShell, chạy tại repo root:

```powershell
$benchmarkRef = 'fa7130860b65db9db342542c51dc57badd30fa9e'
$benchmarkDir = Join-Path $PWD '.venv/benchmark_data/ud-vtb-fa7130860b65'
New-Item -ItemType Directory -Force -Path $benchmarkDir | Out-Null
$benchmarkFiles = @{
  'vi_vtb-ud-train.conllu' = '1df180722db99b724f7368aca50597133bee28606de3394cd8d12a57da89978d'
  'vi_vtb-ud-dev.conllu' = '0b779257b9225575b3b30e15d64cc4ef9945a77f8cceb9f92d794b483a2752dd'
}
foreach ($benchmarkName in $benchmarkFiles.Keys) {
  $benchmarkPath = Join-Path $benchmarkDir $benchmarkName
  $benchmarkUrl = "https://raw.githubusercontent.com/UniversalDependencies/UD_Vietnamese-VTB/$benchmarkRef/$benchmarkName"
  Invoke-WebRequest -Uri $benchmarkUrl -OutFile $benchmarkPath
  if ((Get-FileHash -LiteralPath $benchmarkPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $benchmarkFiles[$benchmarkName]) {
    throw "SHA-256 mismatch: $benchmarkName"
  }
}
```

Chạy lại khi đã có hai file ở đường dẫn trên, tại repo root:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_hmm `
  --train .venv/benchmark_data/ud-vtb-fa7130860b65/vi_vtb-ud-train.conllu `
  --dev .venv/benchmark_data/ud-vtb-fa7130860b65/vi_vtb-ud-dev.conllu `
  --model both --sample-size 100 --repeats 3 `
  --corpus-ref fa7130860b65db9db342542c51dc57badd30fa9e
```

Script benchmark nhận đường dẫn tùy chọn, không hardcode đường dẫn cá nhân.
`--model trigram` chỉ đo 9 cấu hình Trigram. Mã trả về 0 khi
gate tốc độ đạt, 2 khi vượt ngân sách; dữ liệu/đối số lỗi trả khác 0.

## Chạy lại kiểm thử và giới hạn

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.venv/tmp/tv2-tests
.\.venv\Scripts\python.exe -m scripts.smoke_test
```

Self-check TV2 gồm counts và phân số tính tay; chuẩn hóa mọi lịch sử/UNK;
vét cạn câu 1–3 token và 60 corpus train toy ngẫu nhiên seed 2026; score độc lập
với decoder; câu một nhãn, câu dài, OOV, cache và config roundtrip. Không coi các
self-check này là oracle nghiệm thu độc lập của TV1. Oracle TV1 hiện đã được
ghép/nối adapter và chạy thật; không thay các bảng kỳ vọng của TV1.
Chưa chạy runtime Colab thật, chưa tune/chọn mô hình bằng dev và chưa đánh giá test.

Kết quả checkpoint giai đoạn 2: **209 passed in 1.44s**. Sau tích hợp TV1:
**875 passed, 3 subtests passed in 3.69s**, exit 0; smoke **PASS**, exit 0.
Đã xác nhận SHA-256 của các file code trong metadata benchmark khớp bytes local.
Chỉ ghép vào nhánh feature theo chỉ dẫn người dùng, chưa merge main hoặc push.
