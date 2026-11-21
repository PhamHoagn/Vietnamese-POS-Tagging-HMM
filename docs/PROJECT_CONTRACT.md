# PROJECT CONTRACT v0.1 — bản chờ cả nhóm review

## 1. Mục tiêu và phạm vi

Xây dựng hệ thống gán nhãn từ loại tiếng Việt bằng HMM Bigram, Trigram và Viterbi,
so sánh với Most Frequent Tag. Bản khởi tạo chỉ thiết lập API và kiểm tra kết nối;
không có kết quả thực nghiệm hoặc thuật toán HMM hoàn chỉnh.

Contract dưới đây là quy ước của code hiện tại. Cả ba thành viên cần xác nhận trước
khi lập trình song song; mọi thay đổi API chung phải cập nhật tài liệu và test.
Khung này được chuẩn bị trên branch `feature/hmm-viterbi`, sau đó được người dùng
duyệt để commit vào `main` local. Việc push lên remote cần chỉ dẫn riêng.

## 2. Cấu trúc thư mục

```text
src/
  __init__.py
  contracts.py          # kiểu dữ liệu, Protocol, UPOS, validation chung
  data_loader.py        # parser
  baseline.py           # MostFrequentTagger
  hmm_common.py         # counts train-only, UNK và cache của TV2
  hmm_tables.py         # nạp bảng xác suất cho oracle, không smoothing lại
  hmm_bigram.py         # Bigram HMM + Viterbi đã triển khai
  hmm_trigram.py        # Trigram HMM + Viterbi NumPy đã triển khai
  metrics.py            # 2 metric hoàn chỉnh, 3 metric dự phòng
  model_selection.py    # interface grid search, placeholder
tests/
  test_data_loader.py
  test_baseline.py
  test_metrics.py
  test_contract.py
  test_hmm_bigram.py    # TV2 tự kiểm tra; không thay oracle độc lập của TV1
  test_hmm_trigram.py
  test_benchmark_hmm.py
  test_adapters.py
  adapters.py
  oracle_bruteforce.py
  test_bruteforce.py
  test_normalization.py
  test_edge_cases.py
  toy_corpus.py         # dữ liệu và xác suất kỳ vọng tính tay
  fixtures/tiny.conllu
scripts/
  __init__.py
  smoke_test.py
  benchmark_hmm.py      # đo tốc độ train/dev, không chọn tham số theo metric
notebooks/
data/raw/
results/tables/
results/figures/
report/
docs/
  PROJECT_CONTRACT.md
  pre_register_test.md
README.md
requirements.txt
.gitignore
```

Hai module bổ sung so với đề xuất: `contracts.py` tránh lặp kiểu/tagset giữa các
thành viên; `model_selection.py` cung cấp interface grid search có thể import.
Các thư mục trống dùng `.gitkeep`. Không kèm corpus hoặc tài liệu bên thứ ba.

## 3. Kiểu dữ liệu

```python
Sentence = tuple[list[str], list[str]]  # (tokens, tags)
Corpus = list[Sentence]
```

Token lấy nguyên cột FORM, tag lấy cột UPOS. Giữ nguyên Unicode, hoa/thường và
khoảng trắng trong FORM, không tokenize hoặc normalize lại. Token phải là chuỗi
không rỗng. Mỗi câu phải có `len(tokens) == len(tags)`.

## 4. API chính xác

### `src.contracts`

- `Sentence`, `Corpus`, `UPOS_TAGS: tuple[str, ...]`.
- `POSTagger`: runtime-checkable `Protocol`, yêu cầu `fit` và `predict` bên dưới.
  Kiểm tra runtime chỉ xác nhận sự hiện diện của phương thức; test bổ sung kiểm tra
  signature và hành vi, không coi `isinstance` là bằng chứng thuật toán đã chạy.
- `validate_tokens(tokens: list[str]) -> None`.
- `validate_corpus(data: Corpus) -> None`.
- `validate_hmm_config(alpha: float, cutoff: int) -> tuple[float, int]`.
  Validation lỗi gây `ValueError`.

### `src.data_loader`

```python
load_conllu(path: str | pathlib.Path) -> Corpus
```

Đọc UTF-8 (chấp nhận BOM), đúng 10 cột phân cách bằng tab. Bỏ comment bắt đầu `#`,
range ID như `3-4` và empty node như `3.1`; chỉ lấy ID số nguyên dương.
Dòng trống ngăn câu; bỏ câu rỗng và nhận câu cuối dù thiếu dòng trống kết thúc.
Không split FORM theo dấu cách. Thiếu UPOS, ID lỗi hoặc số cột sai gây `ValueError`
kèm đường dẫn/số dòng; lỗi mở file giữ nguyên exception của Python.
Đây là bộ đọc FORM/UPOS, không phải validator toàn bộ cấu trúc phụ thuộc UD:
chưa kiểm tra ID liên tiếp, thứ tự ID, HEAD hoặc quan hệ dependency.

Sau tích hợp TV1, `read_conllu(path, nfc=False)` trả danh sách dataclass
`src.data_loader.Sentence` có `sent_id`, `tokens`, `tags`; không thay kiểu tuple
`src.contracts.Sentence`. `load_conllu` vẫn trả Corpus tuple cho HMM/metrics.
`load_splits(data_dir)` dùng MANIFEST.json và đọc cả ba split, chỉ dùng sau khóa
test; xác minh hash bằng công cụ TV1 trước khi sử dụng dữ liệu thật.

### Các tagger

```python
MostFrequentTagger()
BigramHMMTagger(alpha: float = 0.1, cutoff: int = 1)
TrigramHMMTagger(alpha: float = 0.1, cutoff: int = 1)

fit(train_data: Corpus) -> Self
predict(tokens: list[str]) -> list[str]
```

Quy ước khi đã triển khai: `fit` trả chính model, refit thay toàn bộ trạng thái;
train không có token gây `ValueError`; câu rỗng trong train không đóng góp count.
`predict` trước fit gây `RuntimeError`, kể cả đầu vào `[]`. Sau fit, `predict([])`
trả `[]`. Mỗi token nhận đúng một UPOS, không nhận gold và không cập nhật model
trong predict. Không sửa dữ liệu đầu vào. Hòa điểm dùng thứ tự tên tag tăng dần.

Baseline đã tuân thủ các quy ước này: chọn tag phổ biến nhất theo từng từ; từ lạ
nhận tag phổ biến nhất toàn train. Cả hai mức hòa count đều dùng tên tag tăng dần.
`train_vocab_: frozenset[str]` là tập FORM nguyên gốc của train. Refit thất bại
không làm mất model cũ.

Bigram đã triển khai interface trên và chỉ giải mã trong tập nhãn UPOS có ở train.
API bổ sung của TV2, cấu hình/cache và quy ước câu rỗng nằm trong
[TV2_BIGRAM.md](TV2_BIGRAM.md). `BigramHMMTagger.predict([])` vẫn trả `[]`;
lớp `BigramHMM` mới từ chối câu rỗng. Không thay parser hoặc kiểu Corpus hiện có.

Trigram đã triển khai cùng API, dùng hai START và transition bậc hai; xem
[TV2_TRIGRAM.md](TV2_TRIGRAM.md) về tensor, cache, save/load config và benchmark.
`TrigramHMMTagger.predict([])` trả `[]`; lớp mới `TrigramHMM` từ chối câu rỗng.
Hai HMM đã có self-check TV2 và đã nối oracle độc lập của TV1 qua
`tests/adapters.py`; xem [INTEGRATION_TV1_TV2.md](INTEGRATION_TV1_TV2.md).
API `from_probabilities` nạp bảng toy cho cùng decoder, giữ nguyên xác suất,
không thay hành vi fit hoặc dùng oracle tạo dự đoán.

### `src.metrics`

```python
evaluate_pos(
    gold_data: Corpus,
    predictions: list[list[str]],
    train_vocab: collections.abc.Set[str],
    labels: collections.abc.Sequence[str],
) -> POSMetrics
```

Số câu và số dự đoán mỗi câu phải khớp gold; lỗi gây `ValueError`. `labels` phải
khác rỗng, không trùng, là UPOS hợp lệ và chứa tất cả gold/predicted tags.
Dùng `UPOS_TAGS` cố định cho mọi phép so sánh chính, không suy ra labels từ test.
`train_vocab` nhận tập FORM nguyên gốc học từ train; evaluation không thể tự chứng
minh nguồn gốc tập này, người gọi chịu trách nhiệm chống leakage.

| Key | Trạng thái | Định nghĩa |
| --- | --- | --- |
| `token_accuracy` | Đã chạy | Tổng token đúng / tổng token, gộp mọi câu |
| `macro_f1` | Đã chạy | `f1_score(..., labels=labels, average="macro", zero_division=0)` |
| `oov_accuracy` | Chưa triển khai, trả `None` | Dự kiến: tỷ lệ đúng trên FORM không thuộc raw train_vocab |
| `oov_rate` | Chưa triển khai, trả `None` | Dự kiến: số token OOV / tổng token |
| `accuracy_no_punct` | Chưa triển khai, trả `None` | Dự kiến: accuracy trên token có gold UPOS khác PUNCT |

Các tag vắng ở cả gold/prediction vẫn đóng góp F1=0 trong macro theo labels cố định.
Evaluation rỗng trả 0.0 cho hai metric đã làm; ba metric chưa làm vẫn `None`.
Khi hoàn thiện metric OOV/no-punct, quy ước mẫu số bằng 0 trả 0.0 và báo số lượng
token tương ứng trong báo cáo để không diễn giải nhầm là chất lượng kém.

`predict` chỉ nhận `tokens`; gold chỉ được đưa vào evaluation sau khi đã có
predictions. Không dùng gold để xử lý UNK hoặc chọn đường đi Viterbi.

### `src.model_selection`

```python
grid_search_pos(
    model_factory: Callable[..., POSTagger],
    train_data: Corpus,
    dev_data: Corpus,
    param_grid: Mapping[str, Sequence[Any]],
    *,
    labels: Sequence[str] = UPOS_TAGS,
    scoring: str = "token_accuracy",
) -> list[dict[str, Any]]
```

Hiện luôn gây `NotImplementedError`. Hành vi tương lai cần giữ:

1. Duyệt tích Descartes của grid, tạo model mới cho mỗi cấu hình.
2. Chỉ `fit(train_data)`, predict dev bằng tokens, chấm bằng dev gold.
3. Trả bảng phẳng, mỗi cấu hình một dòng, có `param_<name>`,
   `dev_token_accuracy`, `dev_macro_f1`, `rank` (bắt đầu 1), `selected` (bool).
   Dùng `csv.DictWriter` để lưu, không bắt buộc pandas. Metric chưa làm bỏ khỏi bảng.
4. Chọn score dev cao nhất; hòa điểm dùng cặp `(alpha, cutoff)` tăng dần cho grid
   HMM hiện tại. Chỉ dùng metric đã triển khai làm scoring; chọn metric `None` phải lỗi.
5. Không nhận `test_data`, không đọc file test, không refit train+dev tự động.
   Grid/scoring/tie-break phải được khóa trước đánh giá test.

## 5. Tagset UPOS

17 nhãn cố định, theo thứ tự `UPOS_TAGS`:

```text
ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X
```

Không trả `_`, XPOS, UNK, BOS hoặc EOS làm nhãn dự đoán. Parser không âm thầm
chuyển tag sai sang X. Nếu corpus dùng tagset khác, TV1 phải đề xuất mapping công
khai trước khi đưa vào pipeline.

## 6. Alpha, cutoff, UNK và vocabulary

- `alpha` là Add-alpha smoothing, số hữu hạn > 0, không nhận bool.
- `cutoff=k` là số nguyên >= 0, không nhận bool. Từ có count train **<= k** ánh xạ
  sang UNK; cutoff=0 giữ mọi từ đã xuất hiện. Count trên token FORM, không trên câu.
- HMM tương lai lưu `train_vocab_` là raw FORM, và `vocab_` là vocabulary phát xạ
  sau cutoff gồm các từ được giữ và UNK. Hai tập có ý nghĩa khác nhau.
- Dùng ký hiệu UNK nội bộ không va chạm token thật; chuỗi FORM trùng tên hiển thị
  `<UNK>` vẫn phải phân biệt với sentinel. Luôn dành một ô UNK để nhận từ chưa thấy.
- Mọi từ ngoài vocabulary phát xạ chuyển sang UNK khi predict. Emission UNK học
  từ rare token trong train và smoothing, không dùng dev/test để bổ sung từ.
- Baseline không có alpha/cutoff và dùng fallback tag toàn train.
- OOV dùng **raw train_vocab**, nên rare word đã thấy trong train không phải OOV,
  dù HMM ánh xạ nó sang UNK.

## 7. Start/end-state và xác suất HMM (quy ước cho TV2)

Với HMM, S là các UPOS thực sự xuất hiện trong train, sắp theo tên tăng dần;
chỉ S có emission và được giải mã. 17 UPOS vẫn là tagset hợp lệ và labels cố định
của evaluation hiện tại. BOS/EOS là state biên, không thuộc labels đầu ra.
Mỗi câu reset riêng; không đếm chuyển tiếp xuyên qua hai câu.

Bigram thêm một BOS trước câu, một EOS sau câu; tính các chuyển tiếp
`BOS -> t1 -> ... -> tn -> EOS`. Điểm câu gồm emission mỗi token, transition
ban đầu và transition kết thúc. Trigram thêm hai BOS và một EOS;
tính `P(t_i | t_(i-2), t_(i-1))` từ context `(BOS, BOS)`, và cuối cùng
`P(EOS | t_(n-1), t_n)`; câu một token kết thúc bằng context `(BOS, t1)`.
Không phát xạ từ BOS/EOS, không chuyển tới BOS hoặc chuyển ra EOS.

Chuẩn hóa Add-alpha transition trên tập đích `S + EOS`, emission trên
vocabulary phát xạ gồm UNK, theo count của context/tag tương ứng. Context chưa
thấy vẫn có phân phối nhờ alpha. Câu rỗng bị bỏ khi train, predict trả `[]`.
TV2 dùng log tự nhiên và float64 để tránh underflow; hòa điểm chọn đường đi có
chuỗi tag nhỏ nhất theo thứ tự từ điển. TV1 kiểm chứng trên toy model bằng vét cạn
độc lập, đặc biệt câu một token và điểm kết thúc. Bigram đã có code và self-check
của TV2 cho cả hai bậc; oracle độc lập của TV1 đã tích hợp và chạy thành công.

## 8. Chống leakage và tái lập

- TV1 thống nhất corpus, phiên bản/checksum, giấy phép, split theo câu và seed;
  không shuffle/split lại sau khi biết kết quả test.
- Vocabulary, rare counts, emission, transition và fallback tag chỉ học từ train.
- Dev chọn alpha/cutoff; test không tham gia grid search, EDA định hướng tuning
  hoặc chọn seed. Không có test corpus trong smoke test.
- Không tự gộp train+dev khi chạy kết quả cuối. Nếu muốn gộp, phải sửa và khóa
  protocol trước khi xem test, đồng thời xác định lại train_vocab nhất quán.
- Khóa `pre_register_test.md` sau xác nhận cả nhóm và trước khi xem test.
- Ghi Python/dependency versions, seed, cấu hình, split checksum và commit của lần
  chạy thật. `requirements.txt` hiện là khoảng phiên bản, chưa phải lockfile.
- Fixture là dữ liệu tự tạo cho kiểm tra kỹ thuật, không đại diện chất lượng corpus.

## 9. Phân công và giảm xung đột merge

| Thành viên | File/khu vực phụ trách |
| --- | --- |
| TV1 @angqun | `src/data_loader.py`, `src/baseline.py`, `tests/test_data_loader.py`, `tests/test_baseline.py`, `tests/fixtures/`, notebook EDA, test vét cạn HMM độc lập sau này |
| TV2 @PhamHoagn | `src/hmm_*.py`, `tests/test_hmm_*.py`, `tests/test_benchmark_hmm.py`, `tests/toy_corpus.py`, `scripts/benchmark_hmm.py`, `docs/TV2_*.md`, notebook mô hình; test thuật toán HMM phối hợp TV1 |
| TV3 @PhuongNamkk5 | `src/metrics.py`, `src/model_selection.py`, `tests/test_metrics.py`, `docs/pre_register_test.md`, `results/`, `report/`, bootstrap CI sau này |
| Cả nhóm, một người tích hợp mỗi lần | `src/contracts.py`, `tests/test_contract.py`, `scripts/`, `requirements.txt`, `.gitignore`, `README.md`, `docs/PROJECT_CONTRACT.md` |

Sau khi khung đã có trên main, để cả ba branch công việc xuất phát từ cùng commit.
Không để TV2 phát triển HMM đầy đủ trước khi các branch
còn lại nhận cùng contract. Người sửa API chung cập nhật contract/test trong cùng
thay đổi và thông báo nhóm; không tự sửa file người khác song song.

## 10. Lệnh kiểm tra

Tại repo root, Python >= 3.11, sau khi kích hoạt môi trường hoặc dùng Python của venv:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.smoke_test
```

Smoke test đọc đường dẫn tương đối với chính file script, tách hai câu đầu thành
toy train và câu cuối thành toy dev. Nó kiểm tra accuracy=2/3, macro-F1=2/17 theo
17 labels cố định. Đây là giá trị kỳ vọng tính tay của fixture, không phải kết quả
thực nghiệm. Script kiểm tra import/signature HMM nhưng không gọi fit/predict HMM.
Lỗi import, lệch kỳ vọng hoặc exception bất kỳ làm process trả mã khác 0;
kiểm tra không phụ thuộc `assert` nên vẫn hoạt động với Python `-O`.

## 11. Windows và Colab

Chạy từ repo root với `python -m ...`, dùng import tuyệt đối `from src...` và import
tương đối trong package `src`. Dùng `pathlib`, không hardcode ổ đĩa hoặc đường dẫn
cá nhân. `notebooks/` muốn import phải đặt working directory về repo root.
Lệnh clone/Colab nằm trong README; branch chỉ clone được sau khi nhóm đã push.
Chỉ cần CPU; chưa xác minh bằng một runtime Colab thật trong bước khởi tạo local.

## 12. Trạng thái bàn giao

| Hạng mục | Trạng thái |
| --- | --- |
| CoNLL-U FORM/UPOS parser + fixture | Hoàn chỉnh trong phạm vi đã nêu |
| Most Frequent Tag, fallback, tie-break | Hoàn chỉnh |
| Token Accuracy, fixed-label Macro-F1 | Hoàn chỉnh |
| OOV Accuracy/Rate, accuracy_no_punct | Chưa làm; trả `None` |
| Bigram training, Viterbi NumPy, UNK, log_joint, config/cache | Đã triển khai; xem TV2_BIGRAM.md |
| Trigram training, Viterbi NumPy và benchmark CPU dev | Đã triển khai; xem TV2_TRIGRAM.md và results/tables/bench_dev.csv |
| Grid search | Chỉ signature/tài liệu; `NotImplementedError` |
| Bootstrap CI, EDA, corpus thật, thực nghiệm, báo cáo | Chưa làm |
| Unit test và smoke test | Đã chạy thành công trên Windows local, xem bên dưới |

### Kiểm chứng khung ban đầu ngày 2026-11-20

- Python 3.14.7, NumPy 2.5.5, scikit-learn 1.9.1, pytest 9.2.0, trong `.venv`.
- Cài `requirements.txt`: thành công. Lần đầu sandbox không kết nối được PyPI;
  đã chạy lại thành công qua quyền mạng, không cài vào Python toàn hệ thống.
- `python -m pytest -q --basetemp=.venv/pytest-tmp`: **55 passed in 15.58s**, exit 0.
  Tham số basetemp chỉ đưa file tạm vào workspace của sandbox Windows.
- `python -m scripts.smoke_test`: PASS, exit 0; Token Accuracy =
  `0.6666666666666666`, Macro-F1 = `0.11764705882352941`, ba metric còn lại `null`.
- Đây là kiểm chứng kết nối trên fixture tự tạo. Chưa chạy trong runtime Colab
  thật và chưa đánh giá corpus thực tế.

Rủi ro cần xử lý: giấy phép và tagset corpus thật; thiếu kiểm chứng parser trên
corpus đầy đủ; quy ước smoothing/EOS cần TV1 và TV2 xác nhận bằng brute force;
chi phí Trigram trên CPU; dependency chưa khóa chính xác và Colab chưa chạy thật.
