# Vietnamese-POS-Tagging-HMM

Khung dự án gán nhãn UPOS tiếng Việt bằng baseline Most Frequent Tag và HMM
Bigram/Trigram. Hiện parser, baseline, Token Accuracy và Macro-F1 đã có code;
HMM, các metric còn lại và grid search vẫn là placeholder.

Python **3.11 trở lên**, CPU; không cần GPU hoặc dataset thật để kiểm tra kết nối.
Chạy các lệnh từ thư mục gốc repository.

## Chạy trên Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke_test
```

## Chạy trên Google Colab CPU

Sau khi các thay đổi đã được nhóm review và đưa lên remote, clone branch chứa
khung dự án. Không chạy `git clone` lại nếu đã có checkout trong notebook.

```python
!git clone --branch main https://github.com/PhamHoagn/Vietnamese-POS-Tagging-HMM.git
%cd Vietnamese-POS-Tagging-HMM
!python -m pip install -r requirements.txt
!python -m pytest -q
!python -m scripts.smoke_test
```

Khi chưa push, có thể tải bản ZIP của checkout local lên Colab, giải nén và `%cd`
vào thư mục chứa `requirements.txt`, sau đó chạy ba lệnh cuối. Notebook import
bằng `from src...`; không thêm đường dẫn máy cá nhân vào `sys.path` và không chạy
trực tiếp `python src/<module>.py`.

Smoke test chỉ dùng fixture tự tạo trong `tests/fixtures/tiny.conllu` với toy
train/dev; số in ra dùng để kiểm tra kết nối, không phải kết quả thực nghiệm.

- [API Contract và phân công](docs/PROJECT_CONTRACT.md)
- [Bản nháp đăng ký trước đánh giá test](docs/pre_register_test.md)

Không đưa corpus gốc vào Git trước khi xác nhận giấy phép. Thư mục `data/raw/`
được ignore, ngoại trừ `.gitkeep`.
