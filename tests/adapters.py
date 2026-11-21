"""Điểm nối duy nhất giữa test TV1 và API HMM của TV2, chưa cài đặt.

Khi TV2 chốt API, chỉ bổ sung chuyển đổi dữ liệu và lời gọi tại file này.
Không viết lại decoder, không dùng oracle để thay kết quả của mô hình.
Các test đối chiếu oracle/chuẩn hóa gọi mô hình qua bốn hàm bên dưới.
"""

from collections.abc import Mapping, Sequence
from typing import TypedDict


class ToyTables(TypedDict):
    """Bảng xác suất thường (không phải log/count), lấy từ oracle độc lập.

    labels: thứ tự nhãn dự đoán, ví dụ ("A", "B", "C"); không gồm start/end.
    tokens: tập từ phát xạ, ví dụ ("x", "y", "z"); thêm UNK nếu bảng có UNK.
    pi: nhãn đầu -> xác suất, dùng cho order=1.
    q1: nhãn trước -> {nhãn đích hoặc </s>: xác suất}.
    q2: (hai nhãn lịch sử) -> {nhãn đích hoặc </s>: xác suất};
        có (<s>, <s>), (<s>, tag) và các cặp nhãn thông thường.
    e: nhãn -> {token: xác suất}; không có emission cho start/end.

    Mỗi phân phối có tổng bằng 1; các bảng giữ nguyên giá trị từ test.
    Tên khóa chữ thường ánh xạ PI/Q1/Q2/E trong oracle; không tự suy ra
    pi từ q2 vì hai phân phối có tập đích khác nhau.
    """

    labels: Sequence[str]
    tokens: Sequence[str]
    pi: Mapping[str, float]
    q1: Mapping[str, Mapping[str, float]]
    q2: Mapping[tuple[str, str], Mapping[str, float]]
    e: Mapping[str, Mapping[str, float]]


def build_toy_model(order: int, tables: ToyTables) -> object:
    """Dựng model TV2 từ bảng xác suất cho trước, không fit hay smoothing lại.

    order=1 là Bigram, order=2 là Trigram; giá trị khác phải gây ValueError.
    Nhận ToyTables với cả pi/q1/q2/e; chọn bảng chuyển tương ứng với order.
    Nhãn toy A/B/C được phép dùng dù không thuộc UPOS. Phải giữ thứ tự labels
    và mọi xác suất, kể cả xác suất kết thúc; không sửa tables của bên gọi.
    Nếu API dùng log, adapter chỉ chuyển biểu diễn (p=0 thành -inf).

    Trả đối tượng model opaque dùng được với ba hàm còn lại. Adapter không
    giải mã hoặc huấn luyện thay TV2. Model thiếu khả năng nạp bảng phải báo
    lỗi rõ ràng, không tạo kết quả giả hay dùng oracle làm model thay thế.
    """
    raise NotImplementedError("Chờ TV2 chốt API nạp bảng xác suất toy.")


def decode(model: object, tokens: Sequence[str]) -> tuple[list[str], float]:
    """Gọi decoder TV2 và trả (chuỗi nhãn tối ưu, log-score tự nhiên).

    tokens là chuỗi FORM đã tách đúng ranh giới, không kèm gold và không bị
    adapter tách lại. tags có đúng len(tokens) phần tử, chỉ thuộc labels,
    không chứa <s> hoặc </s>. Không sửa tokens hoặc học thêm từ đầu vào.

    log_score là log xác suất đồng thời của đường đi, quan sát và EOS,
    không phải xác suất hậu nghiệm hoặc score chuẩn hóa theo độ dài:
    - Bigram: log pi[y0] + tổng log emission + tổng log q1 liên tiếp
      + log q1[nhãn cuối][</s>].
    - Trigram: bắt đầu (<s>, <s>), cộng log q2 và emission từng token,
      cuối cùng cộng log q2[hai nhãn lịch sử cuối][</s>].
    Khi hòa, được trả bất kỳ nghiệm tối ưu; test kiểm tra trong tập nghiệm
    của oracle, không ép một đường đi duy nhất. Score phải ứng với tags trả về.
    Nếu decoder chỉ trả tags, chỉ dùng API score độc lập của model TV2;
    không gọi oracle hoặc tự cài thuật toán tính score trong adapter.

    Câu rỗng: Bigram toy có pi trên nhãn nên chưa định nghĩa xác suất câu
    rỗng, phải báo ValueError. Trigram trả ([], log q2[(<s>,<s>)][</s>]).
    OOV dùng UNK theo model nếu có; toy không có UNK phải báo ValueError.
    """
    raise NotImplementedError("Chờ TV2 chốt API decode và log-score.")


def get_transition_row(model: object, history: Sequence[str]) -> dict[str, float]:
    """Trả bản sao phân phối chuyển ở miền xác suất thường, không phải log.

    Bigram nhận history=(tag,), Trigram nhận history=(u,v), gồm các lịch sử
    start hợp lệ của q2. Bigram pi được kiểm tra qua decode; (<s>,) không
    phải hàng q1 của toy và không được tự tạo bằng cách thêm EOS vào pi.
    Trả đủ đích labels ∪ {</s>} (|S|+1), kể cả đích có xác suất bằng 0;
    không có <s>, không bỏ EOS hay chuẩn hóa lại sau khi bỏ cột.

    Với model đã huấn luyện, trả xác suất thực sự model dùng sau smoothing,
    không tự suy ra hàng uniform khi history chưa có. History sai độ dài
    gây ValueError; history không được model định nghĩa gây KeyError.
    Dict trả về không được chia sẻ bộ nhớ có thể sửa trạng thái model.
    """
    raise NotImplementedError("Chờ TV2 chốt API truy xuất transition.")


def get_emission_row(model: object, tag: str) -> dict[str, float]:
    """Trả bản sao {token: xác suất phát xạ} của một nhãn, không phải log.

    tag phải thuộc labels. Trả đầy đủ vocabulary phát xạ thực tế của model,
    gồm mọi lớp UNK dự phòng nếu có, kể cả phần tử có xác suất bằng 0.
    Với toy, tập khóa chính là tables['tokens']; không tự thêm UNK/start/end.
    Với model train, dùng vocabulary sau cutoff cùng các lớp UNK; không dùng
    vocabulary nguyên gốc để thay mẫu số hoặc chuẩn hóa lại hàng.
    Nhãn không tồn tại (kể cả <s>, </s>) gây KeyError. Dict trả về độc lập
    với trạng thái model, tổng xác suất phải phản ánh đúng model đang dùng.
    """
    raise NotImplementedError("Chờ TV2 chốt API truy xuất emission.")
