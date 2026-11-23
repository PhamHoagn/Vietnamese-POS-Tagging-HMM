# Hỏi đáp bảo vệ

Luyện theo [slide nhóm](../slides/NLP_POS_NHOM.pptx) và
[demo ba mô hình](TV2_GIAI_TAY_VITERBI.md). Các câu hỏi dưới đây dùng cho
5 phút hỏi đáp sau phần trình bày. Giá trị Test lấy từ lần chạy đã khóa.

| Người trả lời | Câu hỏi | Ý cần giải thích |
| --- | --- | --- |
| TV1 | Đầu vào và đầu ra là gì? | Đầu vào là câu có token đã tách. Đầu ra là một nhãn UPOS cho mỗi token. FORM trong CoNLL-U có thể chứa khoảng trắng, nên parser đọc theo cột tab. |
| TV1 | Tại sao giữ nguyên split và có kiểm tra trùng? | Câu là đơn vị dự đoán và đánh giá. Giữ split chuẩn UD VTB và ghi các cặp câu trùng trong audit để người đọc đánh giá ảnh hưởng. |
| TV1 | Baseline đo được điều gì? | Most Frequent Tag dự đoán nhãn phổ biến của từng FORM trên train. FORM mới dùng nhãn phổ biến nhất toàn train. Baseline cho biết mô hình ngữ cảnh cải thiện ở những nhóm nào, ví dụ OOV. |
| TV2 | Vì sao chọn HMM và dùng log? | HMM mô hình hóa chuỗi nhãn ẩn và token quan sát. Độc lập phát xạ có điều kiện theo nhãn giúp phân tích mô hình. Log tránh underflow và chuyển phép nhân thành phép cộng khi tính Viterbi. |
| TV2 | Trigram dùng cái gì làm ngữ cảnh? | Hai nhãn UPOS trước đó. Trạng thái Viterbi là cặp nhãn, backpointer lưu nhãn trước cặp. Bắt đầu bằng hai BOS và thêm transition đến EOS. Với `K` nhãn, Bigram tốn `O(TK²)`, Trigram tốn `O(TK³)`. |
| TV2 | Tại sao Add-α có thể cho phân phối đều? | Nếu cặp nhãn trước có tổng đếm bằng 0, mọi đích gồm EOS đều nhận `α/(α(K+1)) = 1/(K+1)`. Interpolation có thể trộn với phân phối bậc thấp nếu được làm như một thí nghiệm riêng. |
| TV2 | Cutoff bằng 0 ảnh hưởng UNK thế nào? | Các FORM đã thấy được giữ, nên UNK có đếm phát xạ bằng 0. Add-α vẫn cho UNK số dương. Mẫu số nhỏ của một nhãn hiếm có thể làm phát xạ UNK lớn hơn. |
| TV3 | Chọn cấu hình và đo OOV thế nào? | Chọn theo Accuracy, Macro-F1, cutoff nhỏ hơn rồi α lớn hơn trên dev. Hai bậc dùng cùng lưới. OOV được định nghĩa theo từ vựng train trước khi áp cutoff, áp dụng như nhau cho mọi cấu hình. |
| TV3 | So sánh tuned và fixed trả lời câu hỏi gì? | Tuned so hai mô hình tối ưu riêng trên dev. Fixed giữ α, cutoff, phát xạ và vocab như nhau để so bậc Markov. Đọc cả hai hiệu số cùng CI và đối chiếu nhóm lỗi. |
| TV3 | Tại sao bootstrap theo câu? | Mỗi mẫu lấy lại các câu có hoàn lại và dùng cùng chỉ số câu cho mọi mô hình. Gom số token đúng và tổng token trước khi tính Accuracy. CI chứa 0 nghĩa là phép so sánh này chưa cho bằng chứng rõ về chiều chênh lệch. |
| Cả nhóm | AI hỗ trợ phần nào? | Đối chiếu nhật ký, nêu những mục đã dùng AI và cách kiểm chứng. Mỗi thành viên giải thích được phần mình trình bày và chịu trách nhiệm về số liệu. |
