# Hướng dẫn dùng project cho team (đọc trước khi code)

Tài liệu này giải thích **project được tổ chức thế nào, và mỗi người nên code vào file/thư mục nào**
theo đúng phân công trong timeline. Không cần đọc hết — tìm tên mình trong bảng dưới, đọc phần đó.

## 1. Nguyên tắc chung

- **Không code trực tiếp lên nhánh `main`.** Tạo nhánh riêng từ `dev`, code xong thì mở Pull Request. Chi tiết ở [CONTRIBUTING.md](../CONTRIBUTING.md).
- **Không commit file dữ liệu, model đã train** (`.jpg`, `.pt`, `.h5`...) — các file này đã được loại trong `.gitignore`, nếu `git status` báo có file lạ dạng đó thì đừng `git add` nó.
- Trước khi code, luôn `git pull` nhánh `dev` mới nhất để tránh conflict.
- Có thắc mắc chỗ nào không hiểu trong code sẵn có → hỏi trong nhóm trước khi tự đoán, tránh sửa sai chỗ người khác đang dùng.

## 2. Từng người code vào đâu (theo timeline)

### Trần Việt Đạt — Quản lý & Đánh giá & Báo cáo
- **Tháng 1, Tháng 8**: không code, tổng hợp nội dung vào `README.md` (phần mô tả project) và viết báo cáo NCKH (file riêng, có thể để trong `docs/`, không nhất thiết phải là code).
- **Tháng 4**: kiểm tra dữ liệu sau xử lý — chỉ cần chạy `python -m src.data.prepare_dataset` do Long/Anh làm ra và mở thử vài ảnh trong `data/processed/` để rà soát bằng mắt, không cần sửa code.
- **Tháng 6**: code vào `src/evaluation/compare_runs.py` (đã có sẵn khung) — dùng để đọc `outputs/experiments_log.csv` và chọn phiên bản tốt nhất. Chạy bằng `python -m src.evaluation.compare_runs`.

### Cao Đình Long & Mẫn Duy Anh — Dữ liệu & Chạy thử nghiệm & UI
- **Tháng 3**:
  - Mẫn Duy Anh: sửa `scripts/download_data.py` cho đúng nguồn dataset nhóm chọn.
  - Cao Đình Long: dùng `src/data/prepare_dataset.py` (đã có sẵn) để thống kê và chia tập — thường không cần sửa, chỉ cần chạy đúng lệnh trong `data/README.md`.
- **Tháng 6**: chạy `python -m src.training.train --run_name <ten-de-nho>` nhiều lần với `configs/config.yaml` khác nhau (đổi model, learning rate...) để thử nghiệm. Mỗi lần chạy tự động lưu riêng, không đè lên nhau — xem chi tiết mục 3 bên dưới.
- **Tháng 7**: code giao diện vào `app/streamlit_app.py` (đã có bản mẫu chạy được) — chỉnh sửa giao diện, thêm hướng dẫn sử dụng cho người dùng cuối, không cần đụng vào code model.

### Lê Minh Tuyền & Phạm Chi Hiếu — Môi trường & Model & Deploy
- **Tháng 2**: nếu đổi từ PyTorch sang TensorFlow/Keras, cần sửa `src/models/model.py` và `requirements.txt`. Nếu giữ PyTorch thì không cần làm gì thêm ở bước này.
- **Tháng 4**: code script tiền xử lý/tăng cường dữ liệu (augmentation) — thêm vào `src/data/dataset.py` (phần `transform`) hoặc tạo file mới `src/data/transforms.py` nếu logic phức tạp.
- **Tháng 5**:
  - Lê Minh Tuyền: sửa `src/models/model.py` để thêm/đổi kiến trúc mạng.
  - Phạm Chi Hiếu: sửa phần training loop trong `src/training/train.py` (hàm loss, optimizer).
- **Tháng 7**: code vào `src/api/main.py` (đã có sẵn endpoint `/predict`) — chỉnh để load đúng checkpoint (`outputs/checkpoints/<run_name>/best_model.pt`) của phiên bản tốt nhất mà Đạt đã chọn ở tháng 6.

## 3. Quy tắc quan trọng: mỗi lần train là 1 "phiên bản" riêng

Từ tháng 6 trở đi, cả nhóm sẽ thử nhiều model/tham số khác nhau. Để không ai vô tình ghi đè kết quả của người khác:

- Luôn chạy training kèm tên riêng: `python -m src.training.train --run_name ten-cua-ban_lan1`
- Kết quả sẽ lưu vào `outputs/checkpoints/<run_name>/best_model.pt` — mỗi run một thư mục riêng.
- Log quá trình train xem bằng: `tensorboard --logdir outputs/logs`
- Bảng so sánh tất cả các lần chạy: `python -m src.evaluation.compare_runs`

Thư mục `outputs/` không commit lên GitHub (quá nặng) — nếu cần chia sẻ checkpoint tốt nhất cho cả nhóm, upload lên Google Drive và dán link vào `docs/architecture.md`.

## 4. Khi không chắc code vào đâu

Nếu việc bạn làm không khớp rõ với thư mục nào ở trên, hỏi trong nhóm trước khi tạo file/thư mục mới ở vị trí khác — tránh mỗi người tự tạo một kiểu, dẫn tới cấu trúc project bị lệch.
