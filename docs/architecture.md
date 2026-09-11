# Kiến trúc hệ thống

## 1. Tổng quan
Mô tả pipeline tổng thể: Dữ liệu → Tiền xử lý → Model → Đánh giá → API → (Frontend/Demo nếu có).

## 2. Pipeline dữ liệu
- Nguồn dữ liệu:
- Các bước tiền xử lý (resize, chuẩn hóa, augmentation):
- Chia tập train/val/test:

## 3. Model
- Kiến trúc sử dụng và lý do lựa chọn:
- Transfer learning / train từ đầu:
- Hyperparameter chính:

## 4. Đánh giá
- Các metric sử dụng (accuracy, precision/recall, F1, AUC, confusion matrix...) và lý do phù hợp với bài toán y khoa (VD: ưu tiên recall để giảm bỏ sót ca bệnh):

## 5. Triển khai (Deployment)
- API backend: FastAPI, endpoint `/predict`
- Cách đóng gói (Docker, nếu có):
- Giới hạn/lưu ý khi dùng ngoài thực tế (đây là công cụ hỗ trợ, không thay thế chẩn đoán của bác sĩ):

## 6. Sơ đồ (điền thêm khi có)

