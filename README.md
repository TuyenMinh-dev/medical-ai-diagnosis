# Medical AI Diagnosis

Ứng dụng AI/Deep Learning hỗ trợ phân loại và cảnh báo sớm nguy cơ bệnh lý qua ảnh chụp y khoa và/hoặc dữ liệu lâm sàng.

## 1. Mục tiêu đề tài

- Xây dựng mô hình Deep Learning phân loại ảnh y khoa (X-quang, CT, MRI, ...) để hỗ trợ chẩn đoán.
- Kết hợp (tùy chọn) dữ liệu lâm sàng dạng bảng (clinical tabular data) để tăng độ chính xác.
- Cung cấp API/backend để tích hợp mô hình vào ứng dụng thực tế, phục vụ demo và đánh giá.

## 2. Cấu trúc thư mục

**Thành viên mới trong team: đọc [docs/TEAM_GUIDE.md](./docs/TEAM_GUIDE.md) trước — chỉ rõ ai code vào file nào theo timeline.**

```
medical-ai-diagnosis/
├── configs/            # File cấu hình (yaml) cho từng experiment
├── data/
│   ├── raw/             # Dữ liệu gốc (raw/<ten_nhan>/*.jpg), KHÔNG commit lên git
│   └── processed/        # Tự sinh bởi src/data/prepare_dataset.py (train/val/test)
├── notebooks/           # Jupyter notebook để EDA, thử nghiệm nhanh
├── app/                 # UI prototype (Streamlit) — demo cho người dùng cuối
├── src/
│   ├── data/             # Dataset/DataLoader, chia train/val/test
│   ├── models/           # Định nghĩa kiến trúc model
│   ├── training/         # Training loop — mỗi lần chạy lưu 1 "run" riêng
│   ├── evaluation/       # Tính metric, so sánh các phiên bản đã train
│   ├── api/              # Backend API (FastAPI) để serve model cho UI
│   └── utils/            # Hàm dùng chung: logging, config loader, seed...
├── outputs/              # Checkpoint + log của từng lần train (KHÔNG commit, tự sinh)
├── tests/                # Unit test
├── docs/                 # TEAM_GUIDE.md, architecture.md — tài liệu cho team
├── scripts/              # Script tải dữ liệu, setup môi trường
└── .github/workflows/    # CI pipeline
```

## 3. Quy ước dữ liệu

Dữ liệu y khoa (ảnh, hồ sơ lâm sàng) **không được commit lên GitHub** vì lý do bản quyền/đạo đức/dung lượng.
- Đặt dữ liệu vào `data/raw/` (đã có trong `.gitignore`).
- Nếu cần chia sẻ dữ liệu trong nhóm, dùng Google Drive/Kaggle/HuggingFace Datasets và ghi rõ nguồn + link trong `data/README.md`.

## 4. Cài đặt môi trường

```bash
git clone https://github.com/<your-org>/medical-ai-diagnosis.git
cd medical-ai-diagnosis
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Chạy thử

```bash
# Train model (ví dụ)
python -m src.training.train --config configs/config.yaml

# Chạy API backend
uvicorn src.api.main:app --reload
```

## 6. Quy trình làm việc nhóm (Git workflow)

Xem chi tiết tại [CONTRIBUTING.md](./CONTRIBUTING.md).

- Branch chính: `main` (luôn ổn định, chỉ merge qua Pull Request).
- Branch phát triển: `dev` (tích hợp các tính năng trước khi lên `main`).
- Nhánh tính năng: `feature/<ten-tinh-nang>`, `fix/<ten-loi>`, `experiment/<ten-thi-nghiem>`.

## 7. Thành viên nhóm

| Tên | Vai trò | GitHub |
|---|---|---|
|  |  |  |

## 8. License

Xem [LICENSE](./LICENSE).
