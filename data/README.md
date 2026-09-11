# Dữ liệu

Thư mục này chỉ chứa cấu trúc, không chứa dữ liệu thật (xem `.gitignore`).

## Nguồn dữ liệu

Ghi rõ nguồn dữ liệu sử dụng, ví dụ:
- Tên dataset:
- Link (Kaggle/HuggingFace/bệnh viện hợp tác...):
- Số lượng ảnh/mẫu:
- Nhãn (label) và ý nghĩa từng nhãn:
- Giấy phép sử dụng dữ liệu (license):

## Cấu trúc thư mục dữ liệu

```
data/
├── raw/                    # Dữ liệu gốc, giữ nguyên không chỉnh sửa. BẮT BUỘC theo cấu trúc:
│   ├── <ten_nhan_1>/       #   VD: raw/pneumonia/anh1.jpg, raw/pneumonia/anh2.jpg...
│   │   └── *.jpg
│   └── <ten_nhan_2>/
│       └── *.jpg
└── processed/              # Tự động sinh ra bởi src/data/prepare_dataset.py, không tự sửa tay
    ├── train/<ten_nhan>/*.jpg
    ├── val/<ten_nhan>/*.jpg
    └── test/<ten_nhan>/*.jpg
```

Mỗi thư mục con trong `raw/` là 1 nhãn (label) — tên thư mục chính là tên nhãn.

## Quy trình chuẩn bị dữ liệu (tháng 3)

1. **Tải dữ liệu thô** (việc của người phụ trách tìm/tải dataset):
   ```
   python scripts/download_data.py --dataset_name <ten-dataset-tren-huggingface>
   ```
   Hoặc tải thủ công và sắp xếp đúng cấu trúc `raw/<ten_nhan>/*.jpg` ở trên.

2. **Thống kê + chia train/val/test** (việc của người phụ trách tổ chức dữ liệu):
   ```
   python -m src.data.prepare_dataset --val_size 0.15 --test_size 0.15
   ```
   Lệnh này in ra số lượng ảnh mỗi nhãn, và tự tạo `data/processed/{train,val,test}/`.
