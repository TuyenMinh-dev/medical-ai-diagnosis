"""
Script mẫu để tải bộ dữ liệu y khoa/lâm sàng mã nguồn mở về data/raw/.
Điền đúng nguồn dữ liệu nhóm chọn (Kaggle, HuggingFace Datasets...) vào đây.

Ví dụ dùng Kaggle CLI (cần "pip install kaggle" và file kaggle.json API key,
xem hướng dẫn: https://github.com/Kaggle/kaggle-api):

    kaggle datasets download -d <chu-so-huu>/<ten-dataset> -p data/raw --unzip

Ví dụ dùng HuggingFace Datasets:
"""
import argparse
from pathlib import Path


def download_from_huggingface(dataset_name: str, output_dir: str) -> None:
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in ds.items():
        for i, example in enumerate(split_data):
            label = str(example.get("label", "unknown"))
            label_dir = out / label
            label_dir.mkdir(exist_ok=True)
            image = example["image"]
            image.save(label_dir / f"{split_name}_{i}.png")

    print(f"Đã tải và lưu dữ liệu vào {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                         help="Tên dataset trên HuggingFace, VD: keremberke/chest-xray-classification")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()

    download_from_huggingface(args.dataset_name, args.output_dir)
