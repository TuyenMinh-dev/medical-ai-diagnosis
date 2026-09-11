"""
Dùng file này để: thống kê số lượng ảnh theo từng nhãn, và chia dữ liệu
thô trong data/raw/<ten_nhan>/*.jpg thành data/processed/{train,val,test}/<ten_nhan>/*.jpg

Cách chạy:
    python -m src.data.prepare_dataset --val_size 0.15 --test_size 0.15
"""
import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def print_label_stats(raw_dir: Path) -> None:
    print("Thống kê số lượng ảnh theo nhãn:")
    for class_dir in sorted(raw_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            print(f"  - {class_dir.name}: {count} ảnh")


def split_dataset(raw_dir: Path, processed_dir: Path, val_size: float, test_size: float, seed: int) -> None:
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        files = list(class_dir.glob("*"))
        train_files, temp_files = train_test_split(files, test_size=val_size + test_size, random_state=seed)
        val_files, test_files = train_test_split(
            temp_files, test_size=test_size / (val_size + test_size), random_state=seed
        )

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            out_dir = processed_dir / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, out_dir / f.name)

        print(f"{class_dir.name}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    print_label_stats(raw_dir)
    split_dataset(raw_dir, Path(args.processed_dir), args.val_size, args.test_size, args.seed)
