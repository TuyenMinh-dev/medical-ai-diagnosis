"""
Đọc outputs/experiments_log.csv (được tự động ghi mỗi lần train.py chạy xong)
và in ra bảng so sánh các phiên bản, sắp xếp theo val_acc giảm dần.

Cách chạy:
    python -m src.evaluation.compare_runs
"""
import csv
from pathlib import Path


def compare_runs(log_path: str = "outputs/experiments_log.csv") -> None:
    path = Path(log_path)
    if not path.exists():
        print(f"Chưa có file {log_path} — cần chạy train.py ít nhất 1 lần trước.")
        return

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["best_val_acc"]), reverse=True)

    print(f"{'run_name':<30}{'model':<15}{'epochs':<8}{'lr':<10}{'best_val_acc':<12}{'timestamp'}")
    for r in rows:
        print(f"{r['run_name']:<30}{r['model']:<15}{r['epochs']:<8}{r['learning_rate']:<10}"
              f"{r['best_val_acc']:<12}{r['timestamp']}")

    if rows:
        print(f"\n>> Phiên bản tốt nhất hiện tại: {rows[0]['run_name']} (val_acc={rows[0]['best_val_acc']})")


if __name__ == "__main__":
    compare_runs()
