"""
Cách chạy (mỗi lần chạy là 1 "phiên bản thử nghiệm" riêng, không đè lên nhau):
    python -m src.training.train --config configs/config.yaml --run_name resnet50_baseline
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.data.dataset import MedicalImageDataset
from src.models.model import build_model
from src.utils.common import get_logger, set_seed
from src.utils.config import load_config

logger = get_logger(__name__)


def train(config_path: str, run_name: str | None) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    run_name = run_name or f"{cfg['model']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"]) / run_name
    log_dir = Path(cfg["output"]["log_dir"]) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
    ])

    dataset = MedicalImageDataset.from_folder(cfg["data"]["processed_dir"], transform=transform)
    val_size = int(len(dataset) * cfg["data"]["val_split"])
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                               num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                             num_workers=cfg["data"]["num_workers"])

    model = build_model(cfg["model"]["name"], cfg["model"]["num_classes"], cfg["model"]["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)
        logger.info(f"[{run_name}] Epoch {epoch + 1}/{cfg['training']['epochs']} "
                    f"- loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["training"]["early_stopping_patience"]:
                logger.info(f"Early stopping tại epoch {epoch + 1}")
                break

    writer.close()
    log_experiment_result(run_name, cfg, best_val_acc)
    logger.info(f"Xong. Checkpoint tốt nhất: {checkpoint_dir / 'best_model.pt'} (val_acc={best_val_acc:.4f})")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def log_experiment_result(run_name: str, cfg: dict, best_val_acc: float) -> None:
    """Ghi 1 dòng vào outputs/experiments_log.csv để so sánh các phiên bản (đúng việc tháng 6)."""
    log_path = Path("outputs/experiments_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        if is_new:
            writer_csv.writerow(["run_name", "model", "epochs", "learning_rate", "best_val_acc", "timestamp"])
        writer_csv.writerow([run_name, cfg["model"]["name"], cfg["training"]["epochs"],
                              cfg["training"]["learning_rate"], f"{best_val_acc:.4f}",
                              datetime.now().isoformat(timespec="seconds")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_name", type=str, default=None,
                         help="Tên cho phiên bản thử nghiệm này, VD: resnet50_lr0001")
    args = parser.parse_args()
    train(args.config, args.run_name)
