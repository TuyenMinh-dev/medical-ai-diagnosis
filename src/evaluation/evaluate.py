"""
Cách chạy (đánh giá 1 phiên bản cụ thể theo run_name đã train trước đó):
    python -m src.evaluation.evaluate --config configs/config.yaml --run_name resnet50_baseline
"""
import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import MedicalImageDataset
from src.models.model import build_model
from src.utils.common import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def run_evaluation(config_path: str, run_name: str) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(cfg["output"]["checkpoint_dir"]) / run_name / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại {checkpoint_path}. "
            f"Kiểm tra lại run_name có đúng với lúc train không."
        )

    transform = transforms.Compose([
        transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
        transforms.ToTensor(),
    ])
    dataset = MedicalImageDataset.from_folder(cfg["data"]["processed_dir"], transform=transform)
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)

    model = build_model(cfg["model"]["name"], cfg["model"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    logger.info(f"Kết quả đánh giá cho run: {run_name}")
    logger.info("Classification report:\n" + classification_report(all_labels, all_preds))
    logger.info("Confusion matrix:\n" + str(confusion_matrix(all_labels, all_preds)))

    if cfg["model"]["num_classes"] == 2:
        auc = roc_auc_score(all_labels, all_preds)
        logger.info(f"ROC AUC: {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run_name", type=str, required=True,
                         help="Tên run cần đánh giá, VD: resnet50_baseline")
    args = parser.parse_args()
    run_evaluation(args.config, args.run_name)
