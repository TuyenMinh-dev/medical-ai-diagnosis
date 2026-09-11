import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from src.models.model import build_model
from src.utils.config import load_config

app = FastAPI(title="Medical AI Diagnosis API")

CONFIG_PATH = "configs/config.yaml"
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"

cfg = load_config(CONFIG_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(cfg["model"]["name"], cfg["model"]["num_classes"], pretrained=False).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
    transforms.ToTensor(),
])

CLASS_NAMES = ["negative", "positive"]  # cập nhật đúng thứ tự nhãn thực tế của nhóm


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }
