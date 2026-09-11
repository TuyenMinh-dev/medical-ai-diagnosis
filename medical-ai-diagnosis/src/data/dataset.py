from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    @classmethod
    def from_folder(cls, root_dir: str, transform=None) -> "MedicalImageDataset":
        root = Path(root_dir)
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        class_to_idx = {name: i for i, name in enumerate(classes)}

        samples = []
        for class_name, label in class_to_idx.items():
            for img_path in (root / class_name).glob("*"):
                samples.append((str(img_path), label))

        return cls(samples, transform=transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
