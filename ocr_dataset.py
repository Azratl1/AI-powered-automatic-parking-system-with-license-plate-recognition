from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset

from ocr_augment import augment_ocr_image


ALPHABET = set("0123456789ABCEHKMOPTXY")


def normalize_label(label: str) -> str:
    label = label.strip().upper().replace(" ", "")
    label = "".join(ch for ch in label if ch in ALPHABET)
    return label


class OCRDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        img_height: int = 32,
        img_width: int = 128,
        augment: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.samples: list[tuple[str, str]] = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = normalize_label(row["label"])
                if not label:
                    continue
                self.samples.append((row["image_path"], label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path, label = self.samples[idx]

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")

        if self.augment:
            img = augment_ocr_image(img)

        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)

        img = img.astype("float32") / 255.0
        img = (img - 0.5) / 0.5

        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        return {
            "image": img,
            "label": label,
            "image_path": image_path,
        }
