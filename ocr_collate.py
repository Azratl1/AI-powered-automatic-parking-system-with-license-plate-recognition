from __future__ import annotations

from typing import Any

import torch


def ocr_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = [x["image"] for x in batch]
    labels = [x["label"] for x in batch]
    image_paths = [x["image_path"] for x in batch]
    stacked = torch.stack(images, dim=0)
    widths = torch.full((len(images),), images[0].shape[-1], dtype=torch.long)

    return {
        "images": stacked,
        "labels": labels,
        "widths": widths,
        "image_paths": image_paths,
    }
