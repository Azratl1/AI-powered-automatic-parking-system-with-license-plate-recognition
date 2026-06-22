from __future__ import annotations

import platform
from pathlib import Path


def main() -> None:
    import torch

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    dataset_yaml = Path("archive/dataset_local.yaml")
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            "Missing archive/dataset_local.yaml. Run prepare_yolo_dataset.py first."
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    batch = 16 if torch.cuda.is_available() else 4
    is_windows = platform.system().lower() == "windows"
    workers = 0 if is_windows else (4 if torch.cuda.is_available() else 0)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=80,
        imgsz=960,
        batch=batch,
        patience=20,
        device=device,
        workers=workers,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=1e-2,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.3,
        dfl=1.5,
        hsv_h=0.012,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=2.0,
        translate=0.08,
        scale=0.45,
        shear=0.0,
        perspective=0.0008,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.35,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.15,
        close_mosaic=10,
        project="runs_yolo",
        name="plate_detector_ru",
        pretrained=True,
        single_cls=True,
        rect=False,
        cos_lr=True,
        amp=torch.cuda.is_available(),
        cache=False,
    )


if __name__ == "__main__":
    main()
