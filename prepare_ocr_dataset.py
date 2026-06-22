from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

ALPHABET = "0123456789ABCEHKMOPTXY"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(label: str) -> str:
    label = label.strip().upper().replace(" ", "")
    label = re.sub(r"[^0-9ABCEHKMOPTXY]", "", label)
    return label


def find_image_for_json(json_path: Path, img_dir_candidates: list[Path]) -> Path | None:
    stem = json_path.stem
    for img_dir in img_dir_candidates:
        if not img_dir.exists():
            continue
        for ext in IMAGE_EXTS:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def collect_split(split_dir: Path, split_name: str) -> list[tuple[str, str]]:
    ann_dir = split_dir / "ann"
    img_dir = split_dir / "img"

    if not ann_dir.exists():
        print(f"[WARN] No ann dir: {ann_dir}")
        return []

    rows: list[tuple[str, str]] = []
    json_files = sorted(ann_dir.glob("*.json"))

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {json_path}: {e}")
            continue

        root = data.get("root", data)
        raw_label = root.get("description", "")
        label = normalize_label(raw_label)

        if not label:
            print(f"[WARN] Empty/invalid label in {json_path}")
            continue

        image_path = find_image_for_json(json_path, [img_dir, split_dir])
        if image_path is None:
            print(f"[WARN] Image not found for {json_path.name}")
            continue

        rows.append((str(image_path.resolve()), label))

    print(f"[INFO] {split_name}: collected {len(rows)} samples")
    return rows


def save_csv(rows: Iterable[tuple[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)
    print(f"[INFO] Saved: {out_csv}")


def main() -> None:
    dataset_root = Path("autoriaNumberplateOcrRu-2021-09-01")
    out_dir = dataset_root / "splits_csv"

    for split_name in ["train", "val", "test"]:
        split_dir = dataset_root / split_name
        rows = collect_split(split_dir, split_name)
        save_csv(rows, out_dir / f"{split_name}.csv")


if __name__ == "__main__":
    main()
