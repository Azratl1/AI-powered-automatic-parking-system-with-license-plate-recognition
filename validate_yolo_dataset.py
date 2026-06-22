from __future__ import annotations

from pathlib import Path

import cv2


def validate_split(images_dir: Path, labels_dir: Path) -> tuple[int, int]:
    broken = 0
    total = 0

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        total += 1

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[MISSING LABEL] {image_path.name}")
            broken += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[BROKEN IMAGE] {image_path.name}")
            broken += 1
            continue

        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            print(f"[EMPTY LABEL] {label_path.name}")
            broken += 1
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                print(f"[BAD LABEL FORMAT] {label_path.name}: {line}")
                broken += 1
                break
            try:
                _, cx, cy, w, h = parts
                values = [float(cx), float(cy), float(w), float(h)]
            except ValueError:
                print(f"[BAD LABEL VALUE] {label_path.name}: {line}")
                broken += 1
                break

            if any(v < 0.0 or v > 1.0 for v in values):
                print(f"[OUT OF RANGE] {label_path.name}: {line}")
                broken += 1
                break

    return total, broken


def main() -> None:
    root = Path("archive")
    for split in ("train", "val"):
        total, broken = validate_split(root / "images" / split, root / "labels" / split)
        print(f"{split}: total={total}, broken={broken}")


if __name__ == "__main__":
    main()
