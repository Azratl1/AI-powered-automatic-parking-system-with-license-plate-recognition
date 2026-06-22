from __future__ import annotations

from pathlib import Path


def main() -> None:
    dataset_root = Path("archive").resolve()
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"

    for path in [images_train, images_val, labels_train, labels_val]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    train_images = list(images_train.iterdir())
    val_images = list(images_val.iterdir())
    train_labels = list(labels_train.iterdir())
    val_labels = list(labels_val.iterdir())

    print(f"Train images: {len(train_images)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Val images: {len(val_images)}")
    print(f"Val labels: {len(val_labels)}")

    yaml_path = dataset_root / "dataset_local.yaml"
    yaml_text = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: license_plate\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"Saved Ultralytics dataset config to: {yaml_path}")


if __name__ == "__main__":
    main()
