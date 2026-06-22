from torch.utils.data import DataLoader

from ocr_dataset import OCRDataset
from ocr_collate import ocr_collate_fn


def main() -> None:
    dataset = OCRDataset(
        "autoriaNumberplateOcrRu-2021-09-01/splits_csv/train.csv",
        img_height=48,
    )

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=ocr_collate_fn,
    )

    batch = next(iter(loader))

    print("images shape:", batch["images"].shape)
    print("labels:", batch["labels"][:3])
    print("widths:", batch["widths"][:3])
    print("image paths:", batch["image_paths"][:2])


if __name__ == "__main__":
    main()