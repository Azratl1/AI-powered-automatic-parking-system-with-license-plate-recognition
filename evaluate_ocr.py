from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from crnn_model import CRNN


ALPHABET = "0123456789ABCEHKMOPTXY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR model in notebook style.")
    parser.add_argument("--dataset-root", default="autoriaNumberplateOcrRu-2021-09-01")
    parser.add_argument("--weights", default="ocr_checkpoints/ocr_best.pt")
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="ocr_checkpoints/eval")
    return parser.parse_args()


def decode_preds(preds: torch.Tensor, int_to_char: dict[int, str]) -> list[str]:
    preds = preds.permute(1, 0, 2)
    preds = torch.argmax(preds, dim=2)
    decoded_texts: list[str] = []
    for pred in preds:
        decoded_seq: list[str] = []
        last_char_idx = 0
        for char_idx in pred:
            idx = char_idx.item()
            if idx != 0 and idx != last_char_idx:
                decoded_seq.append(int_to_char.get(idx, ""))
            last_char_idx = idx
        decoded_texts.append("".join(decoded_seq))
    return decoded_texts


def calculate_cer(preds: list[str], targets: list[str]) -> float:
    def levenshtein_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return len(a)
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            for j, cb in enumerate(b, start=1):
                current.append(
                    min(
                        previous[j] + 1,
                        current[j - 1] + 1,
                        previous[j - 1] + (ca != cb),
                    )
                )
            previous = current
        return previous[-1]

    total_dist = 0
    total_len = 0
    for pred, target in zip(preds, targets):
        total_dist += levenshtein_distance(pred, target)
        total_len += len(target)
    return total_dist / total_len if total_len > 0 else 0.0


class OCRDataset(Dataset):
    def __init__(self, base_dir: str | Path, transform: transforms.Compose, char_to_int: dict[str, int]) -> None:
        self.base_dir = Path(base_dir)
        self.img_dir = self.base_dir / "img"
        self.ann_dir = self.base_dir / "ann"
        self.transform = transform
        self.char_to_int = char_to_int
        self.filenames = sorted([path.stem for path in self.img_dir.iterdir() if path.is_file()])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        img_path = self.img_dir / f"{filename}.png"
        ann_path = self.ann_dir / f"{filename}.json"
        image = Image.open(img_path).convert("L")
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        text = ann.get("description", "")
        text = "".join(ch for ch in text if ch in self.char_to_int)
        image_tensor = self.transform(image)
        return image_tensor, text, image


def collate_fn(batch, char_to_int: dict[str, int]):
    image_tensors, texts, original_images = zip(*batch)
    images = torch.stack(image_tensors, 0)
    encoded_texts = [torch.tensor([char_to_int.get(char, 0) for char in text], dtype=torch.long) for text in texts]
    text_lengths = torch.tensor([len(text) for text in encoded_texts], dtype=torch.long)
    encoded_texts = nn.utils.rnn.pad_sequence(encoded_texts, batch_first=True)
    return images, encoded_texts, text_lengths, texts, original_images


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_root = Path(args.dataset_root)
    test_dir = dataset_root / "test"
    char_to_int = {char: i + 1 for i, char in enumerate(ALPHABET)}
    int_to_char = {i: char for char, i in char_to_int.items()}
    num_classes = len(ALPHABET) + 1

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    test_dataset = OCRDataset(test_dir, transform=transform, char_to_int=char_to_int)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, char_to_int),
        num_workers=0,
    )

    model = CRNN(img_h=args.img_height, img_w=args.img_width, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    test_loss = 0.0
    all_preds: list[str] = []
    all_targets: list[str] = []
    preview: list[tuple[Image.Image, str, str]] = []

    with torch.no_grad():
        for images, texts, text_lengths, original_texts, original_images in tqdm(test_loader, desc="OCR test eval"):
            images = images.to(device)
            texts = texts.to(device)
            text_lengths = text_lengths.to(device)

            preds = model(images)
            pred_lengths = torch.full(
                size=(images.size(0),),
                fill_value=preds.size(0),
                dtype=torch.long,
                device=device,
            )
            test_loss += criterion(preds, texts, pred_lengths, text_lengths).item()

            decoded_preds = decode_preds(preds, int_to_char)
            all_preds.extend(decoded_preds)
            all_targets.extend(list(original_texts))

            if len(preview) < 8:
                for image, gt, pred in zip(original_images, original_texts, decoded_preds):
                    preview.append((image, gt, pred))
                    if len(preview) >= 8:
                        break

    avg_test_loss = test_loss / max(1, len(test_loader))
    test_accuracy = sum(1 for pred, orig in zip(all_preds, all_targets) if pred == orig) / max(1, len(all_targets))
    test_cer = calculate_cer(all_preds, all_targets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (image, gt, pred) in zip(axes.flatten(), preview):
        ax.imshow(image, cmap="gray")
        ax.set_title(f"GT: {gt}\nPred: {pred}", fontsize=10)
        ax.axis("off")
    for ax in axes.flatten()[len(preview):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "ocr_test_samples.png", dpi=180)
    plt.close()

    summary = {
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy,
        "test_cer": test_cer,
    }
    (output_dir / "ocr_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Exact Match Accuracy: {test_accuracy:.4f}")
    print(f"Character Error Rate: {test_cer:.4f}")
    print(f"Saved samples to: {output_dir / 'ocr_test_samples.png'}")


if __name__ == "__main__":
    main()
