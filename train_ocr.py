from __future__ import annotations

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Train OCR model using notebook-style pipeline.")
    parser.add_argument("--dataset-root", default="autoriaNumberplateOcrRu-2021-09-01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default="ocr_checkpoints")
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
        print(f"Found {len(self.filenames)} images in {self.base_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        img_path = self.img_dir / f"{filename}.png"
        ann_path = self.ann_dir / f"{filename}.json"

        try:
            image = Image.open(img_path).convert("L")
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            text = ann.get("description", "")
            text = "".join(ch for ch in text if ch in self.char_to_int)
            if self.transform:
                image = self.transform(image)
            return image, text
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self))


def collate_fn(batch, char_to_int: dict[str, int]):
    images, texts = zip(*batch)
    images = torch.stack(images, 0)
    encoded_texts = [torch.tensor([char_to_int.get(char, 0) for char in text], dtype=torch.long) for text in texts]
    text_lengths = torch.tensor([len(text) for text in encoded_texts], dtype=torch.long)
    encoded_texts = nn.utils.rnn.pad_sequence(encoded_texts, batch_first=True)
    return images, encoded_texts, text_lengths


def save_training_plots(history: dict[str, list[float]], save_dir: Path) -> None:
    print("--- Saving OCR training curves ---")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_title("OCR Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["val_accuracy"], label="Validation Accuracy (Exact Match)")
    axes[1].plot(history["val_cer"], label="Validation CER")
    axes[1].set_title("OCR Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / "ocr_training_curves.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- System check ---")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    print("--------------------\n")

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

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

    train_dataset = OCRDataset(train_dir, transform=transform, char_to_int=char_to_int)
    val_dataset = OCRDataset(val_dir, transform=transform, char_to_int=char_to_int)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, char_to_int),
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, char_to_int),
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = CRNN(img_h=args.img_height, img_w=args.img_width, num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_cer": []}
    best_val_accuracy = 0.0

    print(f"\nStarting OCR training for {args.epochs} epochs...")

    completed_epochs = 0
    interrupted = False

    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
            for images, texts, text_lengths in progress_bar:
                images = images.to(device)
                texts = texts.to(device)
                text_lengths = text_lengths.to(device)

                optimizer.zero_grad()
                preds = model(images)
                pred_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(0),
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(preds, texts, pred_lengths, text_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            all_decoded_preds: list[str] = []
            all_original_texts: list[str] = []
            with torch.no_grad():
                progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
                for images, texts, text_lengths in progress_bar_val:
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
                    val_loss += criterion(preds, texts, pred_lengths, text_lengths).item()

                    decoded_preds = decode_preds(preds, int_to_char)
                    original_texts = [
                        "".join([int_to_char.get(i.item(), "") for i in text if i != 0])
                        for text in texts
                    ]
                    all_decoded_preds.extend(decoded_preds)
                    all_original_texts.extend(original_texts)

            avg_val_loss = val_loss / max(1, len(val_loader))
            val_accuracy = sum(
                1 for pred, orig in zip(all_decoded_preds, all_original_texts) if pred == orig
            ) / max(1, len(all_original_texts))
            val_cer = calculate_cer(all_decoded_preds, all_original_texts)

            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_accuracy)
            history["val_cer"].append(val_cer)

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"Val CER: {val_cer:.4f}"
            )

            torch.save(model.state_dict(), save_dir / "ocr_latest.pt")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), save_dir / "ocr_best.pt")
                print(f"New best OCR model saved with accuracy: {best_val_accuracy:.4f}")

            completed_epochs = epoch + 1
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving current progress...")
        torch.save(model.state_dict(), save_dir / "ocr_interrupted.pt")
        torch.save(model.state_dict(), save_dir / "ocr_latest.pt")

    if history["train_loss"]:
        save_training_plots(history, save_dir)

        best_epoch_idx = max(range(len(history["val_accuracy"])), key=lambda idx: history["val_accuracy"][idx])
        summary = {
            "completed_epochs": completed_epochs,
            "interrupted": interrupted,
            "best_epoch": best_epoch_idx + 1,
            "best_val_accuracy": history["val_accuracy"][best_epoch_idx],
            "best_val_cer": history["val_cer"][best_epoch_idx],
            "history": history,
        }
        (save_dir / "ocr_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved latest model to: {save_dir / 'ocr_latest.pt'}")
    if (save_dir / "ocr_best.pt").exists():
        print(f"Saved best model to: {save_dir / 'ocr_best.pt'}")
    if interrupted:
        print(f"Saved interrupted checkpoint to: {save_dir / 'ocr_interrupted.pt'}")
    if history["train_loss"]:
        print(f"Saved curves to: {save_dir / 'ocr_training_curves.png'}")


if __name__ == "__main__":
    main()
