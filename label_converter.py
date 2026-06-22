from __future__ import annotations

from typing import List

import torch


class CTCLabelConverter:
    def __init__(self, alphabet: str) -> None:
        self.alphabet = alphabet

        # 0 reserved for CTC blank
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(alphabet)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(alphabet)}

    @property
    def num_classes(self) -> int:
        return len(self.alphabet) + 1  # + blank

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = [len(t) for t in texts]
        encoded: List[int] = []

        for text in texts:
            for ch in text:
                if ch not in self.char_to_idx:
                    raise ValueError(f"Unknown character: {ch}")
                encoded.append(self.char_to_idx[ch])

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    def decode_greedy(self, preds_indices: torch.Tensor) -> list[str]:
        """
        preds_indices: [T, B]
        """
        preds_indices = preds_indices.detach().cpu()
        t, b = preds_indices.shape
        results: list[str] = []

        for batch_idx in range(b):
            seq = preds_indices[:, batch_idx].tolist()

            decoded = []
            prev = None
            for idx in seq:
                if idx != 0 and idx != prev:
                    decoded.append(self.idx_to_char[idx])
                prev = idx

            results.append("".join(decoded))

        return results