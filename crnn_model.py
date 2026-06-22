from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(
        self,
        img_h: int,
        num_classes: int,
        img_w: int = 128,
        lstm_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=512 * 2,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already resized in the training/inference transform.
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.reshape(batch, channels * height, width)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)
        return F.log_softmax(x, dim=2)
