from __future__ import annotations

import random

import cv2
import numpy as np


def apply_motion_blur(gray: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(gray, -1, kernel)


def apply_perspective(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dx = 0.08 * w
    dy = 0.18 * h
    dst = src.copy()
    dst[:, 0] += np.random.uniform(-dx, dx, size=4).astype(np.float32)
    dst[:, 1] += np.random.uniform(-dy, dy, size=4).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_occlusion(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    out = gray.copy()
    for _ in range(random.randint(1, 2)):
        occ_w = random.randint(max(2, w // 18), max(4, w // 10))
        x1 = random.randint(0, max(0, w - occ_w))
        value = random.randint(140, 255)
        cv2.rectangle(out, (x1, 0), (min(w - 1, x1 + occ_w), h - 1), value, -1)
    return out


def augment_ocr_image(gray: np.ndarray) -> np.ndarray:
    out = gray.copy()

    if random.random() < 0.55:
        alpha = random.uniform(0.7, 1.35)
        beta = random.uniform(-28, 28)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if random.random() < 0.35:
        gamma = random.uniform(0.65, 1.45)
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(out, lut)

    if random.random() < 0.35:
        out = apply_perspective(out)

    if random.random() < 0.30:
        out = apply_motion_blur(out)

    if random.random() < 0.30:
        out = cv2.GaussianBlur(out, random.choice([(3, 3), (5, 5)]), 0)

    if random.random() < 0.20:
        noise = np.random.normal(0, random.uniform(4, 14), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.18:
        out = apply_occlusion(out)

    if random.random() < 0.28:
        clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.5), tileGridSize=(8, 8))
        out = clahe.apply(out)

    if random.random() < 0.15:
        thresh_type = cv2.THRESH_BINARY if random.random() < 0.5 else cv2.THRESH_BINARY_INV
        out = cv2.threshold(out, 0, 255, thresh_type | cv2.THRESH_OTSU)[1]
        if thresh_type == cv2.THRESH_BINARY_INV:
            out = 255 - out

    return out
