import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def read_yolo_bbox(label_path: str, mode="best_plate"):

    if not os.path.exists(label_path):
        return None

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
    except Exception:
        return None

    if not lines:
        return None

    boxes = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue

        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        w  = float(np.clip(w, 0.0, 1.0))
        h  = float(np.clip(h, 0.0, 1.0))

        boxes.append((cx, cy, w, h))

    if not boxes:
        return None

    if mode == "largest":
        return max(boxes, key=lambda b: b[2]*b[3])

    if mode == "lowest":
        return max(boxes, key=lambda b: b[1])  # cy больше -> ниже

    if mode == "center":
        return min(boxes, key=lambda b: (b[0]-0.5)**2 + (b[1]-0.5)**2)

    # mode == "best_plate"
    # скоринг: похожесть на номер (wide ratio) + адекватная площадь + ближе к низу
    best = None
    best_score = -1e9
    for (cx, cy, w, h) in boxes:
        area = w * h
        ratio = (w / (h + 1e-6))

        # фильтр по "номерности": слишком квадратные/высокие отбрасываем
        if ratio < 1.8 or ratio > 8.0:
            continue
        if area < 1e-4:
            continue

        # score: предпочитаем широкие, нормальной площади и чуть ниже по кадру
        score = 0.0
        score += 2.0 * min(ratio, 6.0)          # ширина важна
        score += 500.0 * area                   # площадь
        score += 1.5 * cy                       # чуть ниже предпочтительнее
        # штраф если слишком высоко
        score -= 1.0 * max(0.0, 0.35 - cy)

        if score > best_score:
            best_score = score
            best = (cx, cy, w, h)

    # если после фильтра ничего не осталось — fallback на largest
    return best if best is not None else max(boxes, key=lambda b: b[2]*b[3])


# -------------------- Basic augs --------------------

def aug_brightness_contrast(img_bgr: np.ndarray, p=0.6):
    if random.random() > p:
        return img_bgr
    alpha = random.uniform(0.6, 1.4)
    beta = random.uniform(-35, 35)
    return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)


def aug_gamma(img_bgr: np.ndarray, p=0.5, gamma_range=(0.6, 1.6)):
    if random.random() > p:
        return img_bgr
    g = random.uniform(*gamma_range)
    lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, lut)


def aug_gaussian_blur(img_bgr: np.ndarray, p=0.25):
    if random.random() > p:
        return img_bgr
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img_bgr, (k, k), 0)


def aug_horizontal_flip(img_bgr: np.ndarray, bbox_cxcywh, p=0.5):
    if random.random() > p:
        return img_bgr, bbox_cxcywh
    img_bgr = cv2.flip(img_bgr, 1)
    cx, cy, w, h = bbox_cxcywh
    cx = 1.0 - cx
    return img_bgr, (cx, cy, w, h)


def aug_low_contrast(img_bgr, alpha_range=(0.35, 0.85), beta_range=(-10, 10)):
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)


def aug_fog_fast(img_bgr: np.ndarray, strength_range=(0.25, 0.70), blur_ks=(11, 31)):
    h, w = img_bgr.shape[:2]
    strength = random.uniform(*strength_range)

    fog_color = random.randint(200, 255)
    fog = np.full((h, w, 3), fog_color, dtype=np.uint8)

    out = cv2.addWeighted(img_bgr, 1.0 - strength, fog, strength, 0)

    k = random.randrange(blur_ks[0], blur_ks[1] + 1, 2)
    out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def aug_glare(img_bgr: np.ndarray, p=0.6):
    if random.random() > p:
        return img_bgr

    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    n = random.randint(1, 3)
    for _ in range(n):
        cx = random.randint(int(0.15 * w), int(0.85 * w))
        cy = random.randint(int(0.15 * h), int(0.85 * h))
        r = random.randint(int(0.04 * min(h, w)), int(0.14 * min(h, w)))
        val = random.randint(200, 255)
        cv2.circle(overlay, (cx, cy), r, (val, val, val), -1)

    k = random.choice([21, 31, 41])
    overlay = cv2.GaussianBlur(overlay, (k, k), 0)

    alpha = random.uniform(0.15, 0.40)
    out = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
    return out


def aug_motion_blur(img_bgr: np.ndarray, p=0.25, k_range=(7, 19)):
    if random.random() > p:
        return img_bgr

    k = random.randrange(k_range[0], k_range[1] + 1, 2)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= float(k)
    return cv2.filter2D(img_bgr, -1, kernel)


def aug_night(img_bgr: np.ndarray, p=0.35):
    if random.random() > p:
        return img_bgr

    out = img_bgr
    out = aug_gamma(out, p=1.0, gamma_range=(0.45, 0.85))
    out = aug_low_contrast(out, alpha_range=(0.35, 0.75), beta_range=(-25, 5))

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = (s.astype(np.float32) * random.uniform(0.4, 0.9)).clip(0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    return out


def aug_bad_visibility_combo(img_bgr: np.ndarray):
    if random.random() < 0.50:
        return img_bgr

    out = img_bgr

    if random.random() < 0.30:
        out = aug_fog_fast(out, strength_range=(0.25, 0.65))

    out = aug_night(out, p=0.35)

    if random.random() < 0.40:
        out = aug_glare(out, p=0.9)

    out = aug_motion_blur(out, p=0.25)
    return out


# -------------------- Dataset --------------------

class PlateBBoxDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        img_size: int = 256,
        training: bool = True,
        apply_aug: bool = True
    ):
        self.img_size = int(img_size)
        self.training = bool(training)
        self.apply_aug = bool(apply_aug)
        self.labels_dir = labels_dir

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.images = []
        for e in exts:
            self.images += glob.glob(os.path.join(images_dir, e))
        self.images = sorted(self.images)

        missing = 0
        for img_path in self.images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            lp = os.path.join(labels_dir, name + ".txt")
            if not os.path.exists(lp):
                missing += 1

        print(f"[Dataset] {images_dir}: {len(self.images)} images, missing labels: {missing} "
              f"({missing / max(1, len(self.images)) * 100:.1f}%)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # try multiple times to avoid corrupted images
        for _ in range(12):
            img_path = self.images[idx]
            name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.labels_dir, name + ".txt")

            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            except cv2.error:
                img = None

            # ---- anti-crash checks ----
            if img is None or not isinstance(img, np.ndarray):
                idx = random.randint(0, len(self.images) - 1)
                continue
            if img.ndim != 3 or img.shape[2] != 3:
                idx = random.randint(0, len(self.images) - 1)
                continue
            if img.dtype != np.uint8:
                # make sure dtype is ok for OpenCV ops
                img = img.astype(np.uint8, copy=False)

            # important: avoid OpenCV sharing memory issues
            img = img.copy()

            bbox = read_yolo_bbox(label_path)
            if bbox is None:
                bbox = (0.5, 0.5, 0.0, 0.0)

            # resize first
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            if self.training and self.apply_aug:
                img, bbox = aug_horizontal_flip(img, bbox, p=0.5)

                img = aug_brightness_contrast(img, p=0.6)
                img = aug_gamma(img, p=0.5, gamma_range=(0.7, 1.5))
                img = aug_gaussian_blur(img, p=0.20)

                img = aug_bad_visibility_combo(img)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            target = torch.tensor(bbox, dtype=torch.float32)
            return img, target, img_path

        raise RuntimeError("Too many failed image reads. Possibly corrupted images in dataset.")