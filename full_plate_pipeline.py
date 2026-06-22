from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import torch

from crnn_model import CRNN
from label_converter import CTCLabelConverter
from plate_storage import PlateStorage


ALPHABET = "0123456789ABCEHKMOPTXY"
LETTER_SET = set("ABCEHKMOPTXY")
RUS_PLATE_RE = re.compile(r"^[ABCEHKMOPTXY]\d{3}[ABCEHKMOPTXY]{2}\d{2,3}$")
LETTER_TO_DIGIT = {"O": "0", "B": "8", "T": "7"}
DIGIT_TO_LETTER = {"0": "O", "8": "B", "7": "T", "4": "A", "3": "E", "1": "H"}
RUS_REGION_CODES = {
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15",
    "16", "17", "18", "19", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
    "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46",
    "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61",
    "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76",
    "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "92",
    "93", "94", "95", "96", "97", "98", "99", "102", "103", "113", "116", "121", "122", "123",
    "124", "125", "126", "134", "136", "138", "142", "147", "150", "152", "154", "159", "161",
    "163", "164", "173", "174", "177", "178", "186", "190", "193", "196", "197", "198", "199",
    "277", "299", "716", "750", "761", "763", "777", "790", "797", "799",
}
RU_LETTER_SUBS = {
    "A": ("A", "H", "M", "X"),
    "B": ("B", "8", "E"),
    "C": ("C", "O", "0"),
    "E": ("E", "B", "3"),
    "H": ("H", "A", "M", "1"),
    "K": ("K", "X", "H"),
    "M": ("M", "H", "T", "X"),
    "O": ("O", "0", "C"),
    "P": ("P", "T", "H"),
    "T": ("T", "P", "M", "7"),
    "X": ("X", "K", "Y"),
    "Y": ("Y", "X"),
    "0": ("0", "O", "C"),
    "1": ("1", "7", "H"),
    "3": ("3", "8", "E"),
    "4": ("4", "A", "H"),
    "5": ("5", "6", "8"),
    "6": ("6", "8", "5"),
    "7": ("7", "1", "T"),
    "8": ("8", "B", "3", "6"),
    "9": ("9", "7"),
}


def log_add_exp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]
    score: float
    source: str


@dataclass
class OCRResult:
    text: str
    confidence: float
    raw_text: str
    variant: str


@dataclass
class PlateReading:
    detection: DetectionResult
    ocr: OCRResult
    crop: np.ndarray
    plate_view: np.ndarray


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 2, x1))
    y1 = max(0, min(height - 2, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    return clamp_bbox(
        int(round(x1 - bw * pad_x)),
        int(round(y1 - bh * pad_y)),
        int(round(x2 + bw * pad_x)),
        int(round(y2 + bh * pad_y)),
        width,
        height,
    )


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def reduce_yellow_cast(bgr: np.ndarray) -> np.ndarray:
    x = bgr.astype(np.float32)
    means = [x[..., idx].mean() for idx in range(3)]
    target = sum(means) / 3.0
    for idx, value in enumerate(means):
        x[..., idx] *= target / (value + 1e-6)
    x = np.clip(x, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.82
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class YoloPlateDetector:
    def __init__(self, weights_path: str, device: str | None = None, imgsz: int = 960) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.device = device

    def detect(self, img_bgr: np.ndarray) -> DetectionResult | None:
        result = self.model.predict(
            source=img_bgr,
            verbose=False,
            imgsz=self.imgsz,
            conf=0.15,
            iou=0.45,
            device=self.device,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        idx = int(np.argmax(conf))
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy[idx]]
        h, w = img_bgr.shape[:2]
        bbox = clamp_bbox(x1, y1, x2, y2, w, h)
        return DetectionResult(bbox=bbox, score=float(conf[idx]), source="yolo")


def try_build_yolo(weights_path: str | None, device: str | None = None) -> YoloPlateDetector | None:
    if not weights_path:
        return None
    candidate_paths = [
        Path(weights_path),
        Path("weights/yolo_plate.pt"),
        Path("runs_yolo/plate_detector_ru/weights/best.pt"),
        Path("runs_yolo/plate_detector_ru2/weights/best.pt"),
        Path("runs/detect/runs_yolo/plate_detector_ru/weights/best.pt"),
        Path("runs/detect/runs_yolo/plate_detector_ru2/weights/best.pt"),
    ]

    discovered_paths = sorted(
        Path(".").glob("runs/**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidate_paths.extend(discovered_paths)

    resolved = next((path for path in candidate_paths if path.exists()), None)
    if resolved is None:
        return None
    try:
        return YoloPlateDetector(str(resolved), device=device)
    except Exception:
        return None


def score_candidate(
    rect: tuple[int, int, int, int],
    gray: np.ndarray,
    grad: np.ndarray,
) -> float:
    x, y, w, h = rect
    if h <= 0 or w <= 0:
        return -1.0
    ratio = w / float(h)
    area_ratio = (w * h) / float(gray.shape[0] * gray.shape[1])
    if not (2.0 <= ratio <= 7.5):
        return -1.0
    if not (0.002 <= area_ratio <= 0.45):
        return -1.0

    patch = gray[y : y + h, x : x + w]
    grad_patch = grad[y : y + h, x : x + w]
    if patch.size == 0 or grad_patch.size == 0:
        return -1.0

    brightness = float(patch.mean()) / 255.0
    contrast = float(patch.std()) / 64.0
    edge_density = float((grad_patch > 0).mean())
    ratio_score = 1.0 - min(abs(ratio - 4.4) / 4.4, 1.0)
    return 0.34 * ratio_score + 0.26 * edge_density + 0.24 * brightness + 0.16 * min(contrast, 1.0)


def classical_plate_search(img_bgr: np.ndarray) -> DetectionResult | None:
    gray = cv2.cvtColor(reduce_yellow_cast(img_bgr), cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
    )
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    denom = grad_x.max() - grad_x.min()
    if denom < 1e-6:
        return None
    grad_x = ((grad_x - grad_x.min()) / denom * 255).astype("uint8")
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    grad_x = cv2.morphologyEx(
        grad_x,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)),
    )
    thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)),
        iterations=1,
    )
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_score = -1.0
    best_bbox: tuple[int, int, int, int] | None = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        score = score_candidate((x, y, w, h), gray, thresh)
        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + w, y + h)

    if best_bbox is None or best_score < 0.18:
        return None

    height, width = img_bgr.shape[:2]
    bbox = expand_bbox(best_bbox, width, height, pad_x=0.06, pad_y=0.20)
    return DetectionResult(bbox=bbox, score=min(best_score, 0.99), source="classic")


def refine_bbox(img_bgr: np.ndarray, seed: DetectionResult) -> DetectionResult:
    height, width = img_bgr.shape[:2]
    roi_box = expand_bbox(seed.bbox, width, height, pad_x=0.55, pad_y=0.75)
    rx1, ry1, rx2, ry2 = roi_box
    roi = img_bgr[ry1:ry2, rx1:rx2]
    refined = classical_plate_search(roi)
    if refined is None:
        return seed

    x1, y1, x2, y2 = refined.bbox
    mapped = clamp_bbox(rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2, width, height)
    refined_full = DetectionResult(
        bbox=mapped,
        score=max(seed.score, refined.score),
        source=f"{seed.source}+classic",
    )

    if bbox_iou(seed.bbox, mapped) < 0.08 and seed.score >= 0.35:
        return seed
    return refined_full


def choose_detection(
    image: np.ndarray,
    yolo_detector: YoloPlateDetector,
) -> DetectionResult:
    chosen = yolo_detector.detect(image)
    if chosen is None:
        raise RuntimeError("YOLO did not detect a license plate on this image.")
    return chosen


def list_detection_candidates(
    image: np.ndarray,
    yolo_detector: YoloPlateDetector,
) -> list[DetectionResult]:
    yolo_result = yolo_detector.detect(image)
    if yolo_result is None:
        return []
    return [yolo_result]


def order_quad(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_quad(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))
    if max_width < 10 or max_height < 10:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def rectify_plate_crop(plate_bgr: np.ndarray) -> np.ndarray:
    corrected = reduce_yellow_cast(plate_bgr)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    best_quad: np.ndarray | None = None
    best_area = 0.0
    best_rect_width = 0
    for thresholded in (
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.Canny(gray, 70, 180),
    ):
        contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue
            area = cv2.contourArea(approx)
            if area < best_area:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(max(h, 1))
            if 2.0 <= ratio <= 7.5:
                best_area = area
                best_quad = approx.reshape(4, 2).astype("float32")
                best_rect_width = w

    use_quad = best_quad is not None
    if use_quad and best_rect_width < int(0.78 * corrected.shape[1]):
        # Guard against perspective crop that chops off the region block on the right.
        use_quad = False
    warped = four_point_transform(corrected, best_quad) if use_quad else corrected
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def preprocess_ocr_input(
    plate_bgr: np.ndarray,
    img_height: int = 32,
    img_width: int = 128,
) -> torch.Tensor:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    gray = gray.astype("float32") / 255.0
    gray = (gray - 0.5) / 0.5
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)


def preprocess_plate_simple(plate_bgr: np.ndarray) -> np.ndarray:
    corrected = reduce_yellow_cast(plate_bgr)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed = corrected
    best_quad: np.ndarray | None = None
    best_area = 0.0
    best_rect_width = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < best_area:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(max(h, 1))
        if 2.0 <= ratio <= 7.5:
            best_area = area
            best_quad = approx.reshape(4, 2).astype("float32")
            best_rect_width = w

    if best_quad is not None and best_rect_width >= int(0.78 * corrected.shape[1]):
        processed = four_point_transform(corrected, best_quad)
    if processed.shape[0] > processed.shape[1]:
        processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)

    target_width = 320
    if processed.shape[1] < target_width:
        scale = target_width / float(max(processed.shape[1], 1))
        scale = min(max(scale, 2.2), 4.2)
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        processed = cv2.resize(processed, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    return processed


def load_ocr_model(
    weights_path: str,
    device: torch.device,
    img_height: int = 32,
    img_width: int = 128,
) -> tuple[CRNN, CTCLabelConverter]:
    converter = CTCLabelConverter(ALPHABET)
    model = CRNN(img_h=img_height, img_w=img_width, num_classes=converter.num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, converter


def normalize_plate_text(text: str) -> str:
    text = re.sub(r"[^0-9A-Z]", "", text.upper())
    if not text:
        return ""
    prefer_region_len = 3 if len(text) >= 9 else 2 if len(text) >= 8 else 0

    def variants_for_char(ch: str, expect_letter: bool) -> list[str]:
        pool = list(dict.fromkeys(RU_LETTER_SUBS.get(ch, (ch,))))
        normalized: list[str] = []
        for candidate in pool:
            if expect_letter:
                candidate = DIGIT_TO_LETTER.get(candidate, candidate)
                if candidate in LETTER_SET:
                    normalized.append(candidate)
            else:
                candidate = LETTER_TO_DIGIT.get(candidate, candidate)
                if candidate.isdigit():
                    normalized.append(candidate)
        if not normalized:
            fallback = DIGIT_TO_LETTER.get(ch, ch) if expect_letter else LETTER_TO_DIGIT.get(ch, ch)
            if (expect_letter and fallback in LETTER_SET) or (not expect_letter and fallback.isdigit()):
                normalized.append(fallback)
        return list(dict.fromkeys(normalized))

    def score_candidate(candidate: str) -> float:
        score = 0.0
        if RUS_PLATE_RE.fullmatch(candidate):
            score += 3.0
        region = candidate[6:]
        if len(region) == prefer_region_len:
            score += 0.55
        elif len(region) == 3:
            score += 0.30
        elif len(region) == 2:
            score += 0.20
        if region in RUS_REGION_CODES:
            score += 0.08
        score += len(candidate) * 0.05
        return score

    best = text[:9]
    best_score = -1e9
    for region_len in (3, 2):
        required_len = 6 + region_len
        if len(text) < required_len:
            continue
        chars = text[:required_len]
        variant_lists = []
        for idx, ch in enumerate(chars):
            expect_letter = idx in {0, 4, 5}
            variant_lists.append(variants_for_char(ch, expect_letter))
        if any(not options for options in variant_lists):
            continue

        candidates = [""]
        for options in variant_lists:
            next_candidates: list[str] = []
            for prefix in candidates:
                for option in options[:3]:
                    next_candidates.append(prefix + option)
            candidates = next_candidates[:243]

        for candidate in candidates:
            score = score_candidate(candidate)
            if score > best_score:
                best_score = score
                best = candidate

    if best_score > -1e8:
        return best
    return text[:9]


def score_plate_text(text: str, confidence: float) -> float:
    if not text:
        return -1.0
    score = confidence
    clean = normalize_plate_text(text)
    if RUS_PLATE_RE.fullmatch(clean):
        score += 2.0
    elif len(clean) >= 8:
        score += 0.5
    score += min(len(clean), 9) * 0.03
    return score


def is_valid_ru_position(ch: str, pos: int) -> bool:
    if pos in {0, 4, 5}:
        return ch in LETTER_SET
    return ch.isdigit()


def is_valid_ru_prefix(text: str) -> bool:
    if len(text) > 9:
        return False
    for pos, ch in enumerate(text):
        if not is_valid_ru_position(ch, pos):
            return False
    return True


def is_valid_ru_complete(text: str) -> bool:
    return len(text) in {8, 9} and RUS_PLATE_RE.fullmatch(text) is not None


def decode_ru_ctc(
    logits: torch.Tensor,
    converter: CTCLabelConverter,
    beam_size: int = 48,
    topk: int = 8,
) -> tuple[str, float, str]:
    probs = logits.exp()
    greedy_indices = probs.argmax(2)
    greedy_text = converter.decode_greedy(greedy_indices)[0]
    log_probs = logits[:, 0, :].detach().cpu()

    beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}
    for t in range(log_probs.shape[0]):
        timestep = log_probs[t]
        values, indices = torch.topk(timestep, k=min(topk, timestep.shape[0]))
        next_beams: dict[str, tuple[float, float]] = {}

        blank_logp = float(timestep[0].item())
        for prefix, (pb, pnb) in beams.items():
            cur_pb, cur_pnb = next_beams.get(prefix, (-math.inf, -math.inf))
            cur_pb = log_add_exp(cur_pb, pb + blank_logp)
            cur_pb = log_add_exp(cur_pb, pnb + blank_logp)
            next_beams[prefix] = (cur_pb, cur_pnb)

            last_char = prefix[-1] if prefix else ""
            for value, index in zip(values.tolist(), indices.tolist()):
                if index == 0:
                    continue
                ch = converter.idx_to_char.get(index)
                if ch is None:
                    continue

                if ch == last_char:
                    cur_pb2, cur_pnb2 = next_beams.get(prefix, (-math.inf, -math.inf))
                    cur_pnb2 = log_add_exp(cur_pnb2, pnb + float(value))
                    next_beams[prefix] = (cur_pb2, cur_pnb2)

                new_prefix = prefix + ch
                if not is_valid_ru_prefix(new_prefix):
                    continue

                cur_pb2, cur_pnb2 = next_beams.get(new_prefix, (-math.inf, -math.inf))
                if ch == last_char:
                    cur_pnb2 = log_add_exp(cur_pnb2, pb + float(value))
                else:
                    cur_pnb2 = log_add_exp(cur_pnb2, pb + float(value))
                    cur_pnb2 = log_add_exp(cur_pnb2, pnb + float(value))
                next_beams[new_prefix] = (cur_pb2, cur_pnb2)

        sorted_beams = sorted(
            next_beams.items(),
            key=lambda item: log_add_exp(item[1][0], item[1][1]),
            reverse=True,
        )
        beams = dict(sorted_beams[:beam_size])

    best_text = ""
    best_score = -math.inf
    fallback_text = ""
    fallback_score = -math.inf
    for prefix, (pb, pnb) in beams.items():
        score = log_add_exp(pb, pnb)
        if is_valid_ru_complete(prefix) and score > best_score:
            best_text = prefix
            best_score = score
        if is_valid_ru_prefix(prefix) and len(prefix) >= len(fallback_text):
            if len(prefix) > len(fallback_text) or score > fallback_score:
                fallback_text = prefix
                fallback_score = score

    chosen = best_text or fallback_text or normalize_plate_text(greedy_text)
    chosen_score = best_score if best_text else fallback_score
    if chosen_score == -math.inf:
        confidence = float(probs.max(2).values.mean().item())
    else:
        confidence = float(max(0.0, min(1.0, math.exp(chosen_score / max(len(chosen), 1)))))
    return chosen, confidence, greedy_text


def recognize_plate_text(
    plate_bgr: np.ndarray,
    ocr_model: torch.nn.Module,
    converter: CTCLabelConverter,
    device: torch.device,
    img_height: int = 32,
    img_width: int = 128,
) -> tuple[str, float, str]:
    x = preprocess_ocr_input(plate_bgr, img_height=img_height, img_width=img_width).to(device)
    with torch.no_grad():
        logits = ocr_model(x)
        text, confidence, raw_text = decode_ru_ctc(logits, converter)
    return text, confidence, raw_text


def run_ocr_simple(
    plate_bgr: np.ndarray,
    ocr_model: torch.nn.Module,
    converter: CTCLabelConverter,
    device: torch.device,
) -> tuple[OCRResult, np.ndarray]:
    processed = preprocess_plate_simple(plate_bgr)
    text, confidence, raw_text = recognize_plate_text(processed, ocr_model, converter, device)
    text = normalize_plate_text(text)
    if confidence < 0.28 and not RUS_PLATE_RE.fullmatch(text):
        text = ""
    return OCRResult(text=text, confidence=confidence, raw_text=raw_text, variant="simple"), processed


def generate_ocr_variants(plate_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    def upscale_for_ocr(img: np.ndarray, target_width: int = 320) -> np.ndarray:
        if img.shape[1] < target_width:
            scale = target_width / float(max(img.shape[1], 1))
            scale = min(max(scale, 2.0), 3.4)
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    raw_crop = upscale_for_ocr(reduce_yellow_cast(plate_bgr))
    plate = upscale_for_ocr(rectify_plate_crop(plate_bgr))
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray_soft = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu = cv2.threshold(gray_soft, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        gray_soft,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)

    def to_bgr(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return [
        ("raw_crop", raw_crop),
        ("raw_rectified", plate),
        ("gray", to_bgr(gray)),
        ("gray_soft", to_bgr(gray_soft)),
        ("clahe", to_bgr(clahe)),
        ("otsu", to_bgr(otsu)),
        ("adaptive", to_bgr(adaptive)),
    ]


def run_ocr_ensemble(
    plate_bgr: np.ndarray,
    ocr_model: torch.nn.Module,
    converter: CTCLabelConverter,
    device: torch.device,
) -> tuple[OCRResult, np.ndarray]:
    def restore_region_from_raw(result: OCRResult) -> OCRResult:
        raw_norm = normalize_plate_text(result.raw_text)
        if len(result.text) == 8 and len(raw_norm) == 9:
            if raw_norm.startswith(result.text):
                return OCRResult(
                    text=raw_norm,
                    confidence=result.confidence,
                    raw_text=result.raw_text,
                    variant=result.variant,
                )
            if raw_norm[:6] == result.text[:6]:
                return OCRResult(
                    text=raw_norm,
                    confidence=result.confidence,
                    raw_text=result.raw_text,
                    variant=result.variant,
                )
        return result

    variant_bias = {
        "raw_crop": 0.46,
        "raw_rectified": 0.30,
        "gray": 0.22,
        "gray_soft": 0.16,
        "clahe": 0.10,
        "otsu": 0.00,
        "adaptive": -0.05,
    }
    candidates: list[tuple[OCRResult, np.ndarray, float, str]] = []
    for variant_name, variant_image in generate_ocr_variants(plate_bgr):
        text, confidence, raw_text = recognize_plate_text(variant_image, ocr_model, converter, device)
        text = normalize_plate_text(text)
        result = OCRResult(text=text, confidence=confidence, raw_text=raw_text, variant=variant_name)
        stable_text = result.text or normalize_plate_text(result.raw_text)
        base_score = score_plate_text(stable_text, result.confidence) + variant_bias.get(variant_name, 0.0)
        candidates.append((result, variant_image, base_score, stable_text))

    if not candidates:
        fallback = OCRResult(text="", confidence=0.0, raw_text="", variant="none")
        return fallback, plate_bgr

    # ANPR-style stabilization: aggregate agreement across multiple OCR variants.
    grouped: dict[str, dict[str, object]] = {}
    for result, variant_image, base_score, stable_text in candidates:
        key = stable_text.strip()
        if not key:
            continue
        entry = grouped.setdefault(
            key,
            {"score_sum": 0.0, "votes": 0, "best_result": result, "best_image": variant_image, "best_score": -1e9},
        )
        entry["score_sum"] = float(entry["score_sum"]) + base_score
        entry["votes"] = int(entry["votes"]) + 1
        if base_score > float(entry["best_score"]):
            entry["best_score"] = base_score
            entry["best_result"] = result
            entry["best_image"] = variant_image

    if grouped:
        best_key = ""
        best_group_score = -1e9
        for key, entry in grouped.items():
            votes = int(entry["votes"])
            avg_score = float(entry["score_sum"]) / max(votes, 1)
            consensus_bonus = min(0.45, 0.17 * (votes - 1))
            format_bonus = 0.35 if RUS_PLATE_RE.fullmatch(key) else 0.0
            group_score = avg_score + consensus_bonus + format_bonus
            if group_score > best_group_score:
                best_group_score = group_score
                best_key = key

        chosen = grouped[best_key]
        best_result = chosen["best_result"]
        best_image = chosen["best_image"]
        assert isinstance(best_result, OCRResult)
        assert isinstance(best_image, np.ndarray)
        if best_result.confidence < 0.28 and not RUS_PLATE_RE.fullmatch(best_result.text):
            best_result = OCRResult(
                text="",
                confidence=best_result.confidence,
                raw_text=best_result.raw_text,
                variant=best_result.variant,
            )
        best_result = restore_region_from_raw(best_result)
        return best_result, best_image

    best_result, best_variant_image, _, _ = max(candidates, key=lambda item: item[2])
    if best_result.confidence < 0.28 and not RUS_PLATE_RE.fullmatch(best_result.text):
        best_result = OCRResult(
            text="",
            confidence=best_result.confidence,
            raw_text=best_result.raw_text,
            variant=best_result.variant,
        )
    best_result = restore_region_from_raw(best_result)
    return best_result, best_variant_image


def score_reading(detection: DetectionResult, ocr: OCRResult) -> float:
    score = score_plate_text(ocr.text, ocr.confidence)
    if detection.source.startswith("yolo"):
        score += 0.45

    if detection.score < 0.18:
        score -= 0.40
    elif detection.score > 0.55:
        score += 0.20

    text = ocr.text
    if len(text) < 7:
        score -= 0.7
    if RUS_PLATE_RE.fullmatch(text):
        score += 1.5

    return score


def select_best_reading(
    image: np.ndarray,
    yolo_detector: YoloPlateDetector,
    ocr_model: torch.nn.Module,
    converter: CTCLabelConverter,
    device: torch.device,
) -> PlateReading:
    height, width = image.shape[:2]
    candidates = list_detection_candidates(image, yolo_detector)
    if not candidates:
        raise RuntimeError("YOLO did not detect a license plate on this image.")
    best_reading: PlateReading | None = None
    best_score = -1e9

    for detection in candidates:
        for pad_x, pad_y in ((0.12, 0.24), (0.18, 0.38), (0.24, 0.48)):
            crop_box = expand_bbox(detection.bbox, width, height, pad_x=pad_x, pad_y=pad_y)
            x1, y1, x2, y2 = crop_box
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            ocr_result, plate_view = run_ocr_ensemble(crop, ocr_model, converter, device)
            total_score = score_reading(detection, ocr_result)

            if best_reading is None or total_score > best_score:
                best_score = total_score
                best_reading = PlateReading(
                    detection=detection,
                    ocr=ocr_result,
                    crop=crop,
                    plate_view=plate_view,
                )

    if best_reading is None:
        fallback_detection = choose_detection(image, yolo_detector)
        crop_box = expand_bbox(fallback_detection.bbox, width, height, pad_x=0.18, pad_y=0.38)
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2].copy()
        ocr_result, plate_view = run_ocr_ensemble(crop, ocr_model, converter, device)
        return PlateReading(
            detection=fallback_detection,
            ocr=ocr_result,
            crop=crop,
            plate_view=plate_view,
        )

    return best_reading


def gather_images(images_dir: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)


def process_image(
    image_path: str,
    output_dir: Path,
    yolo_detector: YoloPlateDetector,
    ocr_model: torch.nn.Module,
    converter: CTCLabelConverter,
    device: torch.device,
    storage: PlateStorage | None = None,
) -> dict[str, object] | None:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    reading = select_best_reading(
        image=image,
        yolo_detector=yolo_detector,
        ocr_model=ocr_model,
        converter=converter,
        device=device,
    )
    detection = reading.detection
    crop = reading.crop
    ocr_result = reading.ocr
    ocr_view = reading.plate_view

    vis = image.copy()
    label = f"{ocr_result.text or ocr_result.raw_text} | {detection.source} | {detection.score:.2f}"
    draw_bbox(vis, detection.bbox, label)

    stem = Path(image_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    vis_path = output_dir / f"vis_{stem}.jpg"
    crop_path = crops_dir / f"crop_{stem}.png"
    plate_path = crops_dir / f"plate_{stem}.png"
    cv2.imwrite(str(vis_path), vis)
    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(plate_path), ocr_view)

    if storage is not None:
        storage.insert_read(
            image_path=image_path,
            plate_text=ocr_result.text,
            raw_text=ocr_result.raw_text,
            detector_source=detection.source,
            detector_score=detection.score,
            ocr_confidence=ocr_result.confidence,
            bbox=detection.bbox,
            vis_path=str(vis_path),
            crop_path=str(crop_path),
            plate_image_path=str(plate_path),
        )

    return {
        "image_path": image_path,
        "plate_text": ocr_result.text,
        "raw_text": ocr_result.raw_text,
        "ocr_variant": ocr_result.variant,
        "detector_source": detection.source,
        "detector_score": detection.score,
        "ocr_confidence": ocr_result.confidence,
        "bbox": detection.bbox,
        "vis_path": str(vis_path),
        "crop_path": str(crop_path),
        "plate_path": str(plate_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plate recognition pipeline with YOLO + CRNN.")
    parser.add_argument("--images-dir", default="my_images")
    parser.add_argument("--ocr-weights", default="ocr_checkpoints/ocr_best.pt")
    parser.add_argument("--yolo-weights", default="weights/yolo_plate.pt")
    parser.add_argument("--output-dir", default="outputs_pipeline")
    parser.add_argument("--pg-dsn", default=None, help="PostgreSQL DSN. Defaults to PLATE_DB_DSN env var.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--disable-db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = gather_images(args.images_dir)
    if not files:
        raise RuntimeError(f"No images found in: {args.images_dir}")

    yolo_detector = try_build_yolo(args.yolo_weights, device=str(device))
    if yolo_detector is None:
        raise RuntimeError(
            "YOLO weights were not found. Set --yolo-weights or place a model at weights/yolo_plate.pt."
        )
    ocr_model, converter = load_ocr_model(args.ocr_weights, device)
    storage = None if args.disable_db else PlateStorage(args.pg_dsn)

    print(f"Device: {device}")
    print(f"Images found: {len(files)}")
    print("YOLO detector: enabled")

    for index, image_path in enumerate(files[: args.limit], start=1):
        result = process_image(
            image_path=image_path,
            output_dir=Path(args.output_dir),
            yolo_detector=yolo_detector,
            ocr_model=ocr_model,
            converter=converter,
            device=device,
            storage=storage,
        )
        if result is None:
            print(f"[{index}] skipped: {image_path}")
            continue

        print(f"[{index}] {result['image_path']}")
        print(
            f" plate={result['plate_text']} raw={result['raw_text']} variant={result['ocr_variant']}"
        )
        print(
            f" detector={result['detector_source']} det_conf={result['detector_score']:.3f} "
            f"ocr_conf={result['ocr_confidence']:.3f}"
        )
        print(f" bbox={result['bbox']}")
        print(f" vis={result['vis_path']}")
        print("-" * 60)

    if storage is not None:
        storage.close()


if __name__ == "__main__":
    main()
