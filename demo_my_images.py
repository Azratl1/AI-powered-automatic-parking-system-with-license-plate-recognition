from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from full_plate_pipeline import (
    draw_bbox,
    gather_images,
    load_ocr_model,
    run_ocr_ensemble,
    score_plate_text,
    YoloPlateDetector,
)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0:
        return image
    scale = height / float(h)
    width = max(1, int(round(w * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def make_text_panel(lines: list[str], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 36
    for line in lines:
        cv2.putText(
            panel,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        y += 38
    return panel


def pad_to_size(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = np.full((target_h, target_w, 3), 250, dtype=np.uint8)
    y = max(0, (target_h - h) // 2)
    x = max(0, (target_w - w) // 2)
    canvas[y : y + h, x : x + w] = image[:target_h, :target_w]
    return canvas


def build_canvas(
    image_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    plate_bgr: np.ndarray,
    title_lines: list[str],
) -> np.ndarray:
    vis_h = 520
    left = resize_to_height(image_bgr, vis_h)
    crop = resize_to_height(ensure_bgr(crop_bgr), 240)
    plate = resize_to_height(ensure_bgr(plate_bgr), 140)
    text_h = max(140, 24 + 38 * len(title_lines))
    text = make_text_panel(title_lines, width=max(crop.shape[1], plate.shape[1], 420), height=text_h)

    right_w = max(crop.shape[1], plate.shape[1], text.shape[1])
    crop = pad_to_size(crop, 240, right_w)
    plate = pad_to_size(plate, 140, right_w)
    text = pad_to_size(text, text_h, right_w)

    right = np.vstack([crop, plate, text])
    right = pad_to_size(right, left.shape[0], right.shape[1])

    gap = np.full((left.shape[0], 20, 3), 235, dtype=np.uint8)
    return np.hstack([left, gap, right])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual check for YOLO + OCR on my_images.")
    parser.add_argument("--images-dir", default="my_images")
    parser.add_argument("--ocr-weights", default="ocr_checkpoints/ocr_best.pt")
    parser.add_argument("--yolo-weights", default="runs/detect/runs_yolo/plate_detector_ru2/weights/best.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-dir", default="outputs_demo")
    return parser.parse_args()


def expand_bbox_lr(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_left: float,
    pad_right: float,
    pad_top: float,
    pad_bottom: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(round(x1 - bw * pad_left)))
    y1 = max(0, int(round(y1 - bh * pad_top)))
    x2 = min(width, int(round(x2 + bw * pad_right)))
    y2 = min(height, int(round(y2 + bh * pad_bottom)))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def choose_best_ocr_for_detection(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    ocr_model: torch.nn.Module,
    converter,
    device: torch.device,
):
    h, w = image.shape[:2]
    pad_variants = [
        (0.00, 0.10, 0.10, 0.12),
        (0.04, 0.20, 0.16, 0.22),
        (0.10, 0.28, 0.20, 0.30),
        (0.00, 0.42, 0.14, 0.24),
        (0.06, 0.48, 0.18, 0.28),
    ]

    best_score = -1e9
    best = None

    for pad_left, pad_right, pad_top, pad_bottom in pad_variants:
        x1, y1, x2, y2 = expand_bbox_lr(
            bbox,
            w,
            h,
            pad_left=pad_left,
            pad_right=pad_right,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
        )
        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        ocr_result, plate_view = run_ocr_ensemble(crop, ocr_model, converter, device)
        text = ocr_result.text or ""

        score = score_plate_text(text, ocr_result.confidence)
        if len(text) == 8:
            score += 0.12
        elif len(text) == 9:
            score += 0.06
        if not text and ocr_result.raw_text:
            score -= 0.20

        if score > best_score:
            best_score = score
            best = (ocr_result, crop, plate_view, (x1, y1, x2, y2))

    return best


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = gather_images(args.images_dir)
    if not files:
        raise RuntimeError(f"No images found in: {args.images_dir}")
    if args.limit > 0:

        files = files[: args.limit]

    weights_path = Path(args.yolo_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    yolo_detector = YoloPlateDetector(str(weights_path), device=str(device))
    ocr_model, converter = load_ocr_model(args.ocr_weights, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Images: {len(files)}")
    print(f"YOLO: {weights_path}")

    for index, image_path in enumerate(files, start=1):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[{index}] skip unreadable: {image_path}")
            continue

        detection = yolo_detector.detect(image)
        if detection is None:
            print(f"[{index}] no yolo detection: {image_path}")
            continue

        best_ocr = choose_best_ocr_for_detection(
            image=image,
            bbox=detection.bbox,
            ocr_model=ocr_model,
            converter=converter,
            device=device,
        )
        if best_ocr is None:
            print(f"[{index}] empty crop after variants: {image_path}")
            continue
        ocr_result, crop, plate_view, vis_bbox = best_ocr

        vis = image.copy()
        draw_bbox(
            vis,
            vis_bbox,
            f"{ocr_result.text or ocr_result.raw_text} | {detection.source} | {detection.score:.2f}",
        )

        canvas = build_canvas(
            vis,
            crop,
            plate_view,
            [
                f"file: {Path(image_path).name}",
                f"plate: {ocr_result.text or '-'}",
                f"raw: {ocr_result.raw_text or '-'}",
                f"variant: {ocr_result.variant}",
                f"detector: {detection.source}  det={detection.score:.2f}  ocr={ocr_result.confidence:.2f}",
            ],
        )

        out_path = save_dir / f"demo_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(out_path), canvas)

        print(f"[{index}] {image_path}")
        print(f" plate={ocr_result.text} raw={ocr_result.raw_text} variant={ocr_result.variant}")
        print(f" saved={out_path}")

        cv2.imshow("Plate Demo", canvas)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
