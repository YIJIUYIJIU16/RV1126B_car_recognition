import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_CLASSES = {0, 1}
CAR_CLASS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate three calibration sets for car recognition RKNN models."
    )
    parser.add_argument("--images", required=True, help="Root directory of source images.")
    parser.add_argument("--labels", required=True, help="Root directory of YOLO-format label txt files.")
    parser.add_argument(
        "--output-root",
        default="../model/calibration_sets",
        help="Output directory for generated calibration images and txt lists.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed used for sampling.")
    parser.add_argument(
        "--max-detect",
        type=int,
        default=300,
        help="Maximum number of full images to keep for detect calibration.",
    )
    parser.add_argument(
        "--max-plate",
        type=int,
        default=500,
        help="Maximum number of plate crops to keep for plate_rec_color calibration.",
    )
    parser.add_argument(
        "--max-car",
        type=int,
        default=500,
        help="Maximum number of car crops to keep for car_recognition calibration.",
    )
    parser.add_argument(
        "--split-double-plate",
        action="store_true",
        help="Apply double-plate split/merge for class 1 plate crops.",
    )
    parser.add_argument(
        "--detect-copy",
        action="store_true",
        help="Copy detect images into output-root/detect instead of listing original paths.",
    )
    return parser.parse_args()


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts.astype(np.float32))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(int(round(width_a)), int(round(width_b)), 1)
    max_height = max(int(round(height_a)), int(round(height_b)), 1)
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def split_double_plate(img):
    h, _, _ = img.shape
    upper = img[0 : int(5 / 12 * h), :]
    lower = img[int(1 / 3 * h) :, :]
    upper = cv2.resize(upper, (lower.shape[1], lower.shape[0]))
    return np.hstack((upper, lower))


def iter_images(images_root):
    for path in sorted(images_root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path


def label_path_for(image_path, images_root, labels_root):
    rel = image_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


def clamp_rect(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    return x1, y1, x2, y2


def parse_label_line(line, width, height):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    values = [float(v) for v in parts]
    cls_id = int(values[0])
    cx, cy, bw, bh = values[1:5]
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height

    landmarks = None
    if len(values) >= 13:
        raw_landmarks = values[5:13]
        if all(v >= 0 for v in raw_landmarks):
            landmarks = np.array(
                [[raw_landmarks[i] * width, raw_landmarks[i + 1] * height] for i in range(0, 8, 2)],
                dtype=np.float32,
            )

    return {
        "cls_id": cls_id,
        "bbox": (x1, y1, x2, y2),
        "landmarks": landmarks,
    }


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_detect_image(src_path, dst_dir):
    dst_path = dst_dir / src_path.name
    stem = src_path.stem
    suffix = src_path.suffix
    idx = 1
    while dst_path.exists():
        dst_path = dst_dir / f"{stem}_{idx}{suffix}"
        idx += 1
    shutil.copy2(src_path, dst_path)
    return dst_path.resolve()


def save_crop(img, dst_dir, prefix, stem, counter):
    dst_path = dst_dir / f"{stem}_{prefix}_{counter:06d}.jpg"
    ok = cv2.imwrite(str(dst_path), img)
    return dst_path.resolve() if ok else None


def main():
    args = parse_args()
    random.seed(args.seed)

    images_root = Path(args.images).resolve()
    labels_root = Path(args.labels).resolve()
    output_root = Path(args.output_root).resolve()

    detect_dir = ensure_dir(output_root / "detect")
    plate_dir = ensure_dir(output_root / "plate_rec")
    car_dir = ensure_dir(output_root / "car_color")

    detect_records = []
    plate_candidates = []
    car_candidates = []

    image_paths = list(iter_images(images_root))
    for image_path in image_paths:
        label_path = label_path_for(image_path, images_root, labels_root)
        if not label_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            continue

        height, width = img.shape[:2]
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        parsed = [parse_label_line(line, width, height) for line in lines]
        parsed = [item for item in parsed if item is not None]
        if not parsed:
            continue

        detect_records.append(image_path.resolve())

        plate_count = 0
        car_count = 0
        for item in parsed:
            cls_id = item["cls_id"]
            x1, y1, x2, y2 = clamp_rect(*item["bbox"], width, height)
            if x2 <= x1 or y2 <= y1:
                continue

            if cls_id in PLATE_CLASSES:
                if item["landmarks"] is None:
                    continue
                crop = four_point_transform(img, item["landmarks"])
                if crop is None or crop.size == 0:
                    continue
                if cls_id == 1 and args.split_double_plate:
                    crop = split_double_plate(crop)
                if crop is None or crop.size == 0:
                    continue
                plate_candidates.append((crop, image_path.stem, plate_count))
                plate_count += 1
            elif cls_id == CAR_CLASS:
                crop = img[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                car_candidates.append((crop, image_path.stem, car_count))
                car_count += 1

    random.shuffle(detect_records)
    random.shuffle(plate_candidates)
    random.shuffle(car_candidates)

    detect_records = detect_records[: args.max_detect]
    plate_candidates = plate_candidates[: args.max_plate]
    car_candidates = car_candidates[: args.max_car]

    detect_list = []
    for image_path in detect_records:
        if args.detect_copy:
            saved = copy_detect_image(image_path, detect_dir)
            detect_list.append(saved)
        else:
            detect_list.append(image_path)

    plate_list = []
    for idx, (crop, stem, local_idx) in enumerate(plate_candidates):
        saved = save_crop(crop, plate_dir, "plate", stem, idx + local_idx)
        if saved is not None:
            plate_list.append(saved)

    car_list = []
    for idx, (crop, stem, local_idx) in enumerate(car_candidates):
        saved = save_crop(crop, car_dir, "car", stem, idx + local_idx)
        if saved is not None:
            car_list.append(saved)

    detect_txt = output_root / "detect_dataset.txt"
    plate_txt = output_root / "plate_rec_dataset.txt"
    car_txt = output_root / "car_color_dataset.txt"

    detect_txt.write_text("\n".join(str(p) for p in detect_list) + ("\n" if detect_list else ""), encoding="utf-8")
    plate_txt.write_text("\n".join(str(p) for p in plate_list) + ("\n" if plate_list else ""), encoding="utf-8")
    car_txt.write_text("\n".join(str(p) for p in car_list) + ("\n" if car_list else ""), encoding="utf-8")

    print("Calibration sets generated:")
    print(f"  detect images: {len(detect_list)} -> {detect_txt}")
    print(f"  plate crops : {len(plate_list)} -> {plate_txt}")
    print(f"  car crops   : {len(car_list)} -> {car_txt}")
    print("Output directories:")
    print(f"  detect: {detect_dir}")
    print(f"  plate : {plate_dir}")
    print(f"  car   : {car_dir}")


if __name__ == "__main__":
    main()
