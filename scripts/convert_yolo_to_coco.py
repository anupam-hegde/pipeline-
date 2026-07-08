"""
convert_yolo_to_coco.py — Fast Parallel YOLO to COCO JSON Annotation Converter

Converts normalized YOLO text labels (<class_id> <x_center> <y_center> <width> <height>)
into standard COCO JSON format required by YOLOX.

Features:
  - Multi-threaded header reading via PIL for ultra-fast conversion of 60k+ images.
  - Automatically handles missing or empty label files (background images).
  - Production-ready error handling and progress logging.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional

try:
    from PIL import Image
except ImportError:
    raise RuntimeError("PIL (Pillow) is required for image header reading. Please install pillow.")

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
DEFAULT_CLASSES = ["fire", "weapon"]


def get_image_info(img_path: Path, img_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Reads image dimensions using PIL header-only parsing for speed."""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
        
        info = {
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        }
        return info, img_path
    except Exception as e:
        print(f"[!] Error reading image header {img_path}: {e}")
        return None, None


def convert_split_to_coco(
    split_dir: Path,
    classes: List[str],
    out_json_path: Path,
    num_workers: int = 8
) -> Dict[str, Any]:
    """
    Converts a single dataset split (containing images/ and labels/ subdirectories)
    into a COCO JSON structure and saves it to out_json_path.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        print(f"[!] Warning: Labels directory not found at {labels_dir}. Creating background-only COCO dataset.")

    print(f"\n[*] Scanning image files in: {images_dir}")
    img_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    print(f"[*] Found {len(img_files)} images. Reading dimensions using {num_workers} threads...")

    images_list: List[Dict[str, Any]] = []
    annotations_list: List[Dict[str, Any]] = []
    categories_list: List[Dict[str, Any]] = [
        {"id": idx, "name": name, "supercategory": "object"} for idx, name in enumerate(classes)
    ]

    # Parallel image header reading
    img_id_map: Dict[Path, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(get_image_info, img_path, idx + 1): img_path
            for idx, img_path in enumerate(img_files)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scanning {split_dir.name} images"):
            info, path = future.result()
            if info is not None and path is not None:
                images_list.append(info)
                img_id_map[path] = info

    # Sort images by ID for consistency
    images_list.sort(key=lambda x: x["id"])

    print(f"[*] Processing YOLO annotations...")
    ann_id = 1
    missing_labels = 0
    empty_labels = 0

    for img_info in tqdm(images_list, desc=f"Converting {split_dir.name} labels"):
        img_id = img_info["id"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]
        
        stem = Path(file_name).stem
        lbl_path = labels_dir / f"{stem}.txt"

        if not lbl_path.exists():
            missing_labels += 1
            continue

        try:
            with open(lbl_path, 'r', errors='replace') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"[!] Error reading label file {lbl_path}: {e}")
            continue

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])
            except ValueError:
                continue

            if cls_id < 0 or cls_id >= len(classes):
                continue

            # Convert YOLO (normalized center) to COCO (pixel top-left)
            w_px = max(0.0, min(float(img_w), w_norm * img_w))
            h_px = max(0.0, min(float(img_h), h_norm * img_h))
            x_min = max(0.0, min(float(img_w - w_px), (x_center * img_w) - (w_px / 2.0)))
            y_min = max(0.0, min(float(img_h - h_px), (y_center * img_h) - (h_px / 2.0)))
            area = w_px * h_px

            if w_px <= 1.0 or h_px <= 1.0:
                continue  # Skip degenerate bounding boxes

            ann_entry = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
                "ignore": 0,
                "segmentation": []
            }
            annotations_list.append(ann_entry)
            ann_id += 1

    coco_dataset = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": categories_list
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving COCO JSON to: {out_json_path}")
    with open(out_json_path, 'w') as f:
        json.dump(coco_dataset, f)

    print(f"=== Conversion Summary ({split_dir.name}) ===")
    print(f"  Total Images:      {len(images_list)}")
    print(f"  Total Annotations: {len(annotations_list)}")
    print(f"  Background Images: {missing_labels + empty_labels}")
    print(f"  Saved JSON:        {out_json_path.as_posix()}")
    print("==============================================")
    return coco_dataset


def convert_dataset_root(dataset_root: Path, classes: List[str]) -> None:
    """Converts standard train/val splits under dataset_root to COCO JSONs."""
    annotations_dir = dataset_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if split_dir.exists() and (split_dir / "images").exists():
            out_json = annotations_dir / f"{split}.json"
            convert_split_to_coco(split_dir, classes, out_json)


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO text annotations to COCO JSON format")
    parser.add_argument("--dataset-dir", type=str, default="../merged_dataset", help="Path to root dataset directory")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="List of class names in order")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for reading image headers")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir).resolve()
    if not dataset_root.exists():
        print(f"[!] Dataset directory not found: {dataset_root}")
        sys.exit(1)

    print("=" * 60)
    print(f"YOLO to COCO Converter — Target: {dataset_root}")
    print(f"Classes ({len(args.classes)}): {args.classes}")
    print("=" * 60)

    convert_dataset_root(dataset_root, args.classes)
    print("\n[*] All splits successfully converted to COCO JSON format!")


if __name__ == "__main__":
    main()
