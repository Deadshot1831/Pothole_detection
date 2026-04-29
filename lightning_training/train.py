#!/usr/bin/env python3
"""
Pothole + Patching Detection — YOLOv8 Training
===============================================
Runs on Lightning.ai (or any GPU machine).

Steps performed automatically:
  1. Converts Pascal VOC XML annotations → YOLO format
  2. Splits into train (90%) / val (10%)
  3. Writes dataset.yaml
  4. Trains YOLOv8

Usage:
    python train.py

Dataset must be placed at:  ./India/
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# RDD India class codes → our two target classes.
# Only these codes produce labels; all other codes (cracks, etc.) are treated
# as background (empty label file keeps the image as a negative example).
CLASS_MAP: dict[str, int] = {
    "D40": 0,   # Pothole  — deep pavement failure (most common in India RDD)
    "D11": 0,   # Pothole  — pothole sub-type
    "D20": 1,   # Patching — repair / patch work on road surface
    "D44": 1,   # Patching — rough surface patch / treatment
}
CLASS_NAMES = ["Pothole", "Patching"]

DATASET_DIR  = Path("India")          # root of the uploaded India dataset
OUTPUT_DIR   = Path("yolo_dataset")   # YOLO-format dataset will be written here

YOLO_MODEL   = "yolo11m.pt"           # pretrained backbone (auto-downloaded)
EPOCHS       = 100
IMG_SIZE     = 640
BATCH_SIZE   = 16    # lower to 8 if you see CUDA out-of-memory errors
VAL_SPLIT    = 0.10  # 10 % of training images used for validation
RANDOM_SEED  = 42

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def voc_box_to_yolo(img_w: int, img_h: int, xmin: float, ymin: float,
                    xmax: float, ymax: float) -> tuple[float, float, float, float]:
    """Convert Pascal VOC xyxy to YOLO normalised xywh."""
    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return xc, yc, bw, bh


def convert_xml(xml_path: Path, label_path: Path) -> int:
    """
    Parse one VOC XML and write the corresponding YOLO .txt label.
    Returns the number of target-class objects written.
    An empty label file is written even when 0 objects match (negative sample).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    if size_el is None:
        return 0
    img_w = int(size_el.find("width").text)
    img_h = int(size_el.find("height").text)
    if img_w == 0 or img_h == 0:
        return 0

    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            continue   # skip cracks and other non-target classes
        cls_id = CLASS_MAP[name]
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        xc, yc, bw, bh = voc_box_to_yolo(img_w, img_h, xmin, ymin, xmax, ymax)
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines))   # empty string → empty file (negative)
    return len(lines)


# ─── STEP 1: CONVERT & SPLIT ──────────────────────────────────────────────────

def build_yolo_dataset() -> Path:
    print("\n[1/3] Converting Pascal VOC → YOLO format and splitting train/val …")

    xml_dir = DATASET_DIR / "train" / "annotations" / "xmls"
    img_dir = DATASET_DIR / "train" / "images"

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    random.seed(RANDOM_SEED)
    random.shuffle(all_images)
    n_val   = max(1, int(len(all_images) * VAL_SPLIT))
    val_set = set(p.stem for p in all_images[:n_val])

    stats = {"train": [0, 0], "val": [0, 0]}   # [with_objects, background]

    for img_path in all_images:
        split = "val" if img_path.stem in val_set else "train"

        dst_img   = OUTPUT_DIR / split / "images" / img_path.name
        dst_label = OUTPUT_DIR / split / "labels" / img_path.with_suffix(".txt").name
        dst_img.parent.mkdir(parents=True, exist_ok=True)

        # Copy image
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Convert label
        xml_path = xml_dir / img_path.with_suffix(".xml").name
        if xml_path.exists():
            n = convert_xml(xml_path, dst_label)
            if n > 0:
                stats[split][0] += 1
            else:
                stats[split][1] += 1
        else:
            # No XML at all → write empty label (background)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("")
            stats[split][1] += 1

    for split, (pos, neg) in stats.items():
        print(f"    {split:5s} → {pos:4d} images with targets  |  {neg:4d} background")

    return OUTPUT_DIR


# ─── STEP 2: WRITE dataset.yaml ───────────────────────────────────────────────

def write_dataset_yaml(dataset_dir: Path) -> Path:
    cfg = {
        "path":  str(dataset_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[2/3] dataset.yaml written → {yaml_path}")
    return yaml_path


# ─── STEP 3: TRAIN ────────────────────────────────────────────────────────────

def train(yaml_path: Path) -> None:
    from ultralytics import YOLO

    print(f"\n[3/3] Starting YOLOv8 training …")
    print(f"      Base model : {YOLO_MODEL}")
    print(f"      Classes    : {CLASS_NAMES}")
    print(f"      Epochs     : {EPOCHS}  (early-stop patience=20)")
    print(f"      Image size : {IMG_SIZE}")
    print(f"      Batch size : {BATCH_SIZE}")

    model = YOLO(YOLO_MODEL)   # downloads pretrained weights automatically
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="pothole_patching_v1",
        project="runs/train",
        patience=20,
        save=True,
        exist_ok=True,
        plots=True,
        val=True,
        workers=4,
        seed=RANDOM_SEED,
        # Augmentations — sensible defaults for road damage
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,    # roads shouldn't be flipped upside-down
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    best = Path("runs/train/pothole_patching_v1/weights/best.pt")
    print(f"\n✓ Training complete.")
    print(f"  Best weights : {best}")
    print(f"  Copy this file back to use in hybrid_pipeline.py.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DATASET_DIR.exists():
        raise SystemExit(
            f"\nERROR: Dataset folder '{DATASET_DIR}' not found.\n"
            "Place the 'India' folder in the same directory as train.py and try again."
        )

    dataset_dir = build_yolo_dataset()
    yaml_path   = write_dataset_yaml(dataset_dir)
    train(yaml_path)
