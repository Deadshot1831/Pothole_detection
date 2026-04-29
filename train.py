"""
Train YOLOv11m on the Pavement_Distress_1 dataset.

The dataset ships with only a `train/` split, so this script auto-creates
an 85/15 train/val split on first run and writes a fresh data.yaml with
absolute paths. Designed for Lightning AI (or any CUDA GPU environment).

Usage:
    python train.py
"""

import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent / "Pavement_Distress_1.v1-video_dataset_1.yolov11"
OUTPUT_DIR  = Path(__file__).parent / "runs"
RUN_NAME    = "pavement_distress_yolo11m"

MODEL       = "yolo11m.pt"
EPOCHS      = 100
IMG_SIZE    = 640
BATCH       = -1          # AutoBatch: fill available VRAM
WORKERS     = 8
VAL_SPLIT   = 0.15
SEED        = 42


# ──────────────────────────────────────────────────────────────────────
# Step 1: create train/val split (if not already split)
# ──────────────────────────────────────────────────────────────────────
def ensure_split() -> Path:
    """Ensure train/val directories exist and return path to data.yaml."""
    train_img = DATASET_DIR / "train" / "images"
    train_lbl = DATASET_DIR / "train" / "labels"
    val_img   = DATASET_DIR / "valid" / "images"
    val_lbl   = DATASET_DIR / "valid" / "labels"

    assert train_img.exists(), f"Missing: {train_img}"
    assert train_lbl.exists(), f"Missing: {train_lbl}"

    if val_img.exists() and any(val_img.iterdir()):
        print(f"Val split already exists at {val_img} — skipping split.")
    else:
        val_img.mkdir(parents=True, exist_ok=True)
        val_lbl.mkdir(parents=True, exist_ok=True)

        images = sorted(train_img.glob("*.jpg")) + sorted(train_img.glob("*.png"))
        random.seed(SEED)
        random.shuffle(images)
        n_val = int(len(images) * VAL_SPLIT)
        val_images = images[:n_val]

        print(f"Moving {n_val} of {len(images)} images to val split ...")
        for img in val_images:
            shutil.move(str(img), val_img / img.name)
            lbl = train_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.move(str(lbl), val_lbl / lbl.name)
        print("Split done.")

    # Write a clean data.yaml with absolute paths
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    cfg["train"] = str(train_img.resolve())
    cfg["val"]   = str(val_img.resolve())
    cfg.pop("test", None)   # no test split

    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Classes ({cfg['nc']}): {cfg['names']}")
    return yaml_path


# ──────────────────────────────────────────────────────────────────────
# Step 2: verify every class has labels (catches silent class drops)
# ──────────────────────────────────────────────────────────────────────
def verify_classes(yaml_path: Path) -> None:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]

    counts = {i: 0 for i in range(len(names))}
    for split in ("train", "valid"):
        lbl_dir = DATASET_DIR / split / "labels"
        if not lbl_dir.exists():
            continue
        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1

    print("\nClass instance counts:")
    for cid, name in enumerate(names):
        print(f"  {cid}: {name:<25} {counts[cid]}")

    missing = [names[i] for i, n in counts.items() if n == 0]
    if missing:
        print(f"\nWARNING: classes with 0 instances (will not be learned): {missing}")
    else:
        print("\nAll classes have instances. Good to train.\n")


# ──────────────────────────────────────────────────────────────────────
# Step 3: train
# ──────────────────────────────────────────────────────────────────────
def train(yaml_path: Path) -> None:
    model = YOLO(MODEL)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=0,
        amp=True,
        cache="ram",
        optimizer="SGD",
        lr0=0.01,
        cos_lr=True,
        close_mosaic=10,
        patience=20,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        seed=SEED,
    )


if __name__ == "__main__":
    import torch
    assert torch.cuda.is_available(), "No CUDA GPU detected."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    yaml_path = ensure_split()
    verify_classes(yaml_path)
    train(yaml_path)

    print(f"\nDone. Best weights: {OUTPUT_DIR / RUN_NAME / 'weights' / 'best.pt'}")
