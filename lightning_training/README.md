# Pothole + Patching Detection — Lightning.ai Training

Train a YOLOv11 model to detect **Potholes** and **Patching** on the India RDD dataset.

---

## Folder Structure (after setup)

```
lightning_training/
├── India/                   ← paste your dataset here
│   ├── train/
│   │   ├── images/
│   │   └── annotations/xmls/
│   └── test/
│       └── images/
├── train.py
├── requirements.txt
└── README.md
```

---

## Steps on Lightning.ai

### 1. Upload this folder
Upload the entire `lightning_training/` folder to a Lightning.ai Studio.
Make sure the `India/` dataset folder is inside it.

### 2. Open a Terminal and run

```bash
pip install -r requirements.txt
python train.py
```

That's it. The script will:
- Convert annotations from Pascal VOC XML → YOLO format
- Split training data into 90% train / 10% val
- Download the YOLOv11-medium pretrained backbone automatically
- Train for up to 100 epochs (stops early if no improvement for 20 epochs)

### 3. Get your trained model

When training finishes, download:
```
runs/train/pothole_patching_v1/weights/best.pt
```
Use this `.pt` file in `hybrid_pipeline.py` with `--yolo-model best.pt`.

---

## Class Mapping

| RDD Code | Meaning              | Trained As |
|----------|----------------------|------------|
| D40      | Pothole              | Pothole    |
| D11      | Pothole (sub-type)   | Pothole    |
| D20      | Patching / repair    | Patching   |
| D44      | Rough surface patch  | Patching   |
| D00, D01, D10, D43, D50 | Cracks / other | Background (ignored) |

To change the mapping, edit `CLASS_MAP` at the top of `train.py`.

---

## GPU Recommendation

| GPU          | Batch Size |
|--------------|------------|
| A10G / A100  | 16 (default) |
| T4           | 8 (set `BATCH_SIZE = 8` in train.py) |
| CPU only     | Not recommended for this dataset size |
