"""
Test trained Pothole + Patching Model on a Video
=================================================
Runs the trained YOLOv11 model on a video and produces:
  - annotated.mp4         (full video with bounding boxes)
  - crops/                (one best crop per unique instance)
  - frames/               (annotated frame for each instance)
  - instances.csv         (tracking log with confidences + timestamps)

Deduplication is done via IoU-based tracking across frames, so each
real-world pothole/patch is counted once even if it appears for
dozens of consecutive frames.

Usage:
    python test_trained_model.py \\
        --model pothole_patching_best.pt \\
        --video "WhatsApp Video 2026-03-26 at 12.56.37.mp4"
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL = "runs_detect_runs_pavement_distress_yolo11m_weights_best.pt"
DEFAULT_VIDEO = "WhatsApp Video 2026-03-26 at 12.56.37.mp4"

# BGR palette — assigned to classes in model-order at runtime.
COLOR_PALETTE = [
    (0,   0, 255),   # red
    (0, 165, 255),   # orange
    (0, 255, 255),   # yellow
    (0, 255,   0),   # green
    (255, 128,  0),  # blue
    (255,   0, 255), # magenta
    (255, 255,   0), # cyan
    (128, 0, 128),   # purple
]
DEFAULT_COLOR = (200, 200, 200)  # gray fallback


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Path to trained YOLO .pt file. Default: {DEFAULT_MODEL}")
    p.add_argument("--video", default=DEFAULT_VIDEO,
                   help=f"Input video path. Default: {DEFAULT_VIDEO}")
    p.add_argument("--output-dir", default="runs/trained_model",
                   help="Base output directory. Default: runs/trained_model")
    p.add_argument("--conf", type=float, default=0.6,
                   help="YOLO confidence threshold. Default: 0.30")
    p.add_argument("--iou-nms", type=float, default=0.45,
                   help="YOLO NMS IoU threshold. Default: 0.45")
    p.add_argument("--track-iou", type=float, default=0.30,
                   help="IoU threshold for linking a detection to an existing track. Default: 0.30")
    p.add_argument("--track-max-gap", type=int, default=15,
                   help="Max frames a track can be unseen before it's considered lost. Default: 15")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limit frames for quick testing.")
    return p.parse_args()


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._") or "run"


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def clip_box(box, w: int, h: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


# ─── TRACKING ─────────────────────────────────────────────────────────────────

def match_track(tracks: dict, frame_idx: int, box, label: str,
                used: set, iou_thr: float, max_gap: int) -> int | None:
    """
    Find the best existing track for this detection.
    Only matches within the same class (no cross-label matching).
    Returns track id or None if no match.
    """
    best_id, best_iou = None, 0.0
    for tid, tr in tracks.items():
        if tid in used:
            continue
        if tr["label"] != label:
            continue
        if frame_idx - tr["last_frame"] > max_gap:
            continue
        i = iou(box, tr["last_box"])
        if i >= iou_thr and i > best_iou:
            best_iou, best_id = i, tid
    return best_id


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Build output dirs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = Path(args.output_dir) / f"{sanitize(video_path.stem)}_{ts}"
    crops_dir  = run_dir / "crops"
    frames_dir = run_dir / "frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] Loading model: {model_path.name}")
    model = YOLO(str(model_path))
    names = getattr(model.model, "names", {}) or {}
    print(f"[init] Model classes: {names}")

    # Assign a stable color per class based on model class order
    class_colors: dict[str, tuple[int, int, int]] = {
        names[i]: COLOR_PALETTE[i % len(COLOR_PALETTE)]
        for i in sorted(names)
    }

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = run_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {out_video}")

    print(f"[video] {video_path.name}  {W}x{H} @ {fps:.1f} fps  ({total} frames)")
    print(f"[cfg]   conf={args.conf}  track-iou={args.track_iou}  track-max-gap={args.track_max_gap}")
    print(f"[out]   {out_video}")

    tracks: dict[int, dict] = {}
    next_tid = 1
    frame_idx = 0
    raw_det_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

            # ── YOLO inference ────────────────────────────────────────────
            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou_nms,
                verbose=False,
            )[0]

            dets: list[tuple[str, float, tuple[int, int, int, int]]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box, conf_val, cls in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    cb = clip_box(tuple(box.tolist()), W, H)
                    if cb is None:
                        continue
                    label = names.get(int(cls), str(cls))
                    dets.append((label, float(conf_val), cb))

            # ── Draw annotations ──────────────────────────────────────────
            annotated = frame.copy()
            for label, conf_val, box in dets:
                color = class_colors.get(label, DEFAULT_COLOR)
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # ── IoU tracking (dedup across frames) ────────────────────────
            ts_sec = frame_idx / fps
            used: set[int] = set()
            for label, conf_val, box in sorted(dets, key=lambda d: d[1], reverse=True):
                raw_det_count += 1
                tid = match_track(tracks, frame_idx, box, label, used,
                                  args.track_iou, args.track_max_gap)

                if tid is None:
                    # New unique instance → save assets
                    tid = next_tid
                    next_tid += 1
                    x1, y1, x2, y2 = box
                    crop = frame[y1:y2, x1:x2]
                    crop_path  = crops_dir  / f"instance_{tid:04d}_{sanitize(label)}.jpg"
                    frame_path = frames_dir / f"instance_{tid:04d}_{sanitize(label)}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    cv2.imwrite(str(frame_path), annotated)
                    tracks[tid] = {
                        "label":         label,
                        "first_frame":   frame_idx,
                        "last_frame":    frame_idx,
                        "first_seconds": ts_sec,
                        "last_box":      box,
                        "best_conf":     conf_val,
                        "best_box":      box,
                        "best_frame":    frame_idx,
                        "match_count":   1,
                        "crop_path":     str(crop_path),
                        "frame_path":    str(frame_path),
                    }
                else:
                    tr = tracks[tid]
                    tr["last_frame"]  = frame_idx
                    tr["last_box"]    = box
                    tr["match_count"] += 1
                    # Replace crop/frame with higher-confidence sighting
                    if conf_val > tr["best_conf"]:
                        tr["best_conf"]  = conf_val
                        tr["best_box"]   = box
                        tr["best_frame"] = frame_idx
                        x1, y1, x2, y2 = box
                        cv2.imwrite(tr["crop_path"], frame[y1:y2, x1:x2])
                        cv2.imwrite(tr["frame_path"], annotated)
                used.add(tid)

            # ── Status overlay on top of every frame ──────────────────────
            # Total "Pavement Distress" = unique tracks of any class.
            per_class_counts: dict[str, int] = {}
            for t in tracks.values():
                per_class_counts[t["label"]] = per_class_counts.get(t["label"], 0) + 1
            total_distress = sum(per_class_counts.values())

            header = f"Frame {frame_idx+1}/{total}  |  Pavement Distress: {total_distress}"
            cv2.putText(annotated, header, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)

            # Per-class breakdown: one line per class, colored to match its boxes
            y = 65
            for cname in sorted(per_class_counts):
                line = f"{cname}: {per_class_counts[cname]}"
                cv2.putText(annotated, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            class_colors.get(cname, DEFAULT_COLOR), 2, cv2.LINE_AA)
                y += 25

            writer.write(annotated)

            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{total}  unique_tracks={len(tracks)}  dets_this_frame={len(dets)}")

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    # ── Write CSV log ─────────────────────────────────────────────────────
    csv_path = run_dir / "instances.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow([
            "instance_id", "label",
            "first_frame", "last_frame",
            "first_seconds", "last_seconds",
            "match_count", "best_confidence",
            "x1", "y1", "x2", "y2",
            "frame_path", "crop_path",
        ])
        for tid in sorted(tracks):
            tr = tracks[tid]
            x1, y1, x2, y2 = tr["best_box"]
            w_csv.writerow([
                tid, tr["label"],
                tr["first_frame"], tr["last_frame"],
                f"{tr['first_frame']/fps:.2f}", f"{tr['last_frame']/fps:.2f}",
                tr["match_count"], f"{tr['best_conf']:.4f}",
                x1, y1, x2, y2,
                tr["frame_path"], tr["crop_path"],
            ])

    # ── Write verification CSV (for manual ground-truth entry) ────────────
    # One row per unique instance. "Predicted" column is 1 if the track's
    # label maps to that class; "Actual" columns are left blank for the
    # user to fill in by reviewing frames/crops.
    # Model label → verification column
    LABEL_TO_VCOL = {
        "Pothole":   "Pothole",
        "Cracking":  "Crack",
        "Ravelling": "Ravelling",
    }
    verify_path = run_dir / "verification.csv"
    with verify_path.open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow([
            "Time Stamp",
            "Pothole Predicted", "Actual Pothole",
            "Crack Predicted",   "Actual Crack",
            "Ravelling Predicted", "Actual Ravelling",
        ])
        for tid in sorted(tracks):
            tr = tracks[tid]
            secs = tr["first_frame"] / fps
            mm, ss = divmod(int(secs), 60)
            ms = int((secs - int(secs)) * 1000)
            timestamp = f"{mm:02d}:{ss:02d}.{ms:03d}"

            vcol = LABEL_TO_VCOL.get(tr["label"])
            p_pothole   = 1 if vcol == "Pothole"   else 0
            p_crack     = 1 if vcol == "Crack"     else 0
            p_ravelling = 1 if vcol == "Ravelling" else 0

            w_csv.writerow([
                timestamp,
                p_pothole,   "",
                p_crack,     "",
                p_ravelling, "",
            ])

    per_class_counts: dict[str, int] = {}
    for t in tracks.values():
        per_class_counts[t["label"]] = per_class_counts.get(t["label"], 0) + 1

    print(f"\n[done] Processed {frame_idx} frames.")
    print(f"       Raw detections : {raw_det_count}")
    print(f"       Unique (after tracking) : {len(tracks)}")
    print(f"\n=== OUTPUT ===")
    print(f"  Annotated video : {out_video}")
    print(f"  Crops dir       : {crops_dir}")
    print(f"  Frames dir      : {frames_dir}")
    print(f"  CSV log         : {csv_path}")
    print(f"  Verification CSV: {verify_path}")
    print(f"\n=== PAVEMENT DISTRESS: {len(tracks)} ===")
    for cname in sorted(per_class_counts):
        print(f"  {cname:<25} {per_class_counts[cname]}")


if __name__ == "__main__":
    main()
