"""
Batch-run the trained pavement-distress YOLO model on every video in
`downloads/` and save the annotated outputs to a single folder.

For each input video produces:
  runs/batch_annotated/<video_stem>/annotated.mp4
  runs/batch_annotated/<video_stem>/instances.csv
  runs/batch_annotated/<video_stem>/verification.csv
  runs/batch_annotated/<video_stem>/crops/
  runs/batch_annotated/<video_stem>/frames/

Loads the YOLO model once and reuses it across videos (vs. shelling out
to test_trained_model.py per video, which would re-load weights each time).
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH    = Path("runs_detect_runs_pavement_distress_yolo11m_weights_best.pt")
INPUT_DIR     = Path("downloads")
OUTPUT_ROOT   = Path("runs/batch_annotated")
VIDEO_EXTS    = (".mp4", ".mov", ".avi", ".mkv", ".webm")

CONF          = 0.6
IOU_NMS       = 0.45
TRACK_IOU     = 0.30
TRACK_MAX_GAP = 15

COLOR_PALETTE = [
    (0,   0, 255), (0, 165, 255), (0, 255, 255),
    (0, 255,   0), (255, 128,  0), (255,   0, 255),
    (255, 255,   0), (128, 0, 128),
]
DEFAULT_COLOR = (200, 200, 200)

LABEL_TO_VCOL = {
    "Pothole":   "Pothole",
    "Cracking":  "Crack",
    "Ravelling": "Ravelling",
}


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._") or "run"


def iou(a, b) -> float:
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


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def match_track(tracks, frame_idx, box, label, used, iou_thr, max_gap):
    best_id, best_iou = None, 0.0
    for tid, tr in tracks.items():
        if tid in used or tr["label"] != label:
            continue
        if frame_idx - tr["last_frame"] > max_gap:
            continue
        i = iou(box, tr["last_box"])
        if i >= iou_thr and i > best_iou:
            best_iou, best_id = i, tid
    return best_id


def process_video(model, names, class_colors, video_path: Path, out_dir: Path) -> dict:
    crops_dir  = out_dir / "crops"
    frames_dir = out_dir / "frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"cannot open {video_path}"}

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = out_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )

    tracks: dict[int, dict] = {}
    next_tid = 1
    frame_idx = 0
    raw_det_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=CONF, iou=IOU_NMS, verbose=False)[0]

            dets = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box, conf_val, cls in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    cb = clip_box(tuple(box.tolist()), W, H)
                    if cb is None:
                        continue
                    dets.append((names.get(int(cls), str(cls)), float(conf_val), cb))

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

            used = set()
            for label, conf_val, box in sorted(dets, key=lambda d: d[1], reverse=True):
                raw_det_count += 1
                tid = match_track(tracks, frame_idx, box, label, used,
                                  TRACK_IOU, TRACK_MAX_GAP)
                if tid is None:
                    tid = next_tid
                    next_tid += 1
                    x1, y1, x2, y2 = box
                    crop_path  = crops_dir  / f"instance_{tid:04d}_{sanitize(label)}.jpg"
                    frame_path = frames_dir / f"instance_{tid:04d}_{sanitize(label)}.jpg"
                    cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
                    cv2.imwrite(str(frame_path), annotated)
                    tracks[tid] = {
                        "label": label, "first_frame": frame_idx, "last_frame": frame_idx,
                        "last_box": box, "best_conf": conf_val, "best_box": box,
                        "best_frame": frame_idx, "match_count": 1,
                        "crop_path": str(crop_path), "frame_path": str(frame_path),
                    }
                else:
                    tr = tracks[tid]
                    tr["last_frame"]  = frame_idx
                    tr["last_box"]    = box
                    tr["match_count"] += 1
                    if conf_val > tr["best_conf"]:
                        tr["best_conf"]  = conf_val
                        tr["best_box"]   = box
                        tr["best_frame"] = frame_idx
                        x1, y1, x2, y2 = box
                        cv2.imwrite(tr["crop_path"], frame[y1:y2, x1:x2])
                        cv2.imwrite(tr["frame_path"], annotated)
                used.add(tid)

            per_class_counts: dict[str, int] = {}
            for t in tracks.values():
                per_class_counts[t["label"]] = per_class_counts.get(t["label"], 0) + 1
            total_distress = sum(per_class_counts.values())
            header = f"Frame {frame_idx+1}/{total}  |  Pavement Distress: {total_distress}"
            cv2.putText(annotated, header, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
            y = 65
            for cname in sorted(per_class_counts):
                cv2.putText(annotated, f"{cname}: {per_class_counts[cname]}",
                            (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            class_colors.get(cname, DEFAULT_COLOR), 2, cv2.LINE_AA)
                y += 25

            writer.write(annotated)

            if frame_idx % 200 == 0:
                print(f"    frame {frame_idx}/{total}  unique={len(tracks)}")
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    # instances.csv
    with (out_dir / "instances.csv").open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["instance_id", "label", "first_frame", "last_frame",
                        "first_seconds", "last_seconds", "match_count",
                        "best_confidence", "x1", "y1", "x2", "y2",
                        "frame_path", "crop_path"])
        for tid in sorted(tracks):
            tr = tracks[tid]
            x1, y1, x2, y2 = tr["best_box"]
            w_csv.writerow([
                tid, tr["label"], tr["first_frame"], tr["last_frame"],
                f"{tr['first_frame']/fps:.2f}", f"{tr['last_frame']/fps:.2f}",
                tr["match_count"], f"{tr['best_conf']:.4f}",
                x1, y1, x2, y2, tr["frame_path"], tr["crop_path"],
            ])

    # verification.csv
    with (out_dir / "verification.csv").open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["Time Stamp",
                        "Pothole Predicted", "Actual Pothole",
                        "Crack Predicted", "Actual Crack",
                        "Ravelling Predicted", "Actual Ravelling"])
        for tid in sorted(tracks):
            tr = tracks[tid]
            secs = tr["first_frame"] / fps
            mm, ss = divmod(int(secs), 60)
            ms = int((secs - int(secs)) * 1000)
            timestamp = f"{mm:02d}:{ss:02d}.{ms:03d}"
            vcol = LABEL_TO_VCOL.get(tr["label"])
            w_csv.writerow([
                timestamp,
                1 if vcol == "Pothole"   else 0, "",
                1 if vcol == "Crack"     else 0, "",
                1 if vcol == "Ravelling" else 0, "",
            ])

    counts = {}
    for t in tracks.values():
        counts[t["label"]] = counts.get(t["label"], 0) + 1

    return {
        "frames": frame_idx,
        "raw_dets": raw_det_count,
        "unique_instances": len(tracks),
        "per_class": counts,
        "annotated": str(out_video),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input dir not found: {INPUT_DIR}")

    videos = sorted(p for p in INPUT_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {INPUT_DIR}")
        return

    print(f"Loading model: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))
    names = getattr(model.model, "names", {}) or {}
    class_colors = {names[i]: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in sorted(names)}
    print(f"Classes: {list(names.values())}\n")
    print(f"Found {len(videos)} videos to process.\n")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = []

    for i, vp in enumerate(videos, 1):
        out_dir = OUTPUT_ROOT / sanitize(vp.stem)
        print(f"[{i}/{len(videos)}] {vp.name}")
        print(f"   → {out_dir}")
        t0 = datetime.now()
        result = process_video(model, names, class_colors, vp, out_dir)
        elapsed = (datetime.now() - t0).total_seconds()
        result["video"] = vp.name
        result["elapsed_sec"] = round(elapsed, 1)
        summary.append(result)
        if "error" in result:
            print(f"   ERROR: {result['error']}\n")
            continue
        print(f"   frames={result['frames']}  unique={result['unique_instances']}  "
              f"per_class={result['per_class']}  ({elapsed:.0f}s)\n")

    # Master summary CSV across all videos
    with (OUTPUT_ROOT / "_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.writer(f)
        all_classes = sorted({c for s in summary for c in s.get("per_class", {})})
        w_csv.writerow(["video", "frames", "raw_dets", "unique_instances",
                        "elapsed_sec", *all_classes])
        for s in summary:
            row = [s.get("video"), s.get("frames"), s.get("raw_dets"),
                   s.get("unique_instances"), s.get("elapsed_sec")]
            for c in all_classes:
                row.append(s.get("per_class", {}).get(c, 0))
            w_csv.writerow(row)

    print(f"\n=== DONE ===")
    print(f"Per-video outputs in : {OUTPUT_ROOT.resolve()}")
    print(f"Summary CSV          : {OUTPUT_ROOT / '_summary.csv'}")


if __name__ == "__main__":
    main()
