"""
Hybrid Pothole Detection Pipeline
==================================
YOLO (best (3).pt) + Depth Anything V1-Small (LiheYoung/depth-anything-small-hf)

Speed improvements over detect_potholes.py:
  - Depth model: V1-Small @ 308 px (vs V2-Small @ 518 px) → ~65% fewer pixels
  - fp16 inference on CUDA
  - --depth-stride: run depth only every N frames, cache between
  - --depth-input-size: pre-scale frame before depth processor
  - Depth only computed when YOLO proposes a candidate (hybrid fusion)
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_YOLO_MODEL = "best (3).pt"
DEFAULT_DEPTH_MODEL = "LiheYoung/depth-anything-small-hf"  # V1-Small, 308 px
DEFAULT_VIDEO = "WhatsApp Video 2026-03-26 at 12.56.37.mp4"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Pothole Detection: YOLO proposals validated by "
            "Depth-Anything-Small depth scoring. Outputs an annotated video."
        )
    )
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL,
                        help=f"Path to YOLO .pt checkpoint. Default: {DEFAULT_YOLO_MODEL}")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL,
                        help=f"HuggingFace depth model id. Default: {DEFAULT_DEPTH_MODEL}")
    parser.add_argument("--video", default=DEFAULT_VIDEO,
                        help=f"Input video path. Default: {DEFAULT_VIDEO}")
    parser.add_argument("--output-dir", default="runs/pothole_detection",
                        help="Base output directory.")
    parser.add_argument("--conf", type=float, default=0.80,
                        help="YOLO acceptance threshold (final). Default: 0.80")
    parser.add_argument("--depth-conf", type=float, default=0.50,
                        help="Depth score threshold for hybrid validation. Default: 0.50")
    parser.add_argument("--hybrid-yolo-conf", type=float, default=0.35,
                        help="Lower YOLO proposal threshold before depth check. Default: 0.35")
    parser.add_argument("--depth-stride", type=int, default=2,
                        help="Run depth model every N frames; reuse cached map between. Default: 2")
    parser.add_argument("--depth-input-size", type=int, default=640,
                        help="Max dimension when pre-scaling frame for depth. Default: 640")
    parser.add_argument("--track-iou-threshold", type=float, default=0.30,
                        help="IoU threshold for track matching. Default: 0.30")
    parser.add_argument("--track-max-gap-frames", type=int, default=10,
                        help="Max frame gap for track matching. Default: 10")
    parser.add_argument("--clip-pre-seconds", type=float, default=1.0,
                        help="Seconds of context before first detection. Default: 1.0")
    parser.add_argument("--clip-post-seconds", type=float, default=1.0,
                        help="Seconds of context after last detection. Default: 1.0")
    parser.add_argument("--show-depth", action="store_true",
                        help="Overlay depth heatmap on annotated video.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames for testing.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "run"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_output_paths(base: Path, video_path: Path) -> dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{sanitize_name(video_path.stem)}_{ts}"
    frames_dir = run_dir / "frames"
    crops_dir = run_dir / "crops"
    clips_dir = run_dir / "clips"
    for d in (frames_dir, crops_dir, clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "frames_dir": frames_dir,
        "crops_dir": crops_dir,
        "clips_dir": clips_dir,
        "video_path": run_dir / "annotated.mp4",
        "csv_path": run_dir / "instances.csv",
        "clips_csv_path": run_dir / "clips.csv",
    }


def calculate_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def clip_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w - 1)), max(0, min(y2, h - 1))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def apply_nms(
    dets: list[tuple[str, float, tuple[int, int, int, int]]],
    iou_thr: float = 0.35,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for det in sorted(dets, key=lambda d: d[1], reverse=True):
        if any(calculate_iou(det[2], k[2]) >= iou_thr for k in kept):
            continue
        kept.append(det)
    return kept


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo(path: str) -> dict:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"YOLO model not found: {p}")
    model = YOLO(str(p))
    return {"model": model, "names": getattr(model.model, "names", {})}


def load_depth(model_id: str) -> dict:
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as e:
        raise RuntimeError("Install `transformers` to use the depth model.") from e

    device = choose_device()
    use_fp16 = device == "cuda"
    dtype = torch.float16 if use_fp16 else torch.float32

    print(f"[depth] Loading {model_id} → device={device}, dtype={'fp16' if use_fp16 else 'fp32'}")
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device=device, dtype=dtype).eval()

    return {"model": model, "processor": processor, "device": device, "dtype": dtype}


# ---------------------------------------------------------------------------
# Depth inference
# ---------------------------------------------------------------------------

def _prescale(frame_rgb: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if max(h, w) <= max_dim:
        return frame_rgb
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def infer_depth_score_map(
    frame: np.ndarray,
    depth_backend: dict,
    depth_input_size: int,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_small = _prescale(rgb, depth_input_size)

    processor = depth_backend["processor"]
    model = depth_backend["model"]
    device = depth_backend["device"]
    dtype = depth_backend["dtype"]

    inputs = processor(images=rgb_small, return_tensors="pt")
    inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        predicted_depth = out.predicted_depth  # (1, H', W') or (H', W')

    if predicted_depth.dim() == 2:
        predicted_depth = predicted_depth.unsqueeze(0).unsqueeze(0)
    elif predicted_depth.dim() == 3:
        predicted_depth = predicted_depth.unsqueeze(1)

    depth_up = torch.nn.functional.interpolate(
        predicted_depth.float(),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float32)

    return _build_score_map(frame, depth_up)


def _build_score_map(frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape
    depth_norm = cv2.normalize(depth_map, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

    # Depression: regions lower than surroundings
    blur_large = cv2.GaussianBlur(depth_norm, (0, 0), max(w / 20, 5), max(h / 20, 5))
    blur_local = cv2.GaussianBlur(depth_norm, (0, 0), max(w / 80, 3), max(h / 80, 3))
    depression = np.maximum(blur_large - blur_local, 0.0)

    # Darkness (shadows)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    darkness = 1.0 - cv2.GaussianBlur(gray, (0, 0), 3.0, 3.0)

    # Roughness
    edges = cv2.Laplacian(depth_norm, cv2.CV_32F, ksize=3)
    roughness = cv2.GaussianBlur(np.abs(edges), (0, 0), 2.0, 2.0)

    score = 0.55 * depression + 0.30 * darkness + 0.15 * roughness

    # ROI mask: ignore sky (top 35%) and extreme sides (5%)
    roi = np.zeros_like(score)
    roi[int(h * 0.35):h, int(w * 0.05):int(w * 0.95)] = 1.0
    score *= roi

    mx = score.max()
    if mx > 0:
        score /= mx
    return score


def extract_depth_detections(
    score_map: np.ndarray,
    frame_w: int,
    frame_h: int,
    threshold: float,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    binary = (score_map >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(150, int(frame_w * frame_h * 0.0015))
    max_area = int(frame_w * frame_h * 0.20)
    dets: list[tuple[str, float, tuple[int, int, int, int]]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 20 or bh < 12:
            continue
        ar = bw / max(bh, 1)
        if ar < 0.35 or ar > 4.5:
            continue
        conf = float(score_map[y:y + bh, x:x + bw].max())
        if conf < threshold:
            continue
        box = clip_box((x, y, x + bw, y + bh), frame_w, frame_h)
        if box is None:
            continue
        dets.append(("Potholes", conf, box))

    return apply_nms(dets)


# ---------------------------------------------------------------------------
# Hybrid detection
# ---------------------------------------------------------------------------

def run_yolo(frame: np.ndarray, yolo: dict, conf: float, run_dir: Path) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    kwargs = {
        "conf": conf, "verbose": False, "save": False,
        "project": str(run_dir), "name": "yolo_cache", "exist_ok": True,
    }
    result = yolo["model"].predict(frame, **kwargs)[0]
    if result.boxes is None:
        return []
    h, w = frame.shape[:2]
    dets = []
    for box, conf_val, cls in zip(
        result.boxes.xyxy.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
    ):
        b = clip_box(tuple(box.tolist()), w, h)
        if b is None:
            continue
        label = yolo["names"].get(int(cls), str(cls))
        dets.append((label, float(conf_val), b))
    return dets


def hybrid_detect(
    frame: np.ndarray,
    yolo: dict,
    depth_backend: dict,
    cached_score_map: np.ndarray | None,
    run_dir: Path,
    conf: float,
    hybrid_yolo_conf: float,
    depth_conf: float,
    depth_input_size: int,
    compute_depth: bool,
) -> tuple[list[tuple[str, float, tuple[int, int, int, int]]], np.ndarray | None]:
    """
    Returns (detections, updated_score_map).
    Depth is only computed if compute_depth=True; otherwise cached_score_map is reused.
    """
    yolo_dets = run_yolo(frame, yolo, hybrid_yolo_conf, run_dir)
    if not yolo_dets:
        return [], cached_score_map

    # Compute or reuse depth score map
    if compute_depth or cached_score_map is None:
        score_map = infer_depth_score_map(frame, depth_backend, depth_input_size)
    else:
        score_map = cached_score_map

    h, w = frame.shape[:2]
    depth_dets = extract_depth_detections(score_map, w, h, depth_conf)

    fused: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for label, yolo_conf, box in yolo_dets:
        x1, y1, x2, y2 = box
        region = score_map[y1:y2, x1:x2]
        depth_peak = float(region.max()) if region.size > 0 else 0.0
        depth_mean = float(region.mean()) if region.size > 0 else 0.0

        best_iou = max((calculate_iou(box, d[2]) for d in depth_dets), default=0.0)

        accepted = yolo_conf >= conf or (
            yolo_conf >= hybrid_yolo_conf
            and depth_peak >= depth_conf
            and best_iou >= 0.10
        )
        if not accepted:
            continue

        hybrid_conf = min(1.0, max(yolo_conf, depth_peak) + 0.10 * best_iou + 0.05 * depth_mean)
        fused.append((label, hybrid_conf, box))

    return apply_nms(fused), score_map


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def draw_detection(img: np.ndarray, box: tuple[int, int, int, int], label: str, conf: float) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


def overlay_depth_heatmap(img: np.ndarray, score_map: np.ndarray, alpha: float = 0.35) -> None:
    heat_u8 = (score_map * 255).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
    cv2.addWeighted(heat_color, alpha, img, 1 - alpha, 0, img)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def match_track(
    tracks: dict[int, dict],
    frame_idx: int,
    box: tuple[int, int, int, int],
    used: set[int],
    iou_thr: float,
    max_gap: int,
) -> int | None:
    best_id, best_iou = None, 0.0
    for tid, tr in tracks.items():
        if tid in used:
            continue
        if frame_idx - int(tr["last_frame"]) > max_gap:
            continue
        iou = calculate_iou(box, tuple(tr["last_box"]))
        if iou >= iou_thr and iou > best_iou:
            best_iou, best_id = iou, tid
    return best_id


def save_assets(track: dict, frames_dir: Path, crops_dir: Path, ann_frame, raw_frame, box) -> None:
    iid = int(track["instance_id"])
    fp = frames_dir / f"instance_{iid:04d}.jpg"
    cp = crops_dir / f"instance_{iid:04d}.jpg"
    x1, y1, x2, y2 = box
    cv2.imwrite(str(fp), ann_frame)
    cv2.imwrite(str(cp), raw_frame[y1:y2, x1:x2])
    track["frame_path"] = str(fp)
    track["crop_path"] = str(cp)


def create_track(tid, label, fidx, ts, box, conf, frames_dir, crops_dir, ann, raw) -> dict:
    tr = {
        "instance_id": tid, "label": label,
        "first_frame": fidx, "last_frame": fidx,
        "last_box": box, "match_count": 1,
        "best_confidence": conf, "best_box": box,
        "best_frame_index": fidx, "best_timestamp_seconds": ts,
        "frame_path": "", "crop_path": "", "clip_path": "",
    }
    save_assets(tr, frames_dir, crops_dir, ann, raw, box)
    return tr


# ---------------------------------------------------------------------------
# Clip saving
# ---------------------------------------------------------------------------

def finalize_windows(tracks: dict, fps: float, total: int, pre_s: float, post_s: float) -> list[dict]:
    pre_f = max(0, int(round(pre_s * fps)))
    post_f = max(0, int(round(post_s * fps)))
    out = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        cs = max(0, int(tr["first_frame"]) - pre_f)
        ce = min(total - 1, int(tr["last_frame"]) + post_f)
        tr["clip_start_frame"] = cs
        tr["clip_end_frame"] = ce
        tr["clip_start_seconds"] = cs / fps
        tr["clip_end_seconds"] = ce / fps
        out.append(tr)
    return out


def save_clips(src: Path, clips_dir: Path, tracks: list[dict], fps: float, size: tuple[int, int]) -> None:
    if not tracks:
        return
    cap = cv2.VideoCapture(str(src))
    try:
        for tr in tracks:
            name = (
                f"instance_{int(tr['instance_id']):04d}_"
                f"t_{float(tr['clip_start_seconds']):08.2f}s_"
                f"to_{float(tr['clip_end_seconds']):08.2f}s.mp4"
            )
            out = cv2.VideoWriter(str(clips_dir / name),
                                  cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tr["clip_start_frame"]))
            count = 0
            for _ in range(int(tr["clip_start_frame"]), int(tr["clip_end_frame"]) + 1):
                ok, f = cap.read()
                if not ok:
                    break
                out.write(f)
                count += 1
            out.release()
            tr["clip_path"] = str(clips_dir / name)
            tr["clip_frame_count"] = count
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def write_instance_log(path: Path, tracks: list[dict], fps: float) -> None:
    fields = ["instance_id","label","first_frame","last_frame","first_seconds","last_seconds",
              "match_count","best_frame_index","best_timestamp_seconds","best_confidence",
              "x1","y1","x2","y2","frame_path","crop_path","clip_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in tracks:
            x1, y1, x2, y2 = tuple(tr["best_box"])
            w.writerow({
                "instance_id": int(tr["instance_id"]),
                "label": tr["label"],
                "first_frame": int(tr["first_frame"]),
                "last_frame": int(tr["last_frame"]),
                "first_seconds": f"{int(tr['first_frame'])/fps:.2f}",
                "last_seconds": f"{int(tr['last_frame'])/fps:.2f}",
                "match_count": int(tr["match_count"]),
                "best_frame_index": int(tr["best_frame_index"]),
                "best_timestamp_seconds": f"{float(tr['best_timestamp_seconds']):.2f}",
                "best_confidence": f"{float(tr['best_confidence']):.4f}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "frame_path": tr["frame_path"],
                "crop_path": tr["crop_path"],
                "clip_path": tr.get("clip_path", ""),
            })


def write_clip_log(path: Path, tracks: list[dict]) -> None:
    fields = ["instance_id","start_frame","end_frame","start_seconds","end_seconds",
              "frame_count","match_count","clip_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in tracks:
            w.writerow({
                "instance_id": int(tr["instance_id"]),
                "start_frame": int(tr["clip_start_frame"]),
                "end_frame": int(tr["clip_end_frame"]),
                "start_seconds": f"{float(tr['clip_start_seconds']):.2f}",
                "end_seconds": f"{float(tr['clip_end_seconds']):.2f}",
                "frame_count": int(tr.get("clip_frame_count", 0)),
                "match_count": int(tr["match_count"]),
                "clip_path": tr.get("clip_path", ""),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    paths = build_output_paths(Path(args.output_dir).resolve(), video_path)

    print(f"[init] Loading YOLO: {args.yolo_model}")
    yolo = load_yolo(args.yolo_model)

    print(f"[init] Loading Depth model: {args.depth_model}")
    depth_backend = load_depth(args.depth_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(paths["video_path"]),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot open output video writer.")

    print(f"[video] {video_path.name}  {width}x{height} @ {fps:.1f} fps  ({total_frames} frames)")
    print(f"[cfg] depth-stride={args.depth_stride}  depth-input-size={args.depth_input_size}  "
          f"conf={args.conf}  depth-conf={args.depth_conf}")
    print(f"[out] {paths['video_path']}")

    tracks: dict[int, dict] = {}
    next_tid = 1
    cached_score_map: np.ndarray | None = None

    frames_processed = 0
    frames_with_dets = 0
    raw_det_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frames_processed >= args.max_frames:
                break

            compute_depth = (frames_processed % args.depth_stride == 0)

            dets, cached_score_map = hybrid_detect(
                frame=frame,
                yolo=yolo,
                depth_backend=depth_backend,
                cached_score_map=cached_score_map,
                run_dir=paths["run_dir"],
                conf=args.conf,
                hybrid_yolo_conf=args.hybrid_yolo_conf,
                depth_conf=args.depth_conf,
                depth_input_size=args.depth_input_size,
                compute_depth=compute_depth,
            )

            annotated = frame.copy()

            if args.show_depth and cached_score_map is not None:
                overlay_depth_heatmap(annotated, cached_score_map)

            for label, conf_val, box in dets:
                draw_detection(annotated, box, label, conf_val)

            if dets:
                frames_with_dets += 1

            ts = frames_processed / fps
            used: set[int] = set()
            for label, conf_val, box in sorted(dets, key=lambda d: d[1], reverse=True):
                raw_det_count += 1
                tid = match_track(tracks, frames_processed, box, used,
                                  args.track_iou_threshold, args.track_max_gap_frames)
                if tid is None:
                    tr = create_track(next_tid, label, frames_processed, ts, box, conf_val,
                                      paths["frames_dir"], paths["crops_dir"], annotated, frame)
                    tracks[next_tid] = tr
                    used.add(next_tid)
                    next_tid += 1
                else:
                    tr = tracks[tid]
                    tr["last_frame"] = frames_processed
                    tr["last_box"] = box
                    tr["match_count"] = int(tr["match_count"]) + 1
                    used.add(tid)
                    if conf_val > float(tr["best_confidence"]):
                        tr["best_confidence"] = conf_val
                        tr["best_box"] = box
                        tr["best_frame_index"] = frames_processed
                        tr["best_timestamp_seconds"] = ts
                        save_assets(tr, paths["frames_dir"], paths["crops_dir"], annotated, frame, box)

            depth_tag = "D" if compute_depth else "d"
            status = (
                f"[{depth_tag}] Frame {frames_processed+1}/{total_frames or '?'} | "
                f"Potholes: {len(tracks)}"
            )
            cv2.putText(annotated, status, (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
            writer.write(annotated)

            if frames_processed % 100 == 0:
                print(f"  frame {frames_processed}/{total_frames or '?'}  "
                      f"tracks={len(tracks)}  dets_this_frame={len(dets)}")

            frames_processed += 1

    finally:
        cap.release()
        writer.release()

    print(f"\n[done] Processed {frames_processed} frames, {frames_with_dets} with detections.")
    print(f"       Raw detections: {raw_det_count}  |  Unique instances: {len(tracks)}")

    tracked = finalize_windows(tracks, fps, frames_processed,
                               args.clip_pre_seconds, args.clip_post_seconds)
    print("[clips] Saving instance clips …")
    save_clips(paths["video_path"], paths["clips_dir"], tracked, fps, (width, height))
    write_instance_log(paths["csv_path"], tracked, fps)
    write_clip_log(paths["clips_csv_path"], tracked)

    print(f"\n=== OUTPUT ===")
    print(f"  Annotated video : {paths['video_path']}")
    print(f"  Frames dir      : {paths['frames_dir']}")
    print(f"  Crops dir       : {paths['crops_dir']}")
    print(f"  Clips dir       : {paths['clips_dir']}")
    print(f"  Instance log    : {paths['csv_path']}")
    print(f"  Clip log        : {paths['clips_csv_path']}")
    print(f"  Unique potholes : {len(tracked)}")


if __name__ == "__main__":
    main()
