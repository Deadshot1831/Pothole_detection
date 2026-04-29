"""
Depth-Only Pothole Detection Pipeline
=======================================
Uses ONLY Depth-Anything (LiheYoung/depth-anything-small-hf) for detection.
Output video is rendered as a depth heatmap (INFERNO colormap) with bounding
boxes drawn on top — exactly like the Depth Anything visualisation style.

Saves:
  - annotated_depth.mp4   : heatmap video with detection boxes
  - frames/               : best heatmap frame per pothole instance
  - crops/                : cropped heatmap region per instance
  - clips/                : short video clips per instance
  - instances.csv
  - clips.csv
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


DEFAULT_DEPTH_MODEL = "LiheYoung/depth-anything-small-hf"
DEFAULT_VIDEO       = "WhatsApp Video 2026-03-26 at 12.56.37.mp4"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Depth-only pothole detection with heatmap output video."
    )
    p.add_argument("--depth-model",    default=DEFAULT_DEPTH_MODEL)
    p.add_argument("--video",          default=DEFAULT_VIDEO)
    p.add_argument("--output-dir",     default="runs/depth_only")
    p.add_argument("--depth-conf",     type=float, default=0.60,
                   help="Score-map threshold for accepting a detection. Default: 0.60")
    p.add_argument("--roi-top",        type=float, default=0.55,
                   help="Fraction from top where road ROI starts (white line). Default: 0.55")
    p.add_argument("--depth-stride",   type=int,   default=2,
                   help="Run depth every N frames. Default: 2")
    p.add_argument("--depth-input-size", type=int, default=640,
                   help="Pre-scale max dimension before depth processor. Default: 640")
    p.add_argument("--track-iou",      type=float, default=0.25,
                   help="IoU to match detection to existing track. Default: 0.25")
    p.add_argument("--track-gap",      type=int,   default=25,
                   help="Max frame gap for continuing a track. Default: 25")
    p.add_argument("--min-match-count", type=int,  default=3,
                   help="Discard tracks seen in fewer than N frames (noise filter). Default: 3")
    p.add_argument("--merge-iou",      type=float, default=0.40,
                   help="Post-tracking: merge instances whose best boxes overlap above this. Default: 0.40")
    p.add_argument("--clip-pre",       type=float, default=1.0)
    p.add_argument("--clip-post",      type=float, default=1.0)
    p.add_argument("--side-by-side",   action="store_true",
                   help="Output original frame beside heatmap (doubled width).")
    p.add_argument("--max-frames",     type=int,   default=None)
    return p.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def sanitize(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", v).strip("._") or "run"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_paths(base: Path, video: Path) -> dict[str, Path]:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{sanitize(video.stem)}_{ts}"
    for sub in ("frames", "crops", "clips"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return {
        "run_dir":       run_dir,
        "frames_dir":    run_dir / "frames",
        "crops_dir":     run_dir / "crops",
        "clips_dir":     run_dir / "clips",
        "video_path":    run_dir / "annotated_depth.mp4",
        "csv_path":      run_dir / "instances.csv",
        "clips_csv":     run_dir / "clips.csv",
    }


def iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih   = max(0,ix2-ix1), max(0,iy2-iy1)
    inter   = iw*ih
    if inter <= 0: return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0.0


def clip_box(box, w, h):
    x1,y1,x2,y2 = box
    x1,y1 = max(0,min(x1,w-1)), max(0,min(y1,h-1))
    x2,y2 = max(0,min(x2,w-1)), max(0,min(y2,h-1))
    return (x1,y1,x2,y2) if x2>x1 and y2>y1 else None


def nms(dets, thr=0.35):
    kept = []
    for d in sorted(dets, key=lambda x: x[1], reverse=True):
        if any(iou(d[2], k[2]) >= thr for k in kept):
            continue
        kept.append(d)
    return kept


# ── Depth model ───────────────────────────────────────────────────────────────

def load_depth(model_id: str) -> dict:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device   = choose_device()
    dtype    = torch.float16 if device == "cuda" else torch.float32
    print(f"[depth] Loading {model_id}  device={device}  dtype={'fp16' if dtype==torch.float16 else 'fp32'}")
    proc  = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device=device, dtype=dtype).eval()
    return {"model": model, "proc": proc, "device": device, "dtype": dtype}


def infer_raw_depth(frame_bgr: np.ndarray, backend: dict, max_dim: int) -> np.ndarray:
    """Return raw depth map upsampled to frame_bgr's original size."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Pre-scale for speed
    if max(h, w) > max_dim:
        scale  = max_dim / max(h, w)
        rgb_in = cv2.resize(rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_in = rgb

    proc   = backend["proc"]
    model  = backend["model"]
    device = backend["device"]
    dtype  = backend["dtype"]

    inputs = proc(images=rgb_in, return_tensors="pt")
    inputs = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

    with torch.no_grad():
        out   = model(**inputs)
        depth = out.predicted_depth   # (1,H',W') or (H',W')

    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(1)

    depth_up = torch.nn.functional.interpolate(
        depth.float(), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy().astype(np.float32)

    return depth_up


# ── Score map & detections ────────────────────────────────────────────────────

def build_score_map(frame_bgr: np.ndarray, depth_raw: np.ndarray,
                    roi_top: float = 0.40) -> np.ndarray:
    h, w    = depth_raw.shape
    d_norm  = cv2.normalize(depth_raw, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

    # Depression: regions locally deeper than surroundings
    blur_l  = cv2.GaussianBlur(d_norm, (0,0), max(w/20, 5),  max(h/20, 5))
    blur_s  = cv2.GaussianBlur(d_norm, (0,0), max(w/80, 3),  max(h/80, 3))
    depr    = np.maximum(blur_l - blur_s, 0.0)

    # Darkness (shadow cue)
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark    = 1.0 - cv2.GaussianBlur(gray, (0,0), 3.0, 3.0)

    # Roughness (depth edge energy)
    edges   = cv2.Laplacian(d_norm, cv2.CV_32F, ksize=3)
    rough   = cv2.GaussianBlur(np.abs(edges), (0,0), 2.0, 2.0)

    score   = 0.55*depr + 0.30*dark + 0.15*rough

    # ROI: only below the roi_top boundary, ignore extreme side edges (5%)
    roi     = np.zeros_like(score)
    roi[int(h*roi_top):h, int(w*0.05):int(w*0.95)] = 1.0
    score  *= roi

    mx = score.max()
    if mx > 0:
        score /= mx
    return score


def score_to_detections(score_map: np.ndarray, w: int, h: int,
                         threshold: float, roi_top: float = 0.40) -> list:
    binary = (score_map >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area    = max(150, int(w * h * 0.0015))
    max_area    = int(w * h * 0.20)
    dets        = []

    roi_y = int(h * roi_top)   # hard pixel boundary — nothing above this counts

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Reject boxes whose centre is above the ROI line
        if (y + bh // 2) < roi_y:
            continue
        if bw < 20 or bh < 12:
            continue
        ar = bw / max(bh, 1)
        if ar < 0.35 or ar > 4.5:
            continue
        conf = float(score_map[y:y+bh, x:x+bw].max())
        if conf < threshold:
            continue
        box = clip_box((x, y, x+bw, y+bh), w, h)
        if box is None:
            continue
        dets.append(("Pothole", conf, box))

    return nms(dets)


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def depth_to_heatmap(depth_raw: np.ndarray) -> np.ndarray:
    """Convert raw depth map → INFERNO BGR image (matches Depth Anything style)."""
    u8 = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def draw_box_on_heatmap(img: np.ndarray, box, label: str, conf: float) -> None:
    x1, y1, x2, y2 = box
    # White box with black shadow for visibility on heatmap
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0),   3)
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, max(22, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0),       3, cv2.LINE_AA)
    cv2.putText(img, text, (x1, max(22, y1-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255),  1, cv2.LINE_AA)


def draw_roi_line(img: np.ndarray, roi_top: float) -> None:
    """Draw the white ROI boundary line — only detections below this line count."""
    h, w = img.shape[:2]
    y = int(h * roi_top)
    cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)


def compose_frame(original: np.ndarray, heatmap: np.ndarray,
                  side_by_side: bool) -> np.ndarray:
    if side_by_side:
        return np.hstack([original, heatmap])
    return heatmap


# ── Tracking ──────────────────────────────────────────────────────────────────

def match_track(tracks, fidx, box, used, iou_thr, gap):
    best_id, best_iou = None, 0.0
    for tid, tr in tracks.items():
        if tid in used: continue
        if fidx - int(tr["last_frame"]) > gap: continue
        v = iou(box, tuple(tr["last_box"]))
        if v >= iou_thr and v > best_iou:
            best_iou, best_id = v, tid
    return best_id


def save_assets(track, frames_dir, crops_dir, heat_ann, heat_raw, box):
    iid  = int(track["instance_id"])
    fp   = frames_dir / f"instance_{iid:04d}.jpg"
    cp   = crops_dir  / f"instance_{iid:04d}.jpg"
    x1,y1,x2,y2 = box
    cv2.imwrite(str(fp), heat_ann)
    cv2.imwrite(str(cp), heat_raw[y1:y2, x1:x2])
    track["frame_path"] = str(fp)
    track["crop_path"]  = str(cp)


def new_track(tid, label, fidx, ts, box, conf, frames_dir, crops_dir, heat_ann, heat_raw):
    tr = {
        "instance_id": tid, "label": label,
        "first_frame": fidx, "last_frame": fidx,
        "last_box": box, "match_count": 1,
        "best_confidence": conf, "best_box": box,
        "best_frame_index": fidx, "best_timestamp_seconds": ts,
        "frame_path": "", "crop_path": "", "clip_path": "",
    }
    save_assets(tr, frames_dir, crops_dir, heat_ann, heat_raw, box)
    return tr


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_tracks(tracks: dict, min_match: int, merge_iou_thr: float) -> dict:
    """
    Two-pass deduplication:
      1. Drop tracks seen in fewer than `min_match` frames  (noise/flicker)
      2. Merge tracks whose best_box overlaps > merge_iou_thr (same physical pothole
         seen across non-consecutive segments)
    Returns a new dict with re-keyed, clean tracks.
    """
    # Pass 1 — minimum match count filter
    stable = {tid: tr for tid, tr in tracks.items()
              if int(tr["match_count"]) >= min_match}

    # Pass 2 — spatial merge: sort by confidence desc, suppress overlapping duplicates
    sorted_tracks = sorted(stable.values(),
                           key=lambda t: float(t["best_confidence"]), reverse=True)
    kept = []
    for tr in sorted_tracks:
        box = tuple(tr["best_box"])
        if any(iou(box, tuple(k["best_box"])) >= merge_iou_thr for k in kept):
            continue   # duplicate of a higher-confidence track
        kept.append(tr)

    # Re-key from 1
    return {i + 1: {**tr, "instance_id": i + 1} for i, tr in enumerate(kept)}


# ── Clip saving ───────────────────────────────────────────────────────────────

def finalize(tracks, fps, total, pre_s, post_s):
    pre_f  = max(0, int(round(pre_s * fps)))
    post_f = max(0, int(round(post_s * fps)))
    out    = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        cs = max(0, int(tr["first_frame"]) - pre_f)
        ce = min(total - 1, int(tr["last_frame"]) + post_f)
        tr.update(clip_start_frame=cs, clip_end_frame=ce,
                  clip_start_seconds=cs/fps, clip_end_seconds=ce/fps)
        out.append(tr)
    return out


def save_clips(src, clips_dir, tracks, fps, size):
    if not tracks: return
    cap = cv2.VideoCapture(str(src))
    try:
        for tr in tracks:
            name = (f"instance_{int(tr['instance_id']):04d}_"
                    f"t_{float(tr['clip_start_seconds']):08.2f}s_"
                    f"to_{float(tr['clip_end_seconds']):08.2f}s.mp4")
            writer = cv2.VideoWriter(str(clips_dir / name),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tr["clip_start_frame"]))
            count = 0
            for _ in range(int(tr["clip_start_frame"]), int(tr["clip_end_frame"]) + 1):
                ok, f = cap.read()
                if not ok: break
                writer.write(f)
                count += 1
            writer.release()
            tr["clip_path"]        = str(clips_dir / name)
            tr["clip_frame_count"] = count
    finally:
        cap.release()


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(path, tracks, fps):
    fields = ["instance_id","label","first_frame","last_frame","first_seconds",
              "last_seconds","match_count","best_frame_index",
              "best_timestamp_seconds","best_confidence",
              "x1","y1","x2","y2","frame_path","crop_path","clip_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in tracks:
            x1,y1,x2,y2 = tuple(tr["best_box"])
            w.writerow({
                "instance_id": int(tr["instance_id"]),
                "label": tr["label"],
                "first_frame": int(tr["first_frame"]),
                "last_frame":  int(tr["last_frame"]),
                "first_seconds": f"{int(tr['first_frame'])/fps:.2f}",
                "last_seconds":  f"{int(tr['last_frame'])/fps:.2f}",
                "match_count":   int(tr["match_count"]),
                "best_frame_index": int(tr["best_frame_index"]),
                "best_timestamp_seconds": f"{float(tr['best_timestamp_seconds']):.2f}",
                "best_confidence": f"{float(tr['best_confidence']):.4f}",
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "frame_path": tr["frame_path"],
                "crop_path":  tr["crop_path"],
                "clip_path":  tr.get("clip_path",""),
            })


def write_clips_csv(path, tracks):
    fields = ["instance_id","start_frame","end_frame","start_seconds",
              "end_seconds","frame_count","match_count","clip_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tr in tracks:
            w.writerow({
                "instance_id":   int(tr["instance_id"]),
                "start_frame":   int(tr["clip_start_frame"]),
                "end_frame":     int(tr["clip_end_frame"]),
                "start_seconds": f"{float(tr['clip_start_seconds']):.2f}",
                "end_seconds":   f"{float(tr['clip_end_seconds']):.2f}",
                "frame_count":   int(tr.get("clip_frame_count", 0)),
                "match_count":   int(tr["match_count"]),
                "clip_path":     tr.get("clip_path",""),
            })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    paths   = build_paths(Path(args.output_dir).resolve(), video_path)
    backend = load_depth(args.depth_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = width * 2 if args.side_by_side else width
    writer = cv2.VideoWriter(
        str(paths["video_path"]),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (out_w, height),
    )
    if not writer.isOpened():
        cap.release(); raise RuntimeError("Cannot open video writer.")

    print(f"[video] {video_path.name}  {width}x{height} @ {fps:.1f} fps  ({total_frames} frames)")
    print(f"[cfg]   depth-stride={args.depth_stride}  depth-conf={args.depth_conf}"
          f"  roi-top={args.roi_top}  side-by-side={args.side_by_side}")
    print(f"[out]   {paths['video_path']}")

    cached_depth : np.ndarray | None = None
    tracks        : dict[int, dict]   = {}
    next_tid      = 1
    fidx          = 0
    frames_with_dets = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if args.max_frames and fidx >= args.max_frames: break

            # --- depth inference ---
            if fidx % args.depth_stride == 0 or cached_depth is None:
                cached_depth = infer_raw_depth(frame, backend, args.depth_input_size)

            score_map = build_score_map(frame, cached_depth, args.roi_top)
            dets      = score_to_detections(score_map, width, height, args.depth_conf, args.roi_top)

            # --- render heatmap frame ---
            heatmap   = depth_to_heatmap(cached_depth)          # INFERNO colourmap
            heat_ann  = heatmap.copy()                           # annotated copy

            for label, conf, box in dets:
                draw_box_on_heatmap(heat_ann, box, label, conf)

            # Always draw the white ROI boundary line
            draw_roi_line(heat_ann, args.roi_top)

            ts = fidx / fps
            if dets:
                frames_with_dets += 1

            # --- tracking ---
            used: set[int] = set()
            for label, conf, box in sorted(dets, key=lambda d: d[1], reverse=True):
                tid = match_track(tracks, fidx, box, used,
                                  args.track_iou, args.track_gap)
                if tid is None:
                    tr = new_track(next_tid, label, fidx, ts, box, conf,
                                   paths["frames_dir"], paths["crops_dir"],
                                   heat_ann, heatmap)
                    tracks[next_tid] = tr
                    used.add(next_tid)
                    next_tid += 1
                else:
                    tr = tracks[tid]
                    tr["last_frame"] = fidx
                    tr["last_box"]   = box
                    tr["match_count"] = int(tr["match_count"]) + 1
                    used.add(tid)
                    if conf > float(tr["best_confidence"]):
                        tr["best_confidence"]        = conf
                        tr["best_box"]               = box
                        tr["best_frame_index"]       = fidx
                        tr["best_timestamp_seconds"] = ts
                        save_assets(tr, paths["frames_dir"], paths["crops_dir"],
                                    heat_ann, heatmap, box)

            # --- status overlay (show only stable confirmed tracks) ---
            confirmed = sum(1 for tr in tracks.values()
                            if int(tr["match_count"]) >= args.min_match_count)
            depth_tag = "D" if fidx % args.depth_stride == 0 else "d"
            status    = (f"[{depth_tag}] Frame {fidx+1}/{total_frames or '?'} | "
                         f"Potholes: {confirmed}")
            cv2.putText(heat_ann, status, (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0),       2, cv2.LINE_AA)
            cv2.putText(heat_ann, status, (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),  1, cv2.LINE_AA)

            out_frame = compose_frame(frame, heat_ann, args.side_by_side)
            writer.write(out_frame)

            if fidx % 100 == 0:
                print(f"  frame {fidx}/{total_frames or '?'}  tracks={len(tracks)}"
                      f"  dets={len(dets)}")

            fidx += 1

    finally:
        cap.release()
        writer.release()

    raw_count = len(tracks)
    print(f"\n[done] Processed {fidx} frames | {frames_with_dets} with detections"
          f" | {raw_count} raw tracks")

    print(f"[dedup] min-match={args.min_match_count}  merge-iou={args.merge_iou} ...")
    tracks = deduplicate_tracks(tracks, args.min_match_count, args.merge_iou)
    print(f"[dedup] {raw_count} raw → {len(tracks)} unique potholes after deduplication")

    tracked = finalize(tracks, fps, fidx, args.clip_pre, args.clip_post)
    print("[clips] Saving instance clips …")
    save_clips(paths["video_path"], paths["clips_dir"], tracked, fps,
               (out_w, height))
    write_csv(paths["csv_path"],   tracked, fps)
    write_clips_csv(paths["clips_csv"], tracked)

    print(f"\n=== OUTPUT ===")
    print(f"  Heatmap video   : {paths['video_path']}")
    print(f"  Frames (heatmap): {paths['frames_dir']}")
    print(f"  Crops  (heatmap): {paths['crops_dir']}")
    print(f"  Clips           : {paths['clips_dir']}")
    print(f"  Instance CSV    : {paths['csv_path']}")
    print(f"  Clip CSV        : {paths['clips_csv']}")
    print(f"  Unique potholes : {len(tracked)}")


if __name__ == "__main__":
    main()
