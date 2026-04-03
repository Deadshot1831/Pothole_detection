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
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_VIDEO = "WhatsApp Video 2026-03-26 at 12.56.37.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect potholes in a video, save an annotated video, and export "
            "each detected pothole as a representative frame, crop, and clip."
        )
    )
    parser.add_argument(
        "--pipeline",
        choices=("hybrid", "yolo", "depth"),
        default="hybrid",
        help="Inference pipeline to run. Default: hybrid",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Legacy single-model override. Use with `--pipeline yolo` or "
            "`--pipeline depth`."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help=f"Path to the YOLO checkpoint used by the yolo/hybrid pipeline. Default: {DEFAULT_YOLO_MODEL}",
    )
    parser.add_argument(
        "--depth-model",
        default=DEFAULT_DEPTH_MODEL,
        help=f"Hugging Face depth model id used by the depth/hybrid pipeline. Default: {DEFAULT_DEPTH_MODEL}",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help=f"Path to the input video. Default: {DEFAULT_VIDEO}",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/pothole_detection",
        help="Base directory where outputs will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.80,
        help="Final acceptance threshold used by the yolo/hybrid pipeline.",
    )
    parser.add_argument(
        "--depth-conf",
        type=float,
        default=0.55,
        help="Depth score threshold used by the depth model and hybrid validation.",
    )
    parser.add_argument(
        "--hybrid-yolo-conf",
        type=float,
        default=0.35,
        help="Lower YOLO proposal threshold used before depth validation in hybrid mode.",
    )
    parser.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.30,
        help="IoU threshold used to match a pothole to an existing tracked instance.",
    )
    parser.add_argument(
        "--track-max-gap-frames",
        type=int,
        default=10,
        help="Maximum frame gap allowed when matching detections to an existing instance.",
    )
    parser.add_argument(
        "--clip-pre-seconds",
        type=float,
        default=1.0,
        help="Seconds to keep before the first frame in a detection clip.",
    )
    parser.add_argument(
        "--clip-post-seconds",
        type=float,
        default=1.0,
        help="Seconds to keep after the last frame in a detection clip.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick testing.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "run"


def draw_detection(
    image,
    box: tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    x1, y1, x2, y2 = box
    color = (0, 255, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    text_y = max(22, y1 - 10)
    cv2.putText(
        image,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def build_output_paths(base_output_dir: Path, video_path: Path) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sanitize_name(video_path.stem)}_{timestamp}"
    run_dir = base_output_dir / run_name
    frames_dir = run_dir / "frames"
    crops_dir = run_dir / "crops"
    clips_dir = run_dir / "clips"

    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "frames_dir": frames_dir,
        "crops_dir": crops_dir,
        "clips_dir": clips_dir,
        "video_path": run_dir / "annotated.mp4",
        "csv_path": run_dir / "instances.csv",
        "clips_csv_path": run_dir / "clips.csv",
    }


def ensure_input_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def choose_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_yolo_model_reference(model_reference: str) -> bool:
    return Path(model_reference).suffix.lower() == ".pt"


def load_yolo_backend(model_reference: str) -> dict[str, object]:
    model_path = Path(model_reference).resolve()
    ensure_input_file(model_path, "Model")
    model = YOLO(str(model_path))
    return {
        "type": "yolo",
        "model": model,
        "class_names": getattr(model.model, "names", {}),
    }


def load_depth_backend(model_reference: str) -> dict[str, object]:
    if is_yolo_model_reference(model_reference):
        raise RuntimeError(
            "Depth pipeline expects a Hugging Face depth model id, not a `.pt` checkpoint."
        )

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError(
            "The selected model is a Hugging Face depth model, but `transformers` "
            "is not installed. Install it before running this script."
        ) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(
            model_reference,
            local_files_only=True,
            use_fast=True,
        )
        model = AutoModelForDepthEstimation.from_pretrained(model_reference, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Hugging Face model `{model_reference}`. "
            "Check the model id and make sure the weights are already cached locally. "
            "If this is the first run on a new machine, download the model once with "
            "network access and then rerun offline."
        ) from exc

    device = choose_torch_device()
    model.to(device)
    model.eval()
    return {
        "type": "depth",
        "model": model,
        "processor": processor,
        "device": device,
    }


def load_inference_backend(
    pipeline: str,
    yolo_model_reference: str,
    depth_model_reference: str,
) -> dict[str, object]:
    if pipeline == "yolo":
        return {
            "type": "yolo",
            "yolo": load_yolo_backend(yolo_model_reference),
        }
    if pipeline == "depth":
        return {
            "type": "depth",
            "depth": load_depth_backend(depth_model_reference),
        }
    return {
        "type": "hybrid",
        "yolo": load_yolo_backend(yolo_model_reference),
        "depth": load_depth_backend(depth_model_reference),
    }


def clip_box_to_frame(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def apply_nms(
    detections: list[tuple[str, float, tuple[int, int, int, int]]],
    iou_threshold: float = 0.35,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    kept: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for detection in sorted(detections, key=lambda item: item[1], reverse=True):
        _, _, box = detection
        if any(calculate_iou(box, kept_box) >= iou_threshold for _, _, kept_box in kept):
            continue
        kept.append(detection)
    return kept


def detect_with_yolo(
    frame,
    backend: dict[str, object],
    predict_kwargs: dict[str, object],
    width: int,
    height: int,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    result = backend["model"].predict(frame, **predict_kwargs)[0]
    detections: list[tuple[str, float, tuple[int, int, int, int]]] = []
    if result.boxes is None:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    class_names = backend["class_names"]
    for box, confidence, class_id in zip(xyxy, confs, classes):
        clipped_box = clip_box_to_frame(tuple(box.tolist()), width, height)
        if clipped_box is None:
            continue
        label = class_names.get(class_id, str(class_id))
        detections.append((label, float(confidence), clipped_box))

    return detections


def infer_depth_score_map(frame, backend: dict[str, object]) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processor = backend["processor"]
    model = backend["model"]
    device = backend["device"]

    inputs = processor(images=rgb_frame, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=rgb_frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = depth.cpu().numpy().astype(np.float32)
    return build_depth_score_map(frame, depth_map)


def build_depth_score_map(frame, depth_map: np.ndarray) -> np.ndarray:
    height, width = depth_map.shape
    depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    large_blur = cv2.GaussianBlur(depth_norm, (0, 0), sigmaX=max(width / 20, 5), sigmaY=max(height / 20, 5))
    local_blur = cv2.GaussianBlur(depth_norm, (0, 0), sigmaX=max(width / 80, 3), sigmaY=max(height / 80, 3))
    depression_score = np.maximum(large_blur - local_blur, 0.0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    darkness_score = 1.0 - cv2.GaussianBlur(gray, (0, 0), sigmaX=3, sigmaY=3)

    edges = cv2.Laplacian(depth_norm, cv2.CV_32F, ksize=3)
    roughness_score = cv2.GaussianBlur(np.abs(edges), (0, 0), sigmaX=2, sigmaY=2)

    score_map = 0.55 * depression_score + 0.30 * darkness_score + 0.15 * roughness_score

    roi_mask = np.zeros_like(score_map, dtype=np.float32)
    top = int(height * 0.35)
    left = int(width * 0.05)
    right = int(width * 0.95)
    roi_mask[top:height, left:right] = 1.0
    score_map *= roi_mask

    max_value = float(score_map.max())
    if max_value > 0:
        score_map /= max_value
    return score_map


def extract_depth_detections(
    frame,
    score_map: np.ndarray,
    confidence_threshold: float,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    height, width = score_map.shape
    binary = (score_map >= confidence_threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[tuple[str, float, tuple[int, int, int, int]]] = []
    min_area = max(150, int(width * height * 0.0015))
    max_area = int(width * height * 0.20)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 12:
            continue

        aspect_ratio = w / max(h, 1)
        if aspect_ratio < 0.35 or aspect_ratio > 4.5:
            continue

        region_score = score_map[y : y + h, x : x + w]
        confidence = float(region_score.max())
        if confidence < confidence_threshold:
            continue

        clipped_box = clip_box_to_frame((x, y, x + w, y + h), width, height)
        if clipped_box is None:
            continue
        detections.append(("Potholes", confidence, clipped_box))

    return apply_nms(detections)


def compute_box_score(
    score_map: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    region = score_map[y1:y2, x1:x2]
    if region.size == 0:
        return (0.0, 0.0)
    return (float(region.max()), float(region.mean()))


def find_best_overlap(
    box: tuple[int, int, int, int],
    detections: list[tuple[str, float, tuple[int, int, int, int]]],
) -> tuple[int | None, float]:
    best_index = None
    best_iou = 0.0
    for index, (_, _, other_box) in enumerate(detections):
        iou = calculate_iou(box, other_box)
        if iou > best_iou:
            best_iou = iou
            best_index = index
    return best_index, best_iou


def detect_with_depth_model(
    frame,
    backend: dict[str, object],
    confidence_threshold: float,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    score_map = infer_depth_score_map(frame, backend)
    return extract_depth_detections(frame, score_map, confidence_threshold)


def detect_with_hybrid_model(
    frame,
    yolo_backend: dict[str, object],
    depth_backend: dict[str, object],
    predict_kwargs: dict[str, object],
    confidence_threshold: float,
    depth_confidence_threshold: float,
    hybrid_yolo_confidence: float,
    width: int,
    height: int,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    yolo_detections = detect_with_yolo(frame, yolo_backend, predict_kwargs, width, height)
    if not yolo_detections:
        return []

    score_map = infer_depth_score_map(frame, depth_backend)
    depth_detections = extract_depth_detections(frame, score_map, depth_confidence_threshold)

    fused_detections: list[tuple[str, float, tuple[int, int, int, int]]] = []
    used_depth_indices: set[int] = set()

    for label, yolo_confidence, box in yolo_detections:
        depth_peak, depth_mean = compute_box_score(score_map, box)
        depth_index, depth_iou = find_best_overlap(box, depth_detections)

        is_confirmed = (
            yolo_confidence >= confidence_threshold
            or (
                yolo_confidence >= hybrid_yolo_confidence
                and depth_peak >= depth_confidence_threshold
                and depth_iou >= 0.10
            )
        )
        if not is_confirmed:
            continue

        hybrid_confidence = min(
            1.0,
            max(yolo_confidence, depth_peak) + 0.10 * depth_iou + 0.05 * depth_mean,
        )
        fused_detections.append((label, hybrid_confidence, box))
        if depth_index is not None:
            used_depth_indices.add(depth_index)

    return apply_nms(fused_detections)


def detect_potholes(
    frame,
    backend: dict[str, object],
    predict_kwargs: dict[str, object] | None,
    confidence_threshold: float,
    depth_confidence_threshold: float,
    hybrid_yolo_confidence: float,
    width: int,
    height: int,
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    if backend["type"] == "yolo":
        return detect_with_yolo(frame, backend["yolo"], predict_kwargs or {}, width, height)
    if backend["type"] == "depth":
        return detect_with_depth_model(frame, backend["depth"], depth_confidence_threshold)
    return detect_with_hybrid_model(
        frame=frame,
        yolo_backend=backend["yolo"],
        depth_backend=backend["depth"],
        predict_kwargs=predict_kwargs or {},
        confidence_threshold=confidence_threshold,
        depth_confidence_threshold=depth_confidence_threshold,
        hybrid_yolo_confidence=hybrid_yolo_confidence,
        width=width,
        height=height,
    )


def calculate_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def save_instance_assets(
    track: dict[str, object],
    frames_dir: Path,
    crops_dir: Path,
    annotated_frame,
    frame,
    box: tuple[int, int, int, int],
) -> None:
    instance_id = int(track["instance_id"])
    frame_path = frames_dir / f"instance_{instance_id:04d}.jpg"
    crop_path = crops_dir / f"instance_{instance_id:04d}.jpg"
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]

    cv2.imwrite(str(frame_path), annotated_frame)
    cv2.imwrite(str(crop_path), crop)

    track["frame_path"] = str(frame_path)
    track["crop_path"] = str(crop_path)


def update_track_best_detection(
    track: dict[str, object],
    frames_dir: Path,
    crops_dir: Path,
    annotated_frame,
    frame,
    box: tuple[int, int, int, int],
    confidence: float,
    timestamp_seconds: float,
    frame_index: int,
) -> None:
    track["best_confidence"] = confidence
    track["best_box"] = box
    track["best_frame_index"] = frame_index
    track["best_timestamp_seconds"] = timestamp_seconds
    save_instance_assets(track, frames_dir, crops_dir, annotated_frame, frame, box)


def create_track(
    track_id: int,
    label: str,
    frame_index: int,
    timestamp_seconds: float,
    box: tuple[int, int, int, int],
    confidence: float,
    frames_dir: Path,
    crops_dir: Path,
    annotated_frame,
    frame,
) -> dict[str, object]:
    track = {
        "instance_id": track_id,
        "label": label,
        "first_frame": frame_index,
        "last_frame": frame_index,
        "last_box": box,
        "match_count": 1,
        "best_confidence": confidence,
        "best_box": box,
        "best_frame_index": frame_index,
        "best_timestamp_seconds": timestamp_seconds,
        "frame_path": "",
        "crop_path": "",
        "clip_path": "",
    }
    save_instance_assets(track, frames_dir, crops_dir, annotated_frame, frame, box)
    return track


def match_detection_to_track(
    tracks: dict[int, dict[str, object]],
    frame_index: int,
    box: tuple[int, int, int, int],
    matched_track_ids: set[int],
    iou_threshold: float,
    max_gap_frames: int,
) -> int | None:
    best_track_id = None
    best_iou = 0.0

    for track_id, track in tracks.items():
        if track_id in matched_track_ids:
            continue
        if frame_index - int(track["last_frame"]) > max_gap_frames:
            continue

        iou = calculate_iou(box, tuple(track["last_box"]))
        if iou >= iou_threshold and iou > best_iou:
            best_iou = iou
            best_track_id = track_id

    return best_track_id


def finalize_track_windows(
    tracks: dict[int, dict[str, object]],
    fps: float,
    total_frames: int,
    pre_seconds: float,
    post_seconds: float,
) -> list[dict[str, object]]:
    if total_frames <= 0:
        return []

    pre_frames = max(0, int(round(pre_seconds * fps)))
    post_frames = max(0, int(round(post_seconds * fps)))
    windows: list[dict[str, object]] = []

    for track_id in sorted(tracks):
        track = tracks[track_id]
        clip_start_frame = max(0, int(track["first_frame"]) - pre_frames)
        clip_end_frame = min(total_frames - 1, int(track["last_frame"]) + post_frames)
        track["clip_start_frame"] = clip_start_frame
        track["clip_end_frame"] = clip_end_frame
        track["clip_start_seconds"] = clip_start_frame / fps
        track["clip_end_seconds"] = clip_end_frame / fps
        windows.append(track)

    return windows


def save_instance_clips(
    source_video_path: Path,
    clips_dir: Path,
    tracks: list[dict[str, object]],
    fps: float,
    frame_size: tuple[int, int],
) -> None:
    if not tracks:
        return

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for clip export: {source_video_path}")

    try:
        for track in tracks:
            clip_name = (
                f"instance_{int(track['instance_id']):04d}_"
                f"t_{float(track['clip_start_seconds']):08.2f}s_"
                f"to_{float(track['clip_end_seconds']):08.2f}s.mp4"
            )
            clip_path = clips_dir / clip_name
            writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                frame_size,
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create clip writer: {clip_path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(track["clip_start_frame"]))
            frames_written = 0
            for _ in range(int(track["clip_start_frame"]), int(track["clip_end_frame"]) + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                frames_written += 1

            writer.release()
            track["clip_path"] = str(clip_path)
            track["clip_frame_count"] = frames_written
    finally:
        cap.release()


def write_instance_log(csv_path: Path, tracks: list[dict[str, object]], fps: float) -> None:
    headers = [
        "instance_id",
        "label",
        "first_frame",
        "last_frame",
        "first_seconds",
        "last_seconds",
        "match_count",
        "best_frame_index",
        "best_timestamp_seconds",
        "best_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "frame_path",
        "crop_path",
        "clip_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=headers)
        writer_csv.writeheader()
        for track in tracks:
            x1, y1, x2, y2 = tuple(track["best_box"])
            writer_csv.writerow(
                {
                    "instance_id": int(track["instance_id"]),
                    "label": track["label"],
                    "first_frame": int(track["first_frame"]),
                    "last_frame": int(track["last_frame"]),
                    "first_seconds": f"{int(track['first_frame']) / fps:.2f}",
                    "last_seconds": f"{int(track['last_frame']) / fps:.2f}",
                    "match_count": int(track["match_count"]),
                    "best_frame_index": int(track["best_frame_index"]),
                    "best_timestamp_seconds": f"{float(track['best_timestamp_seconds']):.2f}",
                    "best_confidence": f"{float(track['best_confidence']):.4f}",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "frame_path": track["frame_path"],
                    "crop_path": track["crop_path"],
                    "clip_path": track.get("clip_path", ""),
                }
            )


def write_clip_log(clips_csv_path: Path, tracks: list[dict[str, object]]) -> None:
    headers = [
        "instance_id",
        "start_frame",
        "end_frame",
        "start_seconds",
        "end_seconds",
        "frame_count",
        "match_count",
        "clip_path",
    ]

    with clips_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=headers)
        writer_csv.writeheader()
        for track in tracks:
            writer_csv.writerow(
                {
                    "instance_id": int(track["instance_id"]),
                    "start_frame": int(track["clip_start_frame"]),
                    "end_frame": int(track["clip_end_frame"]),
                    "start_seconds": f"{float(track['clip_start_seconds']):.2f}",
                    "end_seconds": f"{float(track['clip_end_seconds']):.2f}",
                    "frame_count": int(track.get("clip_frame_count", 0)),
                    "match_count": int(track["match_count"]),
                    "clip_path": track.get("clip_path", ""),
                }
            )


def main() -> None:
    args = parse_args()

    if args.pipeline == "yolo":
        yolo_model_reference = args.model or args.yolo_model
        depth_model_reference = args.depth_model
    elif args.pipeline == "depth":
        yolo_model_reference = args.yolo_model
        depth_model_reference = args.model or args.depth_model
    else:
        yolo_model_reference = args.yolo_model
        depth_model_reference = args.depth_model

    video_path = Path(args.video).resolve()
    output_paths = build_output_paths(Path(args.output_dir).resolve(), video_path)

    ensure_input_file(video_path, "Video")

    backend = load_inference_backend(
        pipeline=args.pipeline,
        yolo_model_reference=yolo_model_reference,
        depth_model_reference=depth_model_reference,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_paths["video_path"]),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video writer.")

    frames_processed = 0
    frames_with_detections = 0
    raw_detection_count = 0
    tracks: dict[int, dict[str, object]] = {}
    next_track_id = 1
    predict_kwargs = None
    if backend["type"] in {"yolo", "hybrid"}:
        yolo_confidence = args.conf if backend["type"] == "yolo" else args.hybrid_yolo_conf
        predict_kwargs = {
            "conf": yolo_confidence,
            "verbose": False,
            "save": False,
            "project": str(output_paths["run_dir"]),
            "name": "ultralytics_cache",
            "exist_ok": True,
        }

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.max_frames is not None and frames_processed >= args.max_frames:
                break

            annotated_frame = frame.copy()
            frame_detections = detect_potholes(
                frame=frame,
                backend=backend,
                predict_kwargs=predict_kwargs,
                confidence_threshold=args.conf,
                depth_confidence_threshold=args.depth_conf,
                hybrid_yolo_confidence=args.hybrid_yolo_conf,
                width=width,
                height=height,
            )
            for label, confidence, box in frame_detections:
                draw_detection(annotated_frame, box, label, confidence)

            if frame_detections:
                frames_with_detections += 1

            timestamp_seconds = frames_processed / fps
            matched_track_ids: set[int] = set()
            for label, confidence, box in sorted(frame_detections, key=lambda item: item[1], reverse=True):
                raw_detection_count += 1
                matched_track_id = match_detection_to_track(
                    tracks=tracks,
                    frame_index=frames_processed,
                    box=box,
                    matched_track_ids=matched_track_ids,
                    iou_threshold=args.track_iou_threshold,
                    max_gap_frames=args.track_max_gap_frames,
                )

                if matched_track_id is None:
                    track = create_track(
                        track_id=next_track_id,
                        label=label,
                        frame_index=frames_processed,
                        timestamp_seconds=timestamp_seconds,
                        box=box,
                        confidence=confidence,
                        frames_dir=output_paths["frames_dir"],
                        crops_dir=output_paths["crops_dir"],
                        annotated_frame=annotated_frame,
                        frame=frame,
                    )
                    tracks[next_track_id] = track
                    matched_track_ids.add(next_track_id)
                    next_track_id += 1
                    continue

                track = tracks[matched_track_id]
                track["last_frame"] = frames_processed
                track["last_box"] = box
                track["match_count"] = int(track["match_count"]) + 1
                matched_track_ids.add(matched_track_id)

                if confidence > float(track["best_confidence"]):
                    update_track_best_detection(
                        track=track,
                        frames_dir=output_paths["frames_dir"],
                        crops_dir=output_paths["crops_dir"],
                        annotated_frame=annotated_frame,
                        frame=frame,
                        box=box,
                        confidence=confidence,
                        timestamp_seconds=timestamp_seconds,
                        frame_index=frames_processed,
                    )

            status_text = (
                f"Frame {frames_processed + 1}/{total_frames or '?'} | "
                f"Unique potholes: {len(tracks)}"
            )
            cv2.putText(
                annotated_frame,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated_frame)

            frames_processed += 1

    finally:
        cap.release()
        writer.release()

    tracked_instances = finalize_track_windows(
        tracks=tracks,
        fps=fps,
        total_frames=frames_processed,
        pre_seconds=args.clip_pre_seconds,
        post_seconds=args.clip_post_seconds,
    )
    save_instance_clips(
        source_video_path=output_paths["video_path"],
        clips_dir=output_paths["clips_dir"],
        tracks=tracked_instances,
        fps=fps,
        frame_size=(width, height),
    )
    write_instance_log(output_paths["csv_path"], tracked_instances, fps)
    write_clip_log(output_paths["clips_csv_path"], tracked_instances)

    print(f"Processed frames: {frames_processed}")
    print(f"Frames with potholes: {frames_with_detections}")
    print(f"Raw frame-level detections: {raw_detection_count}")
    print(f"Unique pothole instances saved: {len(tracked_instances)}")
    print(f"Detection clips saved: {len(tracked_instances)}")
    print(f"Annotated video: {output_paths['video_path']}")
    print(f"Representative frames: {output_paths['frames_dir']}")
    print(f"Representative crops: {output_paths['crops_dir']}")
    print(f"Instance log: {output_paths['csv_path']}")
    print(f"Detection clips: {output_paths['clips_dir']}")
    print(f"Clip log: {output_paths['clips_csv_path']}")


if __name__ == "__main__":
    main()
