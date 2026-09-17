"""Prototype: compares the production pipeline's diameter-based ball
height/world-position estimate (BallDetection.ballDetection.
estimate_ball_world_position - real ball diameter vs. apparent box size,
back-projected through the calibrated camera pose) against Apple's Depth
Pro (a monocular metric depth model, via HuggingFace transformers -
https://huggingface.co/apple/DepthPro-hf) sampled at the same ball pixel on
the same frames, for an already-processed job.

Nothing here is wired into the real pipeline - this is purely to see
whether a general-purpose depth model's reading at the ball's pixel is
close enough to the current diameter trick to be worth pursuing further,
before investing in real integration.

Depth Pro's raw output is a "canonical" inverse depth field that's
resolution-independent and only needs multiplying by a focal length (in
pixels, for the image's actual width) to become metric depth - see
image_processing_depth_pro.py's post_process_depth_estimation upstream.
Rather than let Depth Pro use its own self-predicted focal length/FOV (a
headline feature of the model - it doesn't need external calibration), this
script deliberately substitutes this job's own calibrated focal length
(CAMERA_POSE_FILE's K) instead, and back-projects through that same K - so
the comparison isolates "diameter trick vs. neural depth" while holding the
camera model fixed, rather than also mixing in disagreement about the
camera itself. Depth Pro's own self-predicted focal length is still logged
per-frame as a separate sanity readout.

Usage (from repo root, with the API venv active - needs transformers,
accelerate, torch/CUDA, plus a one-time ~1.9GB weight download the first
time it runs):
    python -m Backend.MachineLearning.ballDetection.depth_pro_experiment
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "Analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from BallDetection.ballDetection import (  # noqa: E402
    centre_of,
    estimate_ball_world_position,
    load_camera_pose,
)
from CourtDetection.court import camera_to_world  # noqa: E402

JOB_ID = "472c96d203da"
JOB_OUTPUT = Path(__file__).resolve().parents[2] / "API" / "data" / JOB_ID / "output"
# The clean, unannotated source at the same resolution ball detection and
# court calibration both ran at (not analysis.mp4 - that one has detection
# boxes/overlays burned in, which would contaminate a depth model's input).
VIDEO_NAME = "source_reduced.mp4"

DEPTH_PRO_MODEL_ID = "apple/DepthPro-hf"
# Capped rather than running Depth Pro over every real frame in the job -
# it's a large model (a full forward pass per frame is seconds, not
# milliseconds), and a few hundred points spread across the whole video is
# already enough to see whether the two methods track each other.
MAX_SAMPLE_FRAMES = 300

OUTPUT_DIR = Path(__file__).resolve().parent / "depth_pro_experiment_results"
RESULTS_NAME = f"{JOB_ID}_comparison.json"


def log(msg: str) -> None:
    print(f"[depth_pro_experiment] {msg}", flush=True)


def load_trajectory_and_baseline(output_path: Path):
    trajectory = json.loads((output_path / "ball_trajectory.json").read_text())["frames"]
    speed_log = json.loads((output_path / "ball_speed.json").read_text())
    return trajectory, speed_log


def pick_sample_frames(trajectory: list[dict], speed_log: list[dict], max_frames: int) -> list[int]:
    """Evenly-spaced real (actually-detected) frame indices that also have a
    baseline height_m to compare against - same "real, not interpolated"
    restriction the production pipeline itself uses before trusting a box's
    apparent size for anything physics-derived (see ballDetection.py's own
    extract_flight_segments)."""
    usable = [
        i for i, (t, s) in enumerate(zip(trajectory, speed_log))
        if t["status"] == "real" and s["height_m"] is not None
    ]
    if len(usable) <= max_frames:
        return usable
    step = len(usable) / max_frames
    return [usable[int(i * step)] for i in range(max_frames)]


def load_depth_pro(device: str):
    from transformers import DepthProForDepthEstimation, DepthProImageProcessor

    log(f"Loading {DEPTH_PRO_MODEL_ID} (first run downloads ~1.9GB of weights)...")
    t0 = time.perf_counter()
    processor = DepthProImageProcessor.from_pretrained(DEPTH_PRO_MODEL_ID)
    model = DepthProForDepthEstimation.from_pretrained(DEPTH_PRO_MODEL_ID, torch_dtype=torch.float16)
    model = model.to(device).eval()
    log(f"Depth Pro loaded in {time.perf_counter() - t0:.1f}s on {device}.")
    return processor, model


def depth_pro_metric_depth_at(processor, model, device, frame_rgb: np.ndarray, focal_px: float, pixel: np.ndarray):
    """Runs Depth Pro on one full frame and returns
    (metric_depth_at_pixel, model_own_focal_length_px) - see this module's
    docstring for why the metric scale is computed from our OWN focal_px
    instead of Depth Pro's self-predicted one, and why that's still logged
    separately."""
    height, width = frame_rgb.shape[:2]
    image = Image.fromarray(frame_rgb)

    inputs = processor(images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model(**inputs)

    # Raw canonical inverse depth at the model's own internal resolution -
    # resolution-independent per-pixel (see module docstring), so it's safe
    # to resize first and scale to metric after.
    raw = outputs.predicted_depth  # shape (1, h, w)
    raw_resized = torch.nn.functional.interpolate(
        raw.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze()

    own_fov = outputs.field_of_view
    own_focal_px = None
    if own_fov is not None:
        own_focal_px = float(0.5 * width / torch.tan(0.5 * torch.deg2rad(own_fov[0])).item())

    x, y = pixel
    xi = int(np.clip(round(x), 0, width - 1))
    yi = int(np.clip(round(y), 0, height - 1))
    raw_at_pixel = float(raw_resized[yi, xi].item())
    if raw_at_pixel <= 0:
        return None, own_focal_px

    metric_depth = focal_px / (raw_at_pixel * width)
    return metric_depth, own_focal_px


def main() -> None:
    if not JOB_OUTPUT.exists():
        raise SystemExit(f"Job output not found: {JOB_OUTPUT}")

    video_path = JOB_OUTPUT / VIDEO_NAME
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    camera_pose = load_camera_pose(JOB_OUTPUT)
    if camera_pose is None:
        raise SystemExit(f"No camera pose calibrated for job {JOB_ID} - nothing to back-project against.")
    K, rvec, tvec = camera_pose
    focal_px = float(K[0, 0])
    log(f"Job {JOB_ID}: calibrated focal length {focal_px:.1f}px.")

    trajectory, speed_log = load_trajectory_and_baseline(JOB_OUTPUT)
    sample_idxs = pick_sample_frames(trajectory, speed_log, MAX_SAMPLE_FRAMES)
    log(f"Sampling {len(sample_idxs)} real, baseline-covered frames out of {len(trajectory)} total.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_depth_pro(device)

    cap = cv2.VideoCapture(str(video_path))

    rows = []
    t_start = time.perf_counter()
    for n, frame_idx in enumerate(sample_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        box = trajectory[frame_idx]["box"]
        pixel = centre_of(box)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        depth_metric, own_focal_px = depth_pro_metric_depth_at(processor, model, device, frame_rgb, focal_px, pixel)

        baseline_world = estimate_ball_world_position(box, camera_pose)
        baseline_height = None if baseline_world is None else float(baseline_world[2])

        depth_pro_height = None
        depth_pro_world = None
        if depth_metric is not None:
            ray = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
            point_camera = depth_metric * ray
            point_world = camera_to_world(point_camera, rvec, tvec)
            depth_pro_world = [float(v) for v in point_world]
            depth_pro_height = float(point_world[2])

        rows.append({
            "frame_idx": frame_idx,
            "pixel": [float(pixel[0]), float(pixel[1])],
            "baseline_height_m": baseline_height,
            "depth_pro_height_m": depth_pro_height,
            "baseline_world": None if baseline_world is None else [float(v) for v in baseline_world],
            "depth_pro_world": depth_pro_world,
            "depth_pro_own_focal_px": own_focal_px,
        })

        if (n + 1) % 25 == 0 or n == len(sample_idxs) - 1:
            elapsed = time.perf_counter() - t_start
            log(f"{n + 1}/{len(sample_idxs)} frames done ({elapsed:.1f}s, {elapsed / (n + 1):.2f}s/frame).")

    cap.release()

    both = [r for r in rows if r["baseline_height_m"] is not None and r["depth_pro_height_m"] is not None]
    log(f"Both methods produced a height on {len(both)}/{len(rows)} sampled frames.")

    if both:
        diffs = np.array([r["depth_pro_height_m"] - r["baseline_height_m"] for r in both])
        baseline_heights = np.array([r["baseline_height_m"] for r in both])
        depth_pro_heights = np.array([r["depth_pro_height_m"] for r in both])
        correlation = float(np.corrcoef(baseline_heights, depth_pro_heights)[0, 1]) if len(both) >= 2 else float("nan")

        log(f"Height diff (Depth Pro - baseline): mean {diffs.mean():+.2f}m | "
            f"median {np.median(diffs):+.2f}m | mean abs {np.abs(diffs).mean():.2f}m | "
            f"std {diffs.std():.2f}m")
        log(f"Baseline height range: [{baseline_heights.min():.2f}, {baseline_heights.max():.2f}]m, "
            f"mean {baseline_heights.mean():.2f}m")
        log(f"Depth Pro height range: [{depth_pro_heights.min():.2f}, {depth_pro_heights.max():.2f}]m, "
            f"mean {depth_pro_heights.mean():.2f}m")
        log(f"Pearson correlation (baseline vs. Depth Pro height): {correlation:.3f}")

        own_focals = [r["depth_pro_own_focal_px"] for r in both if r["depth_pro_own_focal_px"] is not None]
        if own_focals:
            log(f"Depth Pro's own self-predicted focal length: mean {np.mean(own_focals):.1f}px "
                f"vs. this job's calibrated {focal_px:.1f}px "
                f"({np.mean(own_focals) / focal_px:.2f}x).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / RESULTS_NAME
    with open(results_path, "w") as f:
        json.dump({"job_id": JOB_ID, "calibrated_focal_px": focal_px, "rows": rows}, f, indent=2)
    log(f"Per-frame comparison saved: {results_path}")
    log(f"Total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
