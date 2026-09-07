"""Shared core behind hybrid_ball_test.py (annotated video) and
prelabel_for_label_studio.py (Label Studio pre-annotation export): primary
+backup ball detection (fine-tuned yolo26x primary, fine-tuned yolo11x
backup - see model_benchmark.py, which trained both) combined with:

  1. The project's own production BallTracker (BallDetection.ballDetection):
     once locked on, predicts the ball's next position from a Kalman filter
     and searches a small crop around that prediction (zoom_crop), which
     grows automatically the longer a real detection goes missing and
     shrinks back down once reacquired; candidates outside the prediction's
     uncertainty-scaled gate are rejected regardless of confidence
     (match/revalidate). A zoom-crop miss falls through to the same
     full-frame tiled search the initial-acquisition path uses (these
     from-scratch fine-tuned models don't perform as well on a small,
     resized-up crop as the production detector this mechanism was tuned
     around does - see run_hybrid_detection's own comment) rather than
     giving up on the frame, and a periodic full-frame revalidation check
     (see REVALIDATION_INTERVAL_SECONDS) catches a lock that's quietly
     drifted onto the wrong object.
  2. A minimum detection-size filter, CALCULATED from this video's own
     footage rather than a fixed guess (calibrate_min_size): a first pass
     samples the primary model's own confident detections and takes their
     median box size as "what the real, in-play ball looks like here";
     anything smaller than SIZE_FRACTION_OF_MEDIAN of that is rejected
     outright. Aimed at a ball-shaped false positive belonging to an
     unrelated game visible in the background - being farther from the
     camera than the court actually being tracked, it always projects to a
     smaller apparent size than the real ball.
  3. A motion check via dense optical flow (Farneback): a candidate's own
     box region must show real average motion between consecutive frames
     to be accepted.

Confidence threshold is deliberately low - the size, motion, and
tracker-gate filters carry the false-positive rejection instead of a high
confidence cutoff, so recall isn't sacrificed to get precision.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parents[2]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from BallDetection.ballDetection import (  # noqa: E402
    REVALIDATION_INTERVAL_SECONDS,
    BallTracker,
    ZOOM_IMG_SIZE,
    _revalidation_step,
    box_diagonal,
    detect_in_crop,
    detect_tiled,
    generate_tiles,
)
from ultralytics import YOLO  # noqa: E402

ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR.parent / "models"
PRIMARY_WEIGHTS = MODELS_DIR / "ballDetection_yolo26x_best.pt"
# yolo11x over rtdetr-x: the model_benchmark.py comparison showed rtdetr-x's
# uptime advantage (99.8% detection rate, 5 gaps) came at the cost of the
# worst precision of the four models (0.780) - exactly the false-positive-
# prone behaviour this is trying to eliminate, not import as the fallback.
# yolo11x still has real uptime of its own (84.2% alone, second-best gap
# count after rtdetr-x) with meaningfully better precision (0.846).
BACKUP_WEIGHTS = MODELS_DIR / "ballDetection_yolo11x_best.pt"

TILE_GRID = (2, 2)
TILE_OVERLAP_FRACTION = 0.2
ACQUISITION_IMG_SIZE = 960
CONF_THRESHOLD = 0.10
DEVICE = 0

SIZE_SAMPLE_STRIDE = 5
SIZE_SAMPLE_MIN_CONF = 0.5
SIZE_FRACTION_OF_MEDIAN = 0.5

FLOW_RESIZE_WIDTH = 480
MOTION_MIN_MAGNITUDE_PX = 0.6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_models() -> tuple[YOLO, YOLO]:
    if not PRIMARY_WEIGHTS.exists() or not BACKUP_WEIGHTS.exists():
        raise SystemExit(
            f"Fine-tuned weights not found ({PRIMARY_WEIGHTS} / {BACKUP_WEIGHTS}) - run model_benchmark.py first."
        )
    log("Loading models (primary: fine-tuned yolo26x, backup: fine-tuned yolo11x)...")
    return YOLO(str(PRIMARY_WEIGHTS)), YOLO(str(BACKUP_WEIGHTS))


def video_info(video_path: Path) -> tuple[float, int, int, int]:
    """(fps, frame_w, frame_h, total_frames)."""
    probe = cv2.VideoCapture(str(video_path))
    try:
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, w, h, total
    finally:
        probe.release()


def calibrate_min_size(video_path: Path, model: YOLO, tiles, device=DEVICE) -> float:
    """Samples the primary model's own confident detections across the
    video (every SIZE_SAMPLE_STRIDE-th frame) and returns
    SIZE_FRACTION_OF_MEDIAN times their median box diagonal - "how big a
    real ball is in this footage," calculated from the footage itself
    rather than guessed."""
    cap = cv2.VideoCapture(str(video_path))
    diagonals: list[float] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % SIZE_SAMPLE_STRIDE == 0:
                dets = detect_tiled(model, frame, tiles, ACQUISITION_IMG_SIZE, SIZE_SAMPLE_MIN_CONF, device)
                diagonals.extend(box_diagonal(box) for box, _ in dets)
            frame_idx += 1
    finally:
        cap.release()

    if not diagonals:
        log("WARNING: no confident detections during size calibration - proceeding with no size floor.")
        return 0.0

    median = float(np.median(diagonals))
    floor = median * SIZE_FRACTION_OF_MEDIAN
    log(
        f"Size calibration: {len(diagonals)} confident sample(s) across {frame_idx} frames, "
        f"median diagonal {median:.1f}px -> rejecting anything under {floor:.1f}px "
        f"({SIZE_FRACTION_OF_MEDIAN:.0%} of median)."
    )
    return floor


def flow_magnitude_in_box(flow: np.ndarray, box, scale: float) -> float:
    """Average optical-flow magnitude, in ORIGINAL-frame pixels/frame,
    inside `box` (full-frame coordinates) - flow itself was computed on a
    frame downscaled by `scale`."""
    x1, y1, x2, y2 = box
    fx1, fy1 = max(0, int(x1 * scale)), max(0, int(y1 * scale))
    fx2 = min(flow.shape[1], int(np.ceil(x2 * scale)))
    fy2 = min(flow.shape[0], int(np.ceil(y2 * scale)))
    if fx2 <= fx1 or fy2 <= fy1:
        return 0.0
    region = flow[fy1:fy2, fx1:fx2]
    mag = np.hypot(region[..., 0], region[..., 1])
    return float(mag.mean()) / scale


@dataclass
class FrameResult:
    frame_idx: int
    frame: np.ndarray
    matched: Optional[tuple]  # (box[x1,y1,x2,y2], confidence) or None
    used_backup: bool
    zoom_box: Optional[tuple]  # the crop actually searched, or None if full-frame tiled search was used
    tracking_mode: bool  # a track existed and wasn't lost, going into this frame
    due_for_revalidation: bool
    zoom_crop_succeeded: bool  # zoom crop was tried AND it alone found the match (no fallback needed)
    all_rejected: list = field(default_factory=list)  # (box, conf, "size"|"motion") - every filtered-out candidate


def run_hybrid_detection(
    video_path: Path,
    primary: YOLO,
    backup: YOLO,
    min_diagonal: float,
    tracker: BallTracker,
    progress_every: int = 200,
) -> Iterator[FrameResult]:
    """Runs the full primary/backup + size/motion/tracker-gate pipeline over
    every frame of video_path, advancing `tracker` (caller-owned, so it can
    inspect tracker.trail()/is_tracking/etc. between yields) and yielding
    one FrameResult per frame in order."""
    fps, frame_w, frame_h, total_frames = video_info(video_path)
    tiles = generate_tiles(frame_w, frame_h, TILE_GRID, TILE_OVERLAP_FRACTION)

    revalidation_interval_frames = max(1, round(REVALIDATION_INTERVAL_SECONDS * fps))
    frames_since_revalidation = 0
    flow_scale = FLOW_RESIZE_WIDTH / frame_w
    flow_size = (FLOW_RESIZE_WIDTH, max(1, round(frame_h * flow_scale)))
    prev_gray_small = None

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            small = cv2.resize(frame, flow_size, interpolation=cv2.INTER_LINEAR)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            flow = None
            if prev_gray_small is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray_small, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prev_gray_small = gray_small

            def classify(raw, _flow=flow):
                survivors, rejected = [], []
                for box, conf in raw:
                    if box_diagonal(box) < min_diagonal:
                        rejected.append((box, conf, "size"))
                        continue
                    if _flow is not None and flow_magnitude_in_box(_flow, box, flow_scale) < MOTION_MIN_MAGNITUDE_PX:
                        rejected.append((box, conf, "motion"))
                        continue
                    survivors.append((box, conf))
                return survivors, rejected

            predicted, std = tracker.predict()
            tracking_mode = predicted is not None and not tracker.is_lost()
            due_for_revalidation, frames_since_revalidation = _revalidation_step(
                frames_since_revalidation, tracking_mode, revalidation_interval_frames
            )
            if predicted is not None and tracker.is_lost():
                tracker.lose()
                predicted, std = None, None

            def select(candidates, _predicted=predicted, _std=std, _due=due_for_revalidation):
                return tracker.revalidate(candidates, _predicted, _std) if _due else tracker.match(
                    candidates, _predicted, _std
                )

            all_rejected: list = []
            zoom_box = None
            matched = None
            zoom_crop_succeeded = False

            # Zoom crop tried first when there's a track to follow - cheap,
            # and the clearest on-screen demonstration of "a shrinking/
            # growing search area around the predicted position." But
            # trusting it as the ONLY search each frame made detection rate
            # collapse from 86.6% to ~41% in testing: these from-scratch
            # fine-tuned models, unlike the production detector this
            # mechanism was tuned around, don't perform nearly as well on a
            # small, resized-up crop as they do seeing more of the frame -
            # so a crop miss falls straight through to the same full-frame
            # search the acquisition path uses, instead of giving up for
            # the frame.
            if tracking_mode and not due_for_revalidation:
                zoom_box = tracker.zoom_crop(predicted, std)
                raw = detect_in_crop(primary, frame, zoom_box, ZOOM_IMG_SIZE, CONF_THRESHOLD, DEVICE)
                candidates, rejected = classify(raw)
                all_rejected += rejected
                matched = tracker.match(candidates, predicted, std)
                if matched is not None:
                    zoom_crop_succeeded = True

            if matched is None:
                zoom_box = None
                raw = detect_tiled(primary, frame, tiles, ACQUISITION_IMG_SIZE, CONF_THRESHOLD, DEVICE)
                candidates, rejected = classify(raw)
                all_rejected += rejected
                matched = select(candidates)

            used_backup = False
            if matched is None:
                raw_backup = detect_tiled(backup, frame, tiles, ACQUISITION_IMG_SIZE, CONF_THRESHOLD, DEVICE)
                backup_candidates, backup_rejected = classify(raw_backup)
                all_rejected += backup_rejected
                matched = select(backup_candidates)
                used_backup = matched is not None

            tracker.observe(matched, frame_idx)

            yield FrameResult(
                frame_idx=frame_idx,
                frame=frame,
                matched=matched,
                used_backup=used_backup,
                zoom_box=zoom_box,
                tracking_mode=tracking_mode,
                due_for_revalidation=due_for_revalidation,
                zoom_crop_succeeded=zoom_crop_succeeded,
                all_rejected=all_rejected,
            )

            frame_idx += 1
            if progress_every and frame_idx % progress_every == 0:
                log(f"  {frame_idx}/{total_frames} frames...")
    finally:
        cap.release()
