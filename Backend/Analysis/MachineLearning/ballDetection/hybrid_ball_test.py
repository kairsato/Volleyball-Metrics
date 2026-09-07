"""Renders an annotated video from hybrid_ball_detector.run_hybrid_detection
- see that module's own docstring for what the pipeline actually does. This
script's own job is just visualization: draws the search area actually used
each frame (full-frame tiles vs. the tracker's zoom crop), every rejected
candidate color-coded by why it was rejected, the final pick color-coded by
which model found it, and the tracker's own smoothed trail.

Usage (from Backend/Analysis/MachineLearning/):
    python hybrid_ball_test.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# hybrid_ball_detector must be imported before anything from BallDetection -
# it's the one that puts Backend/Analysis/ on sys.path as an import-time
# side effect (see its own top of file), so BallTracker/generate_tiles are
# re-exported from it here rather than imported from BallDetection.ballDetection
# directly, which would run before that path setup has happened.
from hybrid_ball_detector import (  # noqa: E402
    BallTracker,
    TILE_GRID,
    TILE_OVERLAP_FRACTION,
    calibrate_min_size,
    generate_tiles,
    load_models,
    log,
    run_hybrid_detection,
    video_info,
)

# The "faststart" remux (see prelabel_for_label_studio.py's own note) - not
# strictly needed just to run detection, but keeping every script pointed
# at the same file avoids two copies of "the benchmark video" drifting.
BENCHMARK_VIDEO = Path(r"C:\Users\Kai\Documents\VolleyballArea\verycut - Copy - faststart.mp4")
OUTPUT_DIR = Path(__file__).resolve().parent / "hybrid_ball_test_results"
ANNOTATED_VIDEO_NAME = "hybrid_ball_test.mp4"

PRIMARY_COLOR = (0, 255, 0)  # green - yolo26x's own pick
BACKUP_COLOR = (0, 165, 255)  # orange - yolo11x backup's pick
TILE_GRID_COLOR = (90, 90, 90)  # dim gray - full-frame acquisition search area (not currently tracking)
ZOOM_BOX_COLOR = (255, 255, 0)  # cyan - BallTracker's own local search crop (currently tracking; grows on a miss)
REJECTED_SIZE_COLOR = (0, 0, 200)  # red - candidate found, rejected for being too small
REJECTED_MOTION_COLOR = (0, 220, 220)  # yellow - candidate found, rejected for not moving
TRAIL_COLOR = (0, 200, 255)
REJECT_COLORS = {"size": REJECTED_SIZE_COLOR, "motion": REJECTED_MOTION_COLOR}


def draw_legend(frame: np.ndarray) -> None:
    entries = [
        ("search area (full-frame tiles)", TILE_GRID_COLOR),
        ("search area (tracker zoom crop)", ZOOM_BOX_COLOR),
        ("rejected: too small", REJECTED_SIZE_COLOR),
        ("rejected: not moving", REJECTED_MOTION_COLOR),
        ("picked: yolo26x", PRIMARY_COLOR),
        ("picked: yolo11x backup", BACKUP_COLOR),
    ]
    x, y = 12, 24
    for text, color in entries:
        cv2.rectangle(frame, (x, y - 12), (x + 22, y + 2), color, -1)
        cv2.putText(frame, text, (x + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main() -> None:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BENCHMARK_VIDEO.exists():
        raise SystemExit(f"Benchmark video not found: {BENCHMARK_VIDEO}")

    primary, backup = load_models()
    fps, frame_w, frame_h, total_frames = video_info(BENCHMARK_VIDEO)
    tiles = generate_tiles(frame_w, frame_h, TILE_GRID, TILE_OVERLAP_FRACTION)
    tile_rects_px = [tuple(int(v) for v in t) for t in tiles]

    t_calib_start = time.perf_counter()
    min_diagonal = calibrate_min_size(BENCHMARK_VIDEO, primary, tiles)
    calib_elapsed = time.perf_counter() - t_calib_start

    log(f"Running detection pass over {total_frames} frames ({frame_w}x{frame_h} @ {fps:.1f}fps)...")

    out_video_path = OUTPUT_DIR / ANNOTATED_VIDEO_NAME
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    # No court calibration for this ad hoc test clip, so matrix/camera_pose
    # are both None - the tracker still does everything this needs (Kalman
    # position/uncertainty, zoom_crop, gated match) purely in pixel space.
    tracker = BallTracker(None, fps, frame_w, frame_h, None)

    frames_primary = 0
    frames_backup = 0
    frames_none = 0
    frames_tracking_mode = 0
    frames_zoom_crop_succeeded = 0
    frames_revalidating = 0
    rejected_size = 0
    rejected_motion = 0
    frame_idx = 0
    t_detect_start = time.perf_counter()

    try:
        for result in run_hybrid_detection(BENCHMARK_VIDEO, primary, backup, min_diagonal, tracker):
            frame = result.frame
            frame_idx = result.frame_idx + 1

            if result.tracking_mode and not result.due_for_revalidation:
                frames_tracking_mode += 1
                if result.zoom_crop_succeeded:
                    frames_zoom_crop_succeeded += 1
            if result.due_for_revalidation:
                frames_revalidating += 1

            if result.matched is not None:
                if result.used_backup:
                    frames_backup += 1
                else:
                    frames_primary += 1
            else:
                frames_none += 1

            # --- draw ---
            if result.zoom_box is not None:
                zx1, zy1, zx2, zy2 = (int(v) for v in result.zoom_box)
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), ZOOM_BOX_COLOR, 1)
            else:
                for tx1, ty1, tx2, ty2 in tile_rects_px:
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), TILE_GRID_COLOR, 1)

            for box, conf, reason in result.all_rejected:
                if reason == "size":
                    rejected_size += 1
                else:
                    rejected_motion += 1
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), REJECT_COLORS[reason], 1)

            if result.matched is not None:
                box, conf = result.matched
                x1, y1, x2, y2 = (int(v) for v in box)
                color = BACKUP_COLOR if result.used_backup else PRIMARY_COLOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{'yolo11x (backup)' if result.used_backup else 'yolo26x'} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # BallTracker's own trail - one point per frame while tracking
            # (real detections and coasted/predicted positions alike), so
            # unlike a raw list of only-accepted detections this can't
            # develop a "teleport" segment across a multi-frame gap.
            trail_pts = tracker.trail()
            for i in range(1, len(trail_pts)):
                p1 = tuple(int(v) for v in trail_pts[i - 1])
                p2 = tuple(int(v) for v in trail_pts[i])
                cv2.line(frame, p1, p2, TRAIL_COLOR, 2)

            draw_legend(frame)
            writer.write(frame)
    finally:
        writer.release()

    detect_elapsed = time.perf_counter() - t_detect_start
    total_elapsed = time.perf_counter() - t_start
    detection_rate = (frames_primary + frames_backup) / frame_idx if frame_idx else 0.0
    tracking_rate = frames_tracking_mode / frame_idx if frame_idx else 0.0
    zoom_success_rate = frames_zoom_crop_succeeded / frames_tracking_mode if frames_tracking_mode else 0.0

    log("Done.")
    log(
        f"Frames: {frame_idx} total | primary yolo26x: {frames_primary} "
        f"| backup yolo11x: {frames_backup} | neither: {frames_none} "
        f"| detection rate: {detection_rate:.1%}"
    )
    log(
        f"Frames the zoom crop was attempted (locked on): {frames_tracking_mode} ({tracking_rate:.1%}) "
        f"| zoom crop alone found the ball: {frames_zoom_crop_succeeded} ({zoom_success_rate:.1%} of those) "
        f"| needed the full-frame fallback: {frames_tracking_mode - frames_zoom_crop_succeeded}"
    )
    log(f"Frames on a periodic full-frame revalidation check: {frames_revalidating}")
    log(f"Candidates rejected - size filter: {rejected_size} | motion filter: {rejected_motion}")
    log(f"Calculated minimum ball size: {min_diagonal:.1f}px diagonal")
    log(
        f"Timing: size-calibration pass {calib_elapsed:.1f}s | detection+annotation pass {detect_elapsed:.1f}s "
        f"| total {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)"
    )
    log(f"Annotated video: {out_video_path}")


if __name__ == "__main__":
    main()
