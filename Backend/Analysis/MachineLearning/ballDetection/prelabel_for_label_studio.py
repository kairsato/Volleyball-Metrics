"""Pre-labels a video (--video, default DEFAULT_VIDEO) for Label Studio's
video object-tracking interface, using hybrid_ball_detector's combined
primary/backup + filtered + tracker-gated pipeline (see that module's own
docstring) as the starting point a human then corrects in Label Studio,
rather than labeling the whole clip from scratch.

Every frame from the first detection onward gets a box: a real accepted
detection where the pipeline found one that frame, and the tracker's own
coasted (Kalman-extrapolated) position - inflated to the video's own
calibrated typical ball size, see calibrate_min_size - on a frame it
didn't. Frames before the very first detection anywhere in the video are
left out of the sequence entirely (nothing to extrapolate from yet); Label
Studio's timeline just shows no box until the first keyframe, same as any
track that starts partway through.

Output is a single Label Studio task per video (this is one continuous
video, not one task per frame) with a "predictions" entry carrying a
VideoRectangle sequence - Label Studio's own video-object-tracking format,
written to label_studio_prelabel_<video stem>.json so different source
clips don't overwrite each other's exports. A sidecar JSON
(label_studio_prelabel_<video stem>_sources.json) also records, per frame,
whether its box was a real detection (and from which model) or a coasted
guess - Label Studio's UI itself has no way to show that distinction, but
it's useful for later scoring different detection methods against whatever
ground truth comes back out of Label Studio (a coasted-guess frame isn't a
fair "the model found this" data point the way an actual detection is).

IMPORTANT: the from_name/to_name/data-key values below ("box"/"video"/
"video") match Label Studio's own stock "Video Object Detection" project
template. If your project's labeling config uses different control-tag
names, those three values (see LS_FROM_NAME/LS_TO_NAME/LS_DATA_KEY) are the
first thing to check if the pre-annotation doesn't attach when imported -
try importing just this one task first before assuming the per-frame box
data itself is wrong.

Usage (from Backend/Analysis/MachineLearning/):
    python prelabel_for_label_studio.py [--video PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

# hybrid_ball_detector must be imported first - it's the one that puts
# Backend/Analysis/ on sys.path as an import-time side effect, so
# BallTracker is re-exported from it here rather than imported from
# BallDetection.ballDetection directly (see hybrid_ball_test.py's own note).
from hybrid_ball_detector import (  # noqa: E402
    ML_DIR,
    SIZE_FRACTION_OF_MEDIAN,
    TILE_GRID,
    TILE_OVERLAP_FRACTION,
    BallTracker,
    calibrate_min_size,
    generate_tiles,
    load_models,
    log,
    run_hybrid_detection,
    video_info,
)

# Used when --video isn't given. Remuxed with `-movflags +faststart` (moov
# atom moved to the front of the file, lossless) from the original
# "verycut - Copy.mp4" - the original has no faststart at all (moov sits
# after a 373MB mdat, right at the end of the file), which is exactly what
# made Label Studio's video player fail to render it ("please check the
# format is supported") even though the actual codec (H.264/AAC) was fine
# all along - most browser-based video players, Label Studio's included,
# need the moov atom near the front to start decoding/seeking at all. Any
# --video pointed at a non-faststart file will hit the same rendering
# failure in Label Studio even though detection itself runs fine on it -
# check with ffprobe / remux the same way first.
DEFAULT_VIDEO = Path(r"C:\Users\Kai\Documents\VolleyballArea\verycut - Copy - faststart.mp4")
OUTPUT_DIR = ML_DIR / "hybrid_ball_test_results"

# See this file's own docstring - adjust these three to match your actual
# Label Studio labeling config if the stock template names don't apply.
LS_DATA_KEY = "video"
LS_FROM_NAME = "box"
LS_TO_NAME = "video"
LS_LABEL = "ball"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Video to pre-label (default: %(default)s). Must already be faststart - see this file's own docstring.",
    )
    return parser.parse_args()


def box_to_ls_percent(box, frame_w: float, frame_h: float) -> dict:
    x1, y1, x2, y2 = box
    return {
        "x": max(0.0, x1) / frame_w * 100.0,
        "y": max(0.0, y1) / frame_h * 100.0,
        "width": (x2 - x1) / frame_w * 100.0,
        "height": (y2 - y1) / frame_h * 100.0,
    }


def main() -> None:
    args = parse_args()
    video = args.video
    # Per-video output names - two different source clips no longer
    # silently overwrite each other's exports.
    task_name = f"label_studio_prelabel_{video.stem}.json"
    sidecar_name = f"label_studio_prelabel_{video.stem}_sources.json"
    video_url = f"/data/local-files/?d={video.name}"

    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    primary, backup = load_models()
    fps, frame_w, frame_h, total_frames = video_info(video)
    tiles = generate_tiles(frame_w, frame_h, TILE_GRID, TILE_OVERLAP_FRACTION)

    min_diagonal = calibrate_min_size(video, primary, tiles)
    # calibrate_min_size returns SIZE_FRACTION_OF_MEDIAN * the video's own
    # median confident-detection diagonal - undoing that scaling here gives
    # back the median itself, used below as the assumed size for a coasted
    # (no real detection) frame's box, since the tracker only carries a
    # centre point, not a size, once it's extrapolating rather than seeing
    # an actual detection.
    assumed_diagonal = min_diagonal / SIZE_FRACTION_OF_MEDIAN if min_diagonal > 0 else 20.0
    assumed_half_side = assumed_diagonal / math.sqrt(2) / 2

    log(f"Pre-labeling {total_frames} frames ({frame_w}x{frame_h} @ {fps:.1f}fps)...")

    tracker = BallTracker(None, fps, frame_w, frame_h, None)
    sequence: list[dict] = []
    sources: list[dict] = []
    frames_real = 0
    frames_coasted = 0
    frames_skipped = 0

    for result in run_hybrid_detection(video, primary, backup, min_diagonal, tracker):
        frame_idx = result.frame_idx

        if result.matched is not None:
            box, conf = result.matched
            source = "backup_yolo11x" if result.used_backup else "primary_yolo26x"
            frames_real += 1
        elif tracker.is_tracking:
            # No accepted detection this frame, but the Kalman filter still
            # has a track to coast on - tracker.trail()'s last point is
            # exactly that coasted position (see BallTracker.observe).
            cx, cy = tracker.trail()[-1]
            box = (cx - assumed_half_side, cy - assumed_half_side, cx + assumed_half_side, cy + assumed_half_side)
            conf = None
            source = "coasted"
            frames_coasted += 1
        else:
            # Nothing detected yet anywhere in the video - no track to
            # coast from either, so there's genuinely nothing to pre-fill.
            frames_skipped += 1
            continue

        percent = box_to_ls_percent(box, frame_w, frame_h)
        sequence.append(
            {
                "frame": frame_idx + 1,  # Label Studio's video sequence is 1-indexed
                "enabled": True,
                "rotation": 0,
                "time": round(frame_idx / fps, 4),
                **{k: round(v, 3) for k, v in percent.items()},
            }
        )
        sources.append({"frame": frame_idx + 1, "source": source, "confidence": conf})

    duration_s = total_frames / fps

    task = [
        {
            "data": {LS_DATA_KEY: video_url},
            "predictions": [
                {
                    "model_version": "hybrid_ball_detector_v1",
                    "result": [
                        {
                            "id": "ball_track_1",
                            "type": "videorectangle",
                            "origin": "manual",
                            "to_name": LS_TO_NAME,
                            "from_name": LS_FROM_NAME,
                            "value": {
                                "framesCount": total_frames,
                                "duration": duration_s,
                                "sequence": sequence,
                                "labels": [LS_LABEL],
                            },
                        }
                    ],
                }
            ],
        }
    ]

    task_path = OUTPUT_DIR / task_name
    task_path.write_text(json.dumps(task, indent=2))

    sidecar_path = OUTPUT_DIR / sidecar_name
    sidecar_path.write_text(
        json.dumps(
            {
                "video": str(video),
                "fps": fps,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "total_frames": total_frames,
                "frames": sources,
            },
            indent=2,
        )
    )

    total_elapsed = time.perf_counter() - t_start
    log("Done.")
    log(
        f"Frames: {total_frames} total | real detections: {frames_real} | coasted guesses: {frames_coasted} "
        f"| skipped (no track yet): {frames_skipped}"
    )
    log(f"Assumed ball size for coasted frames: {assumed_diagonal:.1f}px diagonal")
    log(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)")
    log(f"Label Studio task: {task_path}")
    log(f"Sidecar (per-frame source/confidence, not for Label Studio): {sidecar_path}")
    log(
        f'Set data.{LS_DATA_KEY} above to whatever URL/path your Label Studio instance actually '
        f"needs to resolve {video.name} before importing - {video_url!r} is only a guess."
    )


if __name__ == "__main__":
    main()
