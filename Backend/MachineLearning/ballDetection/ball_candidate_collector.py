"""Phase 1 of the two-phase ball detection approach: pure evidence
collection, no tracking and no filtering at all - that's Phase 2's job
(ball_trajectory_smoother.py), working from the raw log this script
produces.

Runs BOTH fine-tuned models (yolo26x, yolo11x - see model_benchmark.py,
which trained both) on EVERY frame via the same tiled full-frame
acquisition search the production detector uses (detect_tiled/
generate_tiles from BallDetection.ballDetection), at a low confidence
floor, and records every candidate box either model produced - box,
confidence, and which model found it. Deliberately never picks a "winner"
per frame or tracks anything across frames: the whole point of splitting
this into two phases is that Phase 2 can look at the ENTIRE video's
candidates at once (including frames *after* the one it's reasoning
about) to reconstruct the true trajectory, reject false positives by
consensus, and interpolate through gaps - none of which a single forward
pass over the video can do as well. It also means Phase 2 can be re-run
and re-tuned in seconds against the same cached candidates, without ever
re-running model inference.

Running two full-frame tiled passes every frame (instead of one primary
pass + an occasional backup pass, like the old hybrid_ball_detector.py
approach) costs roughly 2x the runtime of a single-model pass - worth it
here since the entire value of Phase 2 depends on having both models'
opinions for every single frame, not just the frames one model missed.

Usage (from Backend/Analysis/MachineLearning/):
    python ball_candidate_collector.py [--video PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ANALYSIS_DIR = Path(__file__).resolve().parents[2]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from BallDetection.ballDetection import detect_tiled, generate_tiles  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ML_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ML_DIR / "hybrid_ball_test_results"

MODELS_DIR = ML_DIR.parent / "models"
YOLO26X_WEIGHTS = MODELS_DIR / "ballDetection_yolo26x_best.pt"
YOLO11X_WEIGHTS = MODELS_DIR / "ballDetection_yolo11x_best.pt"

DEFAULT_VIDEO = Path(r"C:\Users\Kai\Documents\Volleyball Footage\sideCutFixed_faststart.mp4")

TILE_GRID = (2, 2)
TILE_OVERLAP_FRACTION = 0.2
ACQUISITION_IMG_SIZE = 960
DEVICE = 0
# Low, deliberately - Phase 2 does the real false-positive rejection with
# the benefit of seeing the whole video at once; a high confidence cutoff
# here would throw away candidates Phase 2 could otherwise have used to
# corroborate (or correct) a low-confidence-but-correct detection.
CONF_THRESHOLD = 0.05


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Video to collect candidates from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    if not YOLO26X_WEIGHTS.exists() or not YOLO11X_WEIGHTS.exists():
        raise SystemExit("Fine-tuned yolo26x/yolo11x weights not found - run model_benchmark.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"ball_candidates_{video.stem}.json"

    log("Loading models (yolo26x, yolo11x)...")
    yolo26x = YOLO(str(YOLO26X_WEIGHTS))
    yolo11x = YOLO(str(YOLO11X_WEIGHTS))
    models = [("yolo26x", yolo26x), ("yolo11x", yolo11x)]

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = generate_tiles(frame_w, frame_h, TILE_GRID, TILE_OVERLAP_FRACTION)

    log(f"Collecting candidates from both models over {total_frames} frames ({frame_w}x{frame_h} @ {fps:.1f}fps)...")

    t_start = time.perf_counter()
    frames_out: list[dict] = []
    total_candidates = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            candidates = []
            for model_name, model in models:
                for box, conf in detect_tiled(model, frame, tiles, ACQUISITION_IMG_SIZE, CONF_THRESHOLD, DEVICE):
                    candidates.append(
                        {
                            "box": [round(float(v), 2) for v in box],
                            "conf": round(float(conf), 4),
                            "model": model_name,
                        }
                    )
            total_candidates += len(candidates)
            frames_out.append({"frame_idx": frame_idx, "candidates": candidates})

            frame_idx += 1
            if frame_idx % 200 == 0:
                log(f"  {frame_idx}/{total_frames} frames... ({total_candidates} candidates so far)")
    finally:
        cap.release()

    elapsed = time.perf_counter() - t_start
    payload = {
        "video": str(video),
        "fps": fps,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "total_frames": frame_idx,
        "conf_threshold": CONF_THRESHOLD,
        "models": [m[0] for m in models],
        "frames": frames_out,
    }
    out_path.write_text(json.dumps(payload))

    empty_frames = sum(1 for f in frames_out if not f["candidates"])
    log("Done.")
    log(
        f"Frames: {frame_idx} | total candidates: {total_candidates} "
        f"({total_candidates / frame_idx:.2f}/frame avg) | frames with zero candidates: {empty_frames}"
    )
    log(f"Time: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    log(f"Raw candidate log: {out_path}")


if __name__ == "__main__":
    main()
