"""Reconstructs one ball trajectory per frame from ball_candidate_collector.py's cached per-frame candidates.

Usage (from Backend/Analysis/MachineLearning/):
    python ball_trajectory_smoother.py [--candidates PATH] [--video PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

ML_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ML_DIR / "hybrid_ball_test_results"

STILL_CELL_PX = 6.0
STILL_WINDOW_FRAMES = 15
STILL_MIN_HITS = 10
AGREEMENT_RADIUS_FRAC = 0.03
MIN_SELECT_CONFIDENCE = 0.60
MIN_SIZE_SAMPLES = 30
SIZE_RANGE_STD_FACTOR = 1.5
MAX_INTERPOLATE_GAP_FRAMES = 20
INTERPOLATE_TANGENT_DAMPING = 0.6


def _cell_of(box) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return round((x1 + x2) / 2.0 / STILL_CELL_PX), round((y1 + y2) / 2.0 / STILL_CELL_PX)


def _find_still_cells(frames_raw: list[dict]) -> set[tuple[int, int]]:
    cell_frames: dict[tuple[int, int], list[int]] = {}
    for f in frames_raw:
        for cell in {_cell_of(c["box"]) for c in f["candidates"]}:
            cell_frames.setdefault(cell, []).append(f["frame_idx"])

    still = set()
    for cell, idxs in cell_frames.items():
        left = 0
        for right in range(len(idxs)):
            while idxs[right] - idxs[left] >= STILL_WINDOW_FRAMES:
                left += 1
            if right - left + 1 >= STILL_MIN_HITS:
                still.add(cell)
                break
    return still


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=None,
        help="Raw candidate JSON from ball_candidate_collector.py (default: inferred from --video's stem).",
    )
    parser.add_argument(
        "--video", type=Path,
        default=Path(r"C:\Users\Kai\Documents\Volleyball Footage\sideCutFixed_faststart.mp4"),
        help="Source video - only used to name the default --candidates path and to render the verification video.",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip rendering the annotated verification video (JSON only, faster).",
    )
    return parser.parse_args()


def centre(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def box_diagonal(box) -> float:
    x1, y1, x2, y2 = box
    return float(np.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1)))


def _combined_confidence(cands, centres, i, agreement_radius) -> float:
    agreeing = max(
        (cands[j]["conf"] for j in range(len(cands))
         if cands[j]["model"] != cands[i]["model"] and np.linalg.norm(centres[i] - centres[j]) <= agreement_radius),
        default=0.0,
    )
    return 1.0 - (1.0 - cands[i]["conf"]) * (1.0 - agreeing)


def _select_ball(filtered, confs_per_frame, size_range) -> list[dict]:
    chosen = []
    for cands, confs in zip(filtered, confs_per_frame):
        if not cands:
            chosen.append({"status": "out_of_frame", "box": None, "conf": None, "model": None})
            continue

        if size_range is None:
            best_i = max(range(len(cands)), key=lambda i: confs[i])
        else:
            low_w, high_w, low_h, high_h = size_range

            def deviation(i):
                x1, y1, x2, y2 = cands[i]["box"]
                return max(0.0, low_w - (x2 - x1), (x2 - x1) - high_w) + max(0.0, low_h - (y2 - y1), (y2 - y1) - high_h)

            best_i = min(range(len(cands)), key=lambda i: (deviation(i), -confs[i]))

        if confs[best_i] < MIN_SELECT_CONFIDENCE:
            chosen.append({"status": "out_of_frame", "box": None, "conf": None, "model": None})
            continue

        best = cands[best_i]
        chosen.append({"status": "real", "box": best["box"], "conf": confs[best_i], "model": best["model"]})
    return chosen


def _estimate_size_range(chosen: list[dict]):
    widths = [c["box"][2] - c["box"][0] for c in chosen if c["status"] == "real"]
    heights = [c["box"][3] - c["box"][1] for c in chosen if c["status"] == "real"]
    if len(widths) < MIN_SIZE_SAMPLES:
        return None

    widths, heights = np.array(widths), np.array(heights)
    w_mean, w_std = widths.mean(), widths.std()
    h_mean, h_std = heights.mean(), heights.std()
    low_w, high_w = max(0.0, w_mean - SIZE_RANGE_STD_FACTOR * w_std), w_mean + SIZE_RANGE_STD_FACTOR * w_std
    low_h, high_h = max(0.0, h_mean - SIZE_RANGE_STD_FACTOR * h_std), h_mean + SIZE_RANGE_STD_FACTOR * h_std
    log(f"Ball size profile: {w_mean:.1f}x{h_mean:.1f}px avg over {len(widths)} frames, "
        f"plausible range [{low_w:.1f}-{high_w:.1f}] x [{low_h:.1f}-{high_h:.1f}]px")
    return low_w, high_w, low_h, high_h


def _interpolate_gaps(chosen: list[dict]) -> list[dict]:
    n = len(chosen)
    zero_v = np.array([0.0, 0.0])
    i = 0
    while i < n:
        if chosen[i]["status"] != "out_of_frame":
            i += 1
            continue
        j = i
        while j < n and chosen[j]["status"] == "out_of_frame":
            j += 1
        gap_len = j - i
        left_ok = i > 0 and chosen[i - 1]["status"] == "real"
        right_ok = j < n and chosen[j]["status"] == "real"
        if gap_len <= MAX_INTERPOLATE_GAP_FRAMES and left_ok and right_ok:
            p0 = centre(chosen[i - 1]["box"])
            p1 = centre(chosen[j]["box"])
            box0, box1 = chosen[i - 1]["box"], chosen[j]["box"]

            v_in = zero_v
            if i - 2 >= 0 and chosen[i - 2]["box"] is not None:
                v_in = p0 - centre(chosen[i - 2]["box"])
            v_out = zero_v
            if j + 1 < n and chosen[j + 1]["box"] is not None:
                v_out = centre(chosen[j + 1]["box"]) - p1

            span = gap_len + 1
            m0 = v_in * span * INTERPOLATE_TANGENT_DAMPING
            m1 = v_out * span * INTERPOLATE_TANGENT_DAMPING

            for t, idx in enumerate(range(i, j), start=1):
                frac = t / span
                f2, f3 = frac * frac, frac * frac * frac
                h00 = 2 * f3 - 3 * f2 + 1
                h10 = f3 - 2 * f2 + frac
                h01 = -2 * f3 + 3 * f2
                h11 = f3 - f2
                pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

                w0, h0 = box0[2] - box0[0], box0[3] - box0[1]
                w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
                w = w0 + (w1 - w0) * frac
                h = h0 + (h1 - h0) * frac
                chosen[idx] = {
                    "status": "interpolated",
                    "box": [pos[0] - w / 2, pos[1] - h / 2, pos[0] + w / 2, pos[1] + h / 2],
                    "conf": None,
                    "model": None,
                }
        i = j
    return chosen


def post_processing(frames_raw: list[dict], frame_w: int, frame_h: int) -> list[dict]:
    diagonal = math.hypot(frame_w, frame_h)
    agreement_radius = AGREEMENT_RADIUS_FRAC * diagonal

    still_cells = _find_still_cells(frames_raw)
    filtered = [[c for c in f["candidates"] if _cell_of(c["box"]) not in still_cells] for f in frames_raw]
    centres_per_frame = [[centre(c["box"]) for c in cands] for cands in filtered]
    confs_per_frame = [
        [_combined_confidence(cands, centres, i, agreement_radius) for i in range(len(cands))]
        for cands, centres in zip(filtered, centres_per_frame)
    ]

    chosen = _select_ball(filtered, confs_per_frame, None)
    size_range = _estimate_size_range(chosen)
    if size_range is not None:
        chosen = _select_ball(filtered, confs_per_frame, size_range)

    return _interpolate_gaps(chosen)


PRIMARY_COLOR = (0, 255, 0)  # green - a real, chosen detection
INTERP_COLOR = (255, 200, 0)  # cyan-blue - interpolated through a short gap
OUT_TEXT_COLOR = (0, 0, 255)


def render_video(video_path: Path, chosen: list[dict], out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail: list[tuple[int, int]] = []
    try:
        for i, entry in enumerate(chosen):
            ok, frame = cap.read()
            if not ok:
                break
            if entry["status"] in ("real", "interpolated"):
                x1, y1, x2, y2 = (int(v) for v in entry["box"])
                color = PRIMARY_COLOR if entry["status"] == "real" else INTERP_COLOR
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trail.append((cx, cy))
                if len(trail) > 15:
                    trail.pop(0)
                label = entry["status"] if entry["status"] == "interpolated" else f"{entry['model']} {entry['conf']:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            else:
                trail.clear()
                cv2.putText(frame, "ball out of frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, OUT_TEXT_COLOR, 2, cv2.LINE_AA)
            for k in range(1, len(trail)):
                cv2.line(frame, trail[k - 1], trail[k], (0, 200, 255), 2)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def main() -> None:
    args = parse_args()
    candidates_path = args.candidates or (OUTPUT_DIR / f"ball_candidates_{args.video.stem}.json")
    if not candidates_path.exists():
        raise SystemExit(f"Candidate log not found: {candidates_path} - run ball_candidate_collector.py first.")

    log(f"Loading candidates from {candidates_path}...")
    data = json.loads(candidates_path.read_text())
    frames_raw = data["frames"]
    frame_w, frame_h = data["frame_w"], data["frame_h"]

    log(f"Solving global trajectory over {len(frames_raw)} frames...")
    t_start = time.perf_counter()
    chosen = post_processing(frames_raw, frame_w, frame_h)
    elapsed = time.perf_counter() - t_start

    n_real = sum(1 for c in chosen if c["status"] == "real")
    n_interp = sum(1 for c in chosen if c["status"] == "interpolated")
    n_out = sum(1 for c in chosen if c["status"] == "out_of_frame")
    log(f"Done in {elapsed:.1f}s. real: {n_real} | interpolated: {n_interp} | out_of_frame: {n_out}")

    out_json = OUTPUT_DIR / f"ball_trajectory_{args.video.stem}.json"
    out_json.write_text(json.dumps({"frame_w": frame_w, "frame_h": frame_h, "fps": data["fps"], "frames": chosen}, indent=2))
    log(f"Trajectory JSON: {out_json}")

    if not args.no_video:
        out_video = OUTPUT_DIR / f"ball_trajectory_{args.video.stem}.mp4"
        log(f"Rendering verification video to {out_video}...")
        render_video(args.video, chosen, out_video)
        log(f"Verification video: {out_video}")


if __name__ == "__main__":
    main()
