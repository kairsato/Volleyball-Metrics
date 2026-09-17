"""Scores ball_trajectory_smoother.py's output against real, human-verified
ground truth: the first 1000 frames of sideCutFixed(1000).json, a Label
Studio video-tracking export where the user hand-corrected a prelabeled
track. Only frames 1-1000 (Label Studio's 1-indexed "frame" field) are
trustworthy - the rest of that export is unverified leftover prelabel the
user hasn't gotten to yet, so this script deliberately ignores everything
past frame 1000.

A gap in the sequence (a frame number with no entry) or an explicit
enabled=false entry both mean "ball not visible" - Label Studio only
carries a row for frames where the tracked box is actually shown.

Usage (from Backend/Analysis/MachineLearning/):
    python evaluate_against_ground_truth.py [--trajectory PATH] [--gt PATH] [--max-frame N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ML_DIR / "hybrid_ball_test_results"

DEFAULT_GT = Path(r"C:\Users\Kai\Documents\Volleyball Footage\sideCutFixed(1000).json")
DEFAULT_TRAJECTORY = OUTPUT_DIR / "ball_trajectory_sideCutFixed_faststart.json"

# Ground truth's own header confirms the true match rate (verified frames
# only) - see the module docstring.
MAX_VERIFIED_FRAME = 1000  # Label Studio's 1-indexed "frame" field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--max-frame", type=int, default=MAX_VERIFIED_FRAME)
    return parser.parse_args()


def load_ground_truth(gt_path: Path, max_frame: int) -> tuple[dict[int, tuple[float, float] | None], int, int]:
    """Returns {frame_idx (0-indexed): (cx, cy) or None} for frame_idx in
    [0, max_frame - 1], plus (frame_w, frame_h) isn't known here - caller
    supplies it from the trajectory file, which shares the same source
    video. Also returns the raw frame_w/frame_h Label Studio doesn't tell us
    directly, so percentages are resolved by the caller instead."""
    raw = json.loads(gt_path.read_text(encoding="utf-8"))
    task = raw[0]
    result = task["annotations"][0]["result"][0]
    seq = result["value"]["sequence"]

    by_frame: dict[int, dict] = {}
    for entry in seq:
        f = entry["frame"]
        if f > max_frame:
            continue
        by_frame[f] = entry

    gt: dict[int, tuple[float, float] | None] = {}
    for f in range(1, max_frame + 1):
        entry = by_frame.get(f)
        idx = f - 1
        if entry is None or not entry["enabled"]:
            gt[idx] = None
        else:
            cx = entry["x"] + entry["width"] / 2.0
            cy = entry["y"] + entry["height"] / 2.0
            gt[idx] = (cx, cy)  # still in percent - converted to px by caller
    return gt


def main() -> None:
    args = parse_args()
    if not args.trajectory.exists():
        raise SystemExit(f"Trajectory not found: {args.trajectory} - run ball_trajectory_smoother.py first.")
    if not args.gt.exists():
        raise SystemExit(f"Ground truth not found: {args.gt}")

    traj_data = json.loads(args.trajectory.read_text(encoding="utf-8"))
    frame_w, frame_h = traj_data["frame_w"], traj_data["frame_h"]
    frames = traj_data["frames"]

    gt_percent = load_ground_truth(args.gt, args.max_frame)

    visible_count = sum(1 for v in gt_percent.values() if v is not None)
    hidden_count = sum(1 for v in gt_percent.values() if v is None)
    print(f"Ground truth: {len(gt_percent)} frames (1-{args.max_frame}) | visible: {visible_count} | not visible: {hidden_count}")

    diagonal = (frame_w**2 + frame_h**2) ** 0.5

    errors_px = []
    correct_visible = 0  # GT visible, prediction real/interpolated, within tolerance
    missed_visible = 0  # GT visible, prediction out_of_frame (missed entirely)
    wrong_visible = 0  # GT visible, prediction real/interpolated but far off
    false_positive = 0  # GT not visible, prediction claims real/interpolated
    correct_hidden = 0  # GT not visible, prediction out_of_frame

    TOLERANCE_PX = 0.03 * diagonal  # ~3% of frame diagonal - a generous "close enough" bar for a tiny fast object

    per_frame_rows = []
    for idx in sorted(gt_percent.keys()):
        if idx >= len(frames):
            continue
        gt_val = gt_percent[idx]
        pred = frames[idx]
        pred_status = pred["status"]

        if gt_val is not None:
            gx = gt_val[0] / 100.0 * frame_w
            gy = gt_val[1] / 100.0 * frame_h
            if pred_status == "out_of_frame":
                missed_visible += 1
                per_frame_rows.append((idx, "missed", None))
            else:
                box = pred["box"]
                px = (box[0] + box[2]) / 2.0
                py = (box[1] + box[3]) / 2.0
                err = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                errors_px.append(err)
                if err <= TOLERANCE_PX:
                    correct_visible += 1
                    per_frame_rows.append((idx, "correct", err))
                else:
                    wrong_visible += 1
                    per_frame_rows.append((idx, "wrong", err))
        else:
            if pred_status == "out_of_frame":
                correct_hidden += 1
            else:
                false_positive += 1
                per_frame_rows.append((idx, f"false_positive_{pred_status}", None))

    print()
    print("=== Frames where GT says the ball IS visible ===")
    print(f"  correct (within {TOLERANCE_PX:.1f}px / {TOLERANCE_PX / diagonal * 100:.1f}% of diagonal): {correct_visible}")
    print(f"  wrong (predicted somewhere else): {wrong_visible}")
    print(f"  missed (predicted out_of_frame):  {missed_visible}")
    if visible_count:
        print(f"  -> recall (correct / visible): {correct_visible / visible_count * 100:.1f}%")

    print()
    print("=== Frames where GT says the ball is NOT visible ===")
    print(f"  correctly out_of_frame: {correct_hidden}")
    print(f"  false positive (predicted a position anyway): {false_positive}")
    if hidden_count:
        print(f"  -> specificity (correct / not-visible): {correct_hidden / hidden_count * 100:.1f}%")

    if errors_px:
        errs = sorted(errors_px)
        n = len(errs)
        print()
        print(f"=== Position error (px), among frames GT says visible ===")
        print(f"  mean:   {sum(errs) / n:.2f}")
        print(f"  median: {errs[n // 2]:.2f}")
        print(f"  p90:    {errs[int(n * 0.9)]:.2f}")
        print(f"  max:    {errs[-1]:.2f}")

    # Worst offenders, to spot-check
    wrong_rows = [r for r in per_frame_rows if r[1] == "wrong"]
    wrong_rows.sort(key=lambda r: -(r[2] or 0))
    if wrong_rows:
        print()
        print("Worst 'wrong' frames (frame_idx, error_px):")
        for idx, _, err in wrong_rows[:15]:
            print(f"  frame {idx}: {err:.1f}px")

    missed_rows = [r[0] for r in per_frame_rows if r[1] == "missed"]
    if missed_rows:
        print()
        print(f"'missed' frame_idx values ({len(missed_rows)} total, first 30): {missed_rows[:30]}")

    fp_real = [r[0] for r in per_frame_rows if r[1] == "false_positive_real"]
    fp_interp = [r[0] for r in per_frame_rows if r[1] == "false_positive_interpolated"]
    if fp_real:
        print()
        print(f"false positive as 'real' ({len(fp_real)} total, first 30): {fp_real[:30]}")
    if fp_interp:
        print()
        print(f"false positive as 'interpolated' ({len(fp_interp)} total, first 30): {fp_interp[:30]}")


if __name__ == "__main__":
    main()
