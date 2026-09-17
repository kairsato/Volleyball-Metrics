"""
Renders short, single-concept GIFs for docs/assets/ - one per analysis
pipeline stage (ball detection, player detection, court detection, game
status detection) - by replaying an existing completed job's already-computed
JSON logs over its source video. It does NOT touch the production rendering
pipeline (PostProcessing/renderVideo.py's renderAnnotatedVideo, which bakes
every overlay into one combined video) - this is a separate, demo-only tool,
built by importing and reusing that module's per-overlay draw functions
directly rather than duplicating them.

Usage (from Backend/, so its venv can see cv2/numpy and Analysis/ is importable):
    .venv\\Scripts\\python.exe ..\\docs\\assets\\tools\\render_stage_demos.py --job b44b4bd3ab2d --stage all

Requires ffmpeg on PATH for the mp4 -> gif conversion step.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[3] / "Backend"
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for _directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import cv2  # noqa: E402

from PostProcessing.renderVideo import draw_ball, draw_players  # noqa: E402

DATA_DIR = BACKEND_DIR / "API" / "data"
OUT_DIR = Path(__file__).resolve().parents[1]  # docs/assets/

CLIP_SECONDS = 6
GIF_WIDTH = 420
GIF_FPS = 10

SERVE_DISPLAY_SECONDS = 1.5
PRE_ROLL_SECONDS = 2.0


def _load_json(path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _pick_rally(rallies, actions, need_serve):
    """Picks a rally with real duration to demo - prefers one with a serve
    action logged if the caller needs to demonstrate the serve state."""
    candidates = [r for r in rallies if r["end_frame"] - r["start_frame"] > 60]
    if need_serve:
        served = {a["rally_index"] for a in actions if a["action_type"] == "serve"}
        with_serve = [r for r in candidates if r["rally_index"] in served]
        if with_serve:
            return with_serve[len(with_serve) // 2]
    return candidates[len(candidates) // 2]


def _frame_range(rally, fps, pre_roll=0.0):
    start = max(0, int(rally["start_frame"] - pre_roll * fps))
    end = min(rally["end_frame"], start + int(CLIP_SECONDS * fps))
    return start, end


def _open_source(job_dir):
    video_path = job_dir / "input.mp4"
    if not video_path.exists():
        candidates = list(job_dir.glob("input.*"))
        if not candidates:
            raise FileNotFoundError(f"No input video found in {job_dir}")
        video_path = candidates[0]
    return video_path


def _write_clip(video_path, start_frame, end_frame, draw_frame, tmp_mp4):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(str(tmp_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = start_frame
    while frame_idx < end_frame:
        success, frame = cap.read()
        if not success:
            break
        draw_frame(frame, frame_idx)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return fps


def _blur_players(frame, players):
    """Gaussian-blurs the head/upper-body region of every tracked player box
    - these clips are cut from a real user's own footage, so faces need to
    stay unidentifiable in anything committed to docs/assets/, independent
    of whichever overlay (ball/court/status) is actually being demoed here."""
    height = frame.shape[0]
    for p in players:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        x1, x2 = max(0, min(x1, x2)), min(frame.shape[1], max(x1, x2))
        y1 = max(0, y1)
        head_bottom = min(height, y1 + max(24, int((min(height, y2) - y1) * 0.45)))
        if x2 <= x1 or head_bottom <= y1:
            continue
        region = frame[y1:head_bottom, x1:x2]
        if region.size == 0:
            continue
        k = max(15, (min(region.shape[0], region.shape[1]) // 2) * 2 + 1)
        frame[y1:head_bottom, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)


def _mp4_to_gif(mp4_path, gif_path):
    filters = f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", filters, "-loop", "0", str(gif_path)],
        check=True,
        capture_output=True,
    )


def render_ball_detection(job_dir, output_path, rallies, actions):
    ball_entries = _load_json(output_path / "ball_speed.json", [])
    ball_by_frame = {e["frame_idx"]: e for e in ball_entries}
    rally = _pick_rally(rallies, actions, need_serve=False)
    start, end = _frame_range(rally, fps=30.0)

    trail = []

    def draw_frame(frame, frame_idx):
        entry = ball_by_frame.get(frame_idx)
        current = None
        if entry is not None and entry["pixel"] is not None:
            trail.append(entry["pixel"])
            if len(trail) > 15:
                trail.pop(0)
            label = None
            if entry["real_units"] and entry["speed_m_per_s"] is not None:
                label = f"{entry['speed_m_per_s'] * 3.6:.1f} km/h"
            current = (entry["pixel"], label)
        draw_ball(frame, trail, current)

    return _render(job_dir, start, end, draw_frame, "demo-ball-detection.gif")


def render_player_detection(job_dir, output_path, rallies, actions):
    player_frames = _load_json(output_path / "player_positions.json", [])
    players_by_frame = {f["frame_idx"]: f["players"] for f in player_frames}
    rally = _pick_rally(rallies, actions, need_serve=False)
    start, end = _frame_range(rally, fps=30.0)

    def draw_frame(frame, frame_idx):
        players = players_by_frame.get(frame_idx, [])
        _blur_players(frame, players)
        draw_players(frame, players)

    return _render(job_dir, start, end, draw_frame, "demo-player-detection.gif")


def render_court_detection(job_dir, output_path, rallies, actions):
    court = _load_json(output_path / "court.json", None)
    rally = _pick_rally(rallies, actions, need_serve=False)
    start, end = _frame_range(rally, fps=30.0)

    def pt(name, points_key):
        p = court[points_key][name]
        return (int(p["x"]), int(p["y"]))

    corners = None
    net = None
    if court is not None:
        corners = [
            pt("far_left", "points"),
            pt("far_right", "points"),
            pt("middle_right", "points"),
            pt("middle_left", "points"),
        ]
        if court.get("net_top_calibrated"):
            net = (pt("net_top_left", "net_top_points"), pt("net_top_right", "net_top_points"))

    def draw_frame(frame, frame_idx):
        if corners is None:
            return
        for i in range(len(corners)):
            cv2.line(frame, corners[i], corners[(i + 1) % len(corners)], (0, 255, 255), 2)
        for c in corners:
            cv2.circle(frame, c, 6, (0, 255, 255), -1)
        if net is not None:
            cv2.line(frame, net[0], net[1], (255, 120, 0), 3)
            for p in net:
                cv2.circle(frame, p, 6, (255, 120, 0), -1)
        cv2.putText(frame, "COURT DETECTED", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return _render(job_dir, start, end, draw_frame, "demo-court-detection.gif")


def render_game_status_detection(job_dir, output_path, rallies, actions):
    """
    Approximates the game-status classifier's own PLAY / NO-PLAY / SERVE
    output using logs already produced downstream of it, rather than
    re-running the (heavy, torch-based) VideoMAE model just for a demo clip:
    PLAY/NO-PLAY comes from whether a frame falls inside a logged rally's
    [start_frame, end_frame] (exactly what that classifier's smoothed labels
    were used to derive - see GameStatusDetection.gameStatusDetection), and
    SERVE comes from actions.json's "serve" entries, which are themselves
    partly derived from a timing rule anchored to rally start (see
    PostProcessing.consolidate's CAVEATS) - close enough for a demo, not a
    re-run of the actual per-chunk classification.
    """
    rally = _pick_rally(rallies, actions, need_serve=True)
    start, end = _frame_range(rally, fps=30.0, pre_roll=PRE_ROLL_SECONDS)
    serve_frames = sorted(
        a["frame_idx"] for a in actions if a["rally_index"] == rally["rally_index"] and a["action_type"] == "serve"
    )

    def draw_frame(frame, frame_idx):
        in_rally = rally["start_frame"] <= frame_idx <= rally["end_frame"]
        near_serve = any(0 <= (frame_idx - sf) / 30.0 <= SERVE_DISPLAY_SECONDS for sf in serve_frames)
        if near_serve:
            label, colour = "SERVE", (0, 200, 255)
        elif in_rally:
            label, colour = "PLAY", (0, 255, 120)
        else:
            label, colour = "NO PLAY", (120, 120, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

    return _render(job_dir, start, end, draw_frame, "demo-game-status-detection.gif")


def _render(job_dir, start, end, draw_frame, gif_name):
    video_path = _open_source(job_dir)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_mp4 = Path(tmp) / "clip.mp4"
        _write_clip(video_path, start, end, draw_frame, tmp_mp4)
        gif_path = OUT_DIR / gif_name
        _mp4_to_gif(tmp_mp4, gif_path)
    size_kb = gif_path.stat().st_size / 1024
    print(f"Wrote {gif_path} ({size_kb:.0f} KB)")
    return gif_path


STAGES = {
    "ball": render_ball_detection,
    "player": render_player_detection,
    "court": render_court_detection,
    "status": render_game_status_detection,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Completed job id under Backend/API/data/")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = parser.parse_args()

    job_dir = DATA_DIR / args.job
    output_path = job_dir / "output"
    rallies = _load_json(output_path / "game_status.json", {"rallies": []})["rallies"]
    actions = _load_json(output_path / "actions.json", [])

    stages = STAGES.keys() if args.stage == "all" else [args.stage]
    for stage in stages:
        STAGES[stage](job_dir, output_path, rallies, actions)


if __name__ == "__main__":
    main()
