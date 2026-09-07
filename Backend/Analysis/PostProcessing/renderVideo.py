import json
from pathlib import Path

import cv2
import numpy as np

BALL_SPEED_LOG_NAME = "ball_speed.json"
PLAYER_POSITIONS_LOG_NAME = "player_positions.json"
ACTIONS_LOG_NAME = "actions.json"
GAME_STATUS_LOG_NAME = "game_status.json"
COURT_FILE_NAME = "court.json"
ANNOTATED_VIDEO_NAME = "analysis.mp4"

# Deliberately independent of CourtDefinition/PlayerDetection/BallDetection -
# this only ever draws what those stages already computed and logged, so it
# has no need for their (heavy: torch/ultralytics) imports just to render.

TRAIL_LENGTH = 15
BALL_MARKER_RADIUS = 6

# How long a detected action stays labelled on screen after the instant it
# happened - the contact itself is a single frame, which would be
# unreadable, so this holds the label up and fades it out.
ACTION_DISPLAY_SECONDS = 1.5

MINIMAP_WIDTH = 320
MINIMAP_HEIGHT = 160
MINIMAP_MARGIN_PX = 16
COURT_LENGTH_M = 18.0
COURT_WIDTH_M = 9.0


def _load_json(path, default):
    if not path.exists():
        print(f"No {path.name} found; rendering without it.")
        return default

    with open(path) as f:
        return json.load(f)


def identity_colour(stable_id):
    rng = np.random.default_rng(stable_id * 9781)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def draw_players(frame, players):
    for p in players:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        colour = identity_colour(p["stable_id"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"#{p['stable_id']}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def draw_ball(frame, trail, current):
    for i in range(1, len(trail)):
        fade = i / len(trail)
        colour = (0, int(140 + 100 * fade), int(255 * fade))
        cv2.line(frame, tuple(int(v) for v in trail[i - 1]), tuple(int(v) for v in trail[i]),
                  colour, 2)

    if current is None:
        return

    pixel, label = current
    centre = tuple(int(v) for v in pixel)

    cv2.circle(frame, centre, BALL_MARKER_RADIUS, (0, 200, 255), -1)
    cv2.circle(frame, centre, BALL_MARKER_RADIUS, (20, 20, 20), 1)

    if label:
        cv2.putText(frame, label, (centre[0] + 10, centre[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def draw_rally_status(frame, rally_index):
    label = f"RALLY {rally_index}" if rally_index is not None else "DEAD BALL"
    colour = (0, 255, 120) if rally_index is not None else (120, 120, 255)

    cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)


def draw_action_banner(frame, action, seconds_ago):
    fade = max(0.0, 1.0 - seconds_ago / ACTION_DISPLAY_SECONDS)
    alpha = 0.3 + 0.7 * fade

    width = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 40), (width, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    player_note = f"  (player #{action['player_stable_id']})" if action["player_stable_id"] is not None else ""
    label = f"{action['action_type'].upper()}{player_note}"

    cv2.putText(frame, label, (16, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 255), 2)


def draw_court_minimap(frame, court_trail):
    if not court_trail:
        return

    height, width = frame.shape[:2]
    x1 = width - MINIMAP_WIDTH - MINIMAP_MARGIN_PX
    y1 = MINIMAP_MARGIN_PX
    x2 = x1 + MINIMAP_WIDTH
    y2 = y1 + MINIMAP_HEIGHT

    if x1 < 0 or y2 > height:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def to_minimap(point):
        cx, cy = point
        px = x1 + int(np.clip(cx / COURT_LENGTH_M, 0.0, 1.0) * MINIMAP_WIDTH)
        py = y1 + int(np.clip(cy / COURT_WIDTH_M, 0.0, 1.0) * MINIMAP_HEIGHT)
        return px, py

    net_x = x1 + MINIMAP_WIDTH // 2
    cv2.line(frame, (net_x, y1), (net_x, y2), (200, 200, 200), 1)

    for offset_m in (-3.0, 3.0):
        attack_x = x1 + int(((COURT_LENGTH_M / 2 + offset_m) / COURT_LENGTH_M) * MINIMAP_WIDTH)
        cv2.line(frame, (attack_x, y1), (attack_x, y2), (110, 110, 110), 1)

    for i in range(1, len(court_trail)):
        fade = i / len(court_trail)
        colour = (0, int(140 + 100 * fade), int(255 * fade))
        cv2.line(frame, to_minimap(court_trail[i - 1]), to_minimap(court_trail[i]), colour, 2)

    cv2.circle(frame, to_minimap(court_trail[-1]), 4, (0, 255, 255), -1)
    cv2.putText(frame, "court map", (x1 + 6, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def renderAnnotatedVideo(video_path, output_path):
    """
    Renders one combined video overlaying everything the pipeline knows for
    each frame - ball position/trail, tracked players and their stable_id,
    the current rally/dead-ball status, a recently-detected action, and a
    top-down court minimap - by reading the JSON logs each earlier stage
    produced, not by re-running any detection. This replaces needing to
    look at ball.mp4/fused.mp4 separately.
    """
    output_path = Path(output_path)

    ball_entries = _load_json(output_path / BALL_SPEED_LOG_NAME, [])
    player_frames = _load_json(output_path / PLAYER_POSITIONS_LOG_NAME, [])
    actions = _load_json(output_path / ACTIONS_LOG_NAME, [])
    status = _load_json(output_path / GAME_STATUS_LOG_NAME, {"rallies": []})

    ball_by_frame = {e["frame_idx"]: e for e in ball_entries}
    players_by_frame = {f["frame_idx"]: f["players"] for f in player_frames}
    rallies = status.get("rallies", [])
    actions = sorted(actions, key=lambda a: a["frame_idx"])

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_file = str(output_path / ANNOTATED_VIDEO_NAME)
    writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                              (frame_width, frame_height))

    pixel_trail = []
    court_trail = []
    action_cursor = 0
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        ball_entry = ball_by_frame.get(frame_idx)
        current_ball = None

        # ball_speed.json is dense (one entry per frame - see
        # BallDetection.ballDetection.build_speed_log), so ball_entry itself
        # is basically never None; a frame the ball tracker couldn't place
        # at all still gets an entry, just with pixel/court left null - skip
        # those the same way a genuinely missing entry would be skipped.
        if ball_entry is not None and ball_entry["pixel"] is not None:
            pixel_trail.append(ball_entry["pixel"])
            if ball_entry["court"] is not None:
                court_trail.append(ball_entry["court"])

            if len(pixel_trail) > TRAIL_LENGTH:
                pixel_trail.pop(0)
            if len(court_trail) > TRAIL_LENGTH:
                court_trail.pop(0)

            if ball_entry["real_units"] and ball_entry["speed_m_per_s"] is not None:
                label = f"{ball_entry['speed_m_per_s'] * 3.6:.1f} km/h"
            elif ball_entry["speed_px_per_s"] is not None:
                label = f"{ball_entry['speed_px_per_s']:.0f} px/s"
            else:
                label = None

            current_ball = (ball_entry["pixel"], label)

        draw_ball(frame, pixel_trail, current_ball)
        draw_players(frame, players_by_frame.get(frame_idx, []))

        rally = next((r for r in rallies if r["start_frame"] <= frame_idx <= r["end_frame"]), None)
        draw_rally_status(frame, rally["rally_index"] if rally is not None else None)

        while action_cursor < len(actions) and actions[action_cursor]["frame_idx"] < frame_idx:
            action_cursor += 1

        if action_cursor > 0:
            last_action = actions[action_cursor - 1]
            seconds_ago = (frame_idx - last_action["frame_idx"]) / fps
            if 0 <= seconds_ago <= ACTION_DISPLAY_SECONDS:
                draw_action_banner(frame, last_action, seconds_ago)

        draw_court_minimap(frame, court_trail)

        writer.write(frame)

        if frame_idx % 500 == 0:
            print(f"Rendering frame {frame_idx}/{total_frames}...")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Annotated analysis video saved: {video_file}")
