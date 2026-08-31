import json
from pathlib import Path

import cv2
import numpy as np

from CourtDefinition.court import COURT_LENGTH

BALL_SPEED_LOG_NAME = "ball_speed.json"
PLAYER_POSITIONS_LOG_NAME = "player_positions.json"
GAME_STATUS_LOG_NAME = "game_status.json"
ACTIONS_LOG_NAME = "actions.json"

# A "hit" is detected as a sharp change in the ball's own tracked velocity -
# either a sudden change of direction (a deflection) or a sudden change of
# speed (an acceleration, e.g. a serve toss turning into a serve, or a soft
# set turning into a hard spike). Either condition alone is enough; a real
# contact usually shows both, but a near-vertical set-to-spike can be mostly
# a speed change with little direction change, and a block/dig can be mostly
# a direction change without much speed change.
HIT_ANGLE_THRESHOLD_DEG = 40.0
HIT_SPEED_RATIO_THRESHOLD = 1.6

# Velocity either side of a candidate contact needs at least this many
# frames of separation - most consecutive real detections are only 1 frame
# apart (~0.03s), and dividing ordinary box-position jitter by a window that
# small produces huge, meaningless apparent velocity swings. Widening the
# window averages that jitter out while still localising a genuine contact
# to within a fraction of a second.
MIN_VELOCITY_WINDOW_FRAMES = 5

# ...but a bigger gap than this means the ball was lost and reacquired,
# possibly on an unrelated part of the rally, and any "velocity change"
# computed across that gap would be meaningless in the other direction.
MAX_FRAME_GAP_FOR_VELOCITY = 15

# One real contact can trip the threshold on several consecutive candidate
# frames as the direction change plays out - anything this close to the
# previous kept hit is treated as the same contact, not a separate one.
MIN_HIT_SEPARATION_SECONDS = 0.3

# How far a tracked player can be from the ball at a hit and still be
# credited with it. Metres when both have a court-space position, otherwise
# a fraction of the frame diagonal in pixel space.
MAX_PLAYER_ATTRIBUTION_DISTANCE_M = 2.5
MAX_PLAYER_ATTRIBUTION_DISTANCE_FRACTION = 0.15
PLAYER_SEARCH_FRAME_RADIUS = 5  # nearest player-log frame within this many frames of the hit

# A hit within this long of its rally's start (see game_status.json) is
# always called a serve, rather than relying on the geometric heuristics
# below - the very first contact of a rally has no "incoming" ball to reason
# about anyway.
SERVE_WINDOW_SECONDS = 0.5

# "Near the net" for the spike/block heuristics, in court-length metres from
# the net line (COURT_LENGTH / 2). Only meaningful when court calibration
# exists; without it, spike/block are never guessed (see NOTE below).
NET_PROXIMITY_M = 2.5

# An incoming hit faster than this (the previous hit's outgoing speed)
# suggests this contact is receiving a hard-hit ball (a dig), rather than
# building an attack from a soft one (a set).
FAST_INCOMING_SPEED_MS = 8.0

# NOTE ON ACCURACY: action TYPE (spike/set/dig/block/serve) is a rough
# geometric heuristic, not a trained classifier - there's no labelled
# action-recognition dataset or model for this project yet (see the ball
# detection fine-tuning for what that would take). Hit TIMING and rough
# court position are much more trustworthy than the type label; expect
# real misclassification, especially for sets/digs which look similar
# from trajectory alone. "hit" is used whenever no rule confidently applies.


def _velocity(entry_a, entry_b, fps):
    dt = (entry_b["frame_idx"] - entry_a["frame_idx"]) / fps
    if dt <= 0:
        return None

    if entry_a["court"] is not None and entry_b["court"] is not None:
        a = np.array(entry_a["court"], dtype=float)
        b = np.array(entry_b["court"], dtype=float)
    else:
        a = np.array(entry_a["pixel"], dtype=float)
        b = np.array(entry_b["pixel"], dtype=float)

    return (b - a) / dt


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _neighbor_before(real, i):
    """Nearest earlier real entry at least MIN_VELOCITY_WINDOW_FRAMES before
    real[i], within MAX_FRAME_GAP_FOR_VELOCITY - or None if neither exists."""
    target_frame = real[i]["frame_idx"]

    for j in range(i - 1, -1, -1):
        gap = target_frame - real[j]["frame_idx"]
        if gap > MAX_FRAME_GAP_FOR_VELOCITY:
            return None
        if gap >= MIN_VELOCITY_WINDOW_FRAMES:
            return real[j]

    return None


def _neighbor_after(real, i):
    target_frame = real[i]["frame_idx"]

    for j in range(i + 1, len(real)):
        gap = real[j]["frame_idx"] - target_frame
        if gap > MAX_FRAME_GAP_FOR_VELOCITY:
            return None
        if gap >= MIN_VELOCITY_WINDOW_FRAMES:
            return real[j]

    return None


def _find_hits(ball_entries, fps):
    """
    Real (non-interpolated) ball detections only - an interpolated point is
    the Kalman filter's own smoothed guess, not a measurement, and using it
    here would mean detecting "hits" partly manufactured by the filter's own
    smoothing rather than anything that happened in the footage.
    """
    real = [e for e in ball_entries if not e["interpolated"]]
    hits = []

    for i in range(len(real)):
        curr = real[i]
        prev = _neighbor_before(real, i)
        nxt = _neighbor_after(real, i)

        if prev is None or nxt is None:
            continue

        v_in = _velocity(prev, curr, fps)
        v_out = _velocity(curr, nxt, fps)

        if v_in is None or v_out is None:
            continue

        speed_in, speed_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        angle = _angle_between(v_in, v_out)

        speed_ratio = (max(speed_in, speed_out) / speed_in) if speed_in > 1e-6 else float("inf")

        is_direction_change = angle is not None and angle >= HIT_ANGLE_THRESHOLD_DEG
        is_speed_change = speed_ratio >= HIT_SPEED_RATIO_THRESHOLD

        if is_direction_change or is_speed_change:
            # A hit's in/out speed is only trustworthy as a real-world m/s
            # reading when BOTH endpoints on that side had a court position
            # to compute it from - otherwise _velocity silently fell back to
            # pixel space, same "real_units" idea ball_speed.json's own
            # per-point readings already use.
            real_units_in = prev["court"] is not None and curr["court"] is not None
            real_units_out = curr["court"] is not None and nxt["court"] is not None
            hits.append({
                "frame_idx": curr["frame_idx"],
                "pixel": curr["pixel"],
                "court": curr["court"],
                "height_m": curr.get("height_m"),
                "speed_in_ms_or_pxs": float(speed_in),
                "speed_out_ms_or_pxs": float(speed_out),
                "speed_in_real_units": real_units_in,
                "speed_out_real_units": real_units_out,
                "angle_change_deg": angle
            })

    return _merge_nearby_hits(hits, fps)


def _merge_nearby_hits(hits, fps):
    """
    One real contact can show up as several consecutive candidate frames all
    crossing the threshold (the direction change plays out over a few noisy
    frames near the actual contact) - collapse anything within
    MIN_HIT_SEPARATION_SECONDS of the previous kept hit into a single event,
    keeping whichever candidate had the larger angle change as the best
    estimate of the actual contact frame.
    """
    if not hits:
        return hits

    min_gap_frames = MIN_HIT_SEPARATION_SECONDS * fps
    merged = [hits[0]]

    for hit in hits[1:]:
        last = merged[-1]

        if hit["frame_idx"] - last["frame_idx"] > min_gap_frames:
            merged.append(hit)
            continue

        if (hit["angle_change_deg"] or 0) > (last["angle_change_deg"] or 0):
            merged[-1] = hit

    return merged


def _rally_for_frame(rallies, frame_idx):
    for rally in rallies:
        if rally["start_frame"] <= frame_idx <= rally["end_frame"]:
            return rally

    return None


def _nearest_player(frame_players, ball_pixel, ball_court, diagonal):
    best_id, best_score = None, None

    for p in frame_players:
        if ball_court is not None and p["court"] is not None:
            distance = float(np.linalg.norm(np.array(p["court"]) - np.array(ball_court)))
            limit = MAX_PLAYER_ATTRIBUTION_DISTANCE_M
        else:
            distance = float(np.linalg.norm(np.array(p["pixel"]) - np.array(ball_pixel)))
            distance = distance / diagonal
            limit = MAX_PLAYER_ATTRIBUTION_DISTANCE_FRACTION

        if distance > limit:
            continue

        score = distance / limit
        if best_score is None or score < best_score:
            best_score = score
            best_id = p["stable_id"]

    return best_id


def _players_near_frame(players_by_frame, frame_idx):
    for offset in range(PLAYER_SEARCH_FRAME_RADIUS + 1):
        for idx in {frame_idx - offset, frame_idx + offset}:
            if idx in players_by_frame:
                return players_by_frame[idx]

    return None


def _classify(hit, prev_hit, rally, serve_window_frames):
    """
    Rough rule-based action-type guess - see the accuracy note at the top of
    this file. Rules are checked in order; the first that applies wins.
    """
    if rally is not None and hit["frame_idx"] - rally["start_frame"] <= serve_window_frames:
        return "serve"

    court = hit["court"]
    near_net = court is not None and abs(court[0] - COURT_LENGTH / 2) <= NET_PROXIMITY_M

    if near_net and hit["angle_change_deg"] is not None and hit["angle_change_deg"] >= 120:
        return "block"

    if near_net and hit["speed_out_ms_or_pxs"] > hit["speed_in_ms_or_pxs"]:
        return "spike"

    if prev_hit is not None and prev_hit["speed_out_ms_or_pxs"] >= FAST_INCOMING_SPEED_MS \
            and hit["speed_out_ms_or_pxs"] < prev_hit["speed_out_ms_or_pxs"]:
        return "dig"

    if hit["speed_out_ms_or_pxs"] < hit["speed_in_ms_or_pxs"]:
        return "set"

    return "hit"


def detectActions(video_path, output_path):

    ball_file = Path(output_path) / BALL_SPEED_LOG_NAME
    players_file = Path(output_path) / PLAYER_POSITIONS_LOG_NAME
    status_file = Path(output_path) / GAME_STATUS_LOG_NAME

    if not ball_file.exists():
        print(f"No ball speed log found at {ball_file}; run ball detection first.")
        return

    with open(ball_file) as f:
        ball_entries = sorted(json.load(f), key=lambda e: e["frame_idx"])

    players_by_frame = {}
    if players_file.exists():
        with open(players_file) as f:
            for frame in json.load(f):
                players_by_frame[frame["frame_idx"]] = frame["players"]
    else:
        print(f"No player positions log found at {players_file}; "
              f"actions will be logged without a player attribution.")

    rallies = []
    if status_file.exists():
        with open(status_file) as f:
            rallies = json.load(f)["rallies"]
    else:
        print(f"No game status log found at {status_file}; "
              f"serve detection and rally association will be skipped.")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    diagonal = float(np.hypot(frame_width, frame_height))
    serve_window_frames = max(1, round(SERVE_WINDOW_SECONDS * fps))

    hits = _find_hits(ball_entries, fps)

    actions = []
    prev_hit_in_rally = {}

    for hit in hits:

        rally = _rally_for_frame(rallies, hit["frame_idx"])
        rally_index = rally["rally_index"] if rally is not None else None

        prev_hit = prev_hit_in_rally.get(rally_index)
        action_type = _classify(hit, prev_hit, rally, serve_window_frames)
        prev_hit_in_rally[rally_index] = hit

        frame_players = _players_near_frame(players_by_frame, hit["frame_idx"]) or []
        player_id = _nearest_player(frame_players, hit["pixel"], hit["court"], diagonal)

        # A hit's in/out speed is only exposed in real units (m/s) when both
        # sides of it were - never a mix of one real reading and one pixel
        # reading masquerading as the same unit. See _find_hits above.
        real_units = hit["speed_in_real_units"] and hit["speed_out_real_units"]
        time_since_prev_touch_s = (
            (hit["frame_idx"] - prev_hit["frame_idx"]) / fps if prev_hit is not None else None
        )

        actions.append({
            "frame_idx": hit["frame_idx"],
            "timestamp_s": hit["frame_idx"] / fps,
            "rally_index": rally_index,
            "player_stable_id": player_id,
            "action_type": action_type,
            "ball_pixel": hit["pixel"],
            "ball_court": hit["court"],
            "ball_height_m": hit["height_m"],
            "speed_in_m_per_s": hit["speed_in_ms_or_pxs"] if real_units else None,
            "speed_out_m_per_s": hit["speed_out_ms_or_pxs"] if real_units else None,
            "real_units": real_units,
            "time_since_prev_touch_s": time_since_prev_touch_s,
        })

    actions_file = Path(output_path) / ACTIONS_LOG_NAME

    with open(actions_file, "w") as f:
        json.dump(actions, f, indent=2)

    print(f"Detected {len(actions)} hit event(s).")
    print(f"Actions saved: {actions_file}")
