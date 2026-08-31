"""
Weighted quality scoring for the 4 core actions (serve, receive, set, spike),
computed on request from already-persisted stage outputs (actions.json,
player_positions.json, ball_speed.json, court.json, teams.assign_teams) -
not a new pipeline stage. This mirrors how teams.build_matchup/score.
compute_summary already work: cheap, re-computable any time without
touching the expensive detection stages, so retuning a weight or reference
value here never requires reprocessing a video.

Every action type's overall score is a weighted average of a handful of
0-1 "factors", combined via _weighted_score - a factor that isn't available
for a given instance (missing team data, no camera-pose height, etc.) simply
drops out and the remaining weights renormalize, rather than penalizing or
failing outright. See CAVEATS below for what these scores do and don't mean.
"""
import bisect
import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import teams
from .players import PLAYER_POSITIONS_NAME

ACTIONS_NAME = "actions.json"
GROUPED_ACTIONS_NAME = "actions_grouped.json"
BALL_SPEED_NAME = "ball_speed.json"

# Half of CourtDefinition.court.COURT_LENGTH - duplicated here rather than
# cross-imported, matching how teams.py/ballDetection.py each keep their own
# copy of court.json's schema/dimensions instead of reaching into the
# Analysis package.
NET_X_M = 9.0

# --- Weight defaults - one dict per action type, always summing to 1.0.
# Retuning these needs no reprocessing, just a different number here. ---

SERVE_WEIGHTS = {"speed": 0.35, "placement": 0.40, "trajectory_height": 0.25}
RECEIVE_WEIGHTS = {"placement": 0.35, "reaction": 0.30, "handling_speed": 0.35}
SET_WEIGHTS = {"placement": 0.40, "height": 0.25, "time_given": 0.35}
SPIKE_WEIGHTS = {"positioning": 0.25, "speed": 0.30, "placement": 0.25, "blockers": 0.20}

# --- Reference values each factor normalizes against (see the individual
# _score_* functions for how each is used) ---

SERVE_SPEED_REFERENCE_MS = 20.0
SERVE_PLACEMENT_REFERENCE_M = 3.0
SERVE_HEIGHT_REFERENCE_M = 3.5

RECEIVE_PLACEMENT_REFERENCE_M = 2.5
RECEIVE_REACTION_EXCESS_REFERENCE_S = 0.4
# Matches actionDetection.FAST_INCOMING_SPEED_MS - the same "hard hit" bar
# used to classify a dig in the first place doubles as the reference a
# well-controlled outgoing pass should stay well under.
RECEIVE_SPEED_REFERENCE_MS = 8.0

SET_PLACEMENT_REFERENCE_M = 2.5
SET_HEIGHT_IDEAL_M = 3.0
SET_HEIGHT_TOLERANCE_M = 2.0
SET_TIME_REFERENCE_S = 1.5

SPIKE_POSITION_REFERENCE_M = 3.0
SPIKE_SPEED_REFERENCE_MS = 22.0
SPIKE_PLACEMENT_REFERENCE_M = 3.0
SPIKE_BLOCKER_REFERENCE_COUNT = 2
BLOCKER_PROXIMITY_M = 1.5

# Mirrors actionDetection.PLAYER_SEARCH_FRAME_RADIUS - how far (frames) to
# look either side of an action's own frame for a player-positions reading,
# since player tracking and the action's own frame don't always land on
# exactly the same frame index.
PLAYER_SEARCH_FRAME_RADIUS = 5

CAVEATS = [
    "\"Receive\" here means action-detection's \"dig\" label (any touch "
    "responding to a fast incoming ball) - there's no separate serve-receive "
    "category in this pipeline's action classifier, and no reliable way to "
    "split \"received a serve\" from \"dug an attack\" from trajectory alone.",

    "Every score below is a heuristic built from ball/player position, "
    "speed, and timing - not a trained model, and not validated against real "
    "coaching judgment. Weights and reference values are defaults, not "
    "measured constants; read scores as directional, not authoritative.",

    "\"Ideal location\" for receive/set placement is learned per-team from "
    "that team's own set/spike contact locations so far, not a fixed "
    "textbook target - a team with very few recorded sets/spikes gets a "
    "noisier target to score against.",

    "\"Blockers encountered\" counts opposing players near the net at the "
    "moment of the spike - it is NOT real jump/block detection, just a "
    "position-based proxy. It credits attacking into a crowded net as "
    "harder, regardless of whether the block was actually beaten.",

    "Placement factors for serve/spike use the NEXT recorded touch's "
    "position as a stand-in for where the ball landed - an ace or an "
    "unreturned kill (no next touch at all) can't be scored this way.",

    "Team/opponent-relative factors (placement, blockers) are only "
    "available when court calibration supports team splitting - see "
    "teams.py's own caveats. Trajectory-height factors are only available "
    "when this job's calibration includes a solved camera pose (net-top "
    "points marked) - see CalibrationPointsOut.camera_pose_available.",
]


def _load_actions(output_path: Path) -> list[dict]:
    """Prefers actions_grouped.json (canonical stable_ids, ignored players'
    actions already dropped) over the raw actions.json, exactly the same
    preference teams.build_matchup uses - keeps player/team attribution
    consistent between the two."""
    grouped_file = output_path / GROUPED_ACTIONS_NAME
    path = grouped_file if grouped_file.exists() else output_path / ACTIONS_NAME
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _players_by_frame(output_path: Path) -> dict[int, list[dict]]:
    positions_file = output_path / PLAYER_POSITIONS_NAME
    if not positions_file.exists():
        return {}
    return {entry["frame_idx"]: entry["players"] for entry in json.loads(positions_file.read_text())}


def _players_near_frame(players_by_frame: dict[int, list[dict]], frame_idx: int) -> Optional[list[dict]]:
    for offset in range(PLAYER_SEARCH_FRAME_RADIUS + 1):
        for idx in {frame_idx - offset, frame_idx + offset}:
            if idx in players_by_frame:
                return players_by_frame[idx]
    return None


def _build_height_lookup(output_path: Path) -> tuple[list[int], list[float]]:
    """Sorted (frame_idx, height_m) arrays for every frame ball_speed.json
    has a real height reading at - used by _apex_height to find the peak
    height within a frame range via bisect, without an O(actions * frames)
    scan."""
    ball_file = output_path / BALL_SPEED_NAME
    if not ball_file.exists():
        return [], []
    entries = json.loads(ball_file.read_text())
    entries = sorted(
        (e for e in entries if e.get("height_m") is not None), key=lambda e: e["frame_idx"]
    )
    return [e["frame_idx"] for e in entries], [e["height_m"] for e in entries]


def _apex_height(start_frame: int, end_frame: int, frames: list[int], heights: list[float]) -> Optional[float]:
    if not frames or end_frame < start_frame:
        return None
    lo = bisect.bisect_left(frames, start_frame)
    hi = bisect.bisect_right(frames, end_frame)
    if lo >= hi:
        return None
    return max(heights[lo:hi])


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _nearest_distance(point, candidates: list) -> Optional[float]:
    if point is None or not candidates:
        return None
    p = np.array(point, dtype=float)
    return float(min(np.linalg.norm(np.array(c, dtype=float) - p) for c in candidates))


def _opposing_player_distance_factor(
    landing_action: Optional[dict],
    players_by_frame: dict[int, list[dict]],
    team: Optional[str],
    team_by_id: dict[int, str],
    reference_m: float,
) -> Optional[float]:
    """Distance from landing_action's ball_court to the nearest opposing-
    team player at that moment, normalized against reference_m (bigger gap
    -> closer to 1.0) - shared by serve/spike placement, which both reward
    the ball landing far from whoever had to handle it. Falls back to
    considering every tracked player (not just the opposing team's) when
    team data isn't available for this instance, rather than returning None
    outright - a weaker signal, but still meaningful."""
    if landing_action is None or landing_action.get("ball_court") is None:
        return None

    frame_players = _players_near_frame(players_by_frame, landing_action["frame_idx"])
    if not frame_players:
        return None

    opposing = ("B" if team == "A" else "A") if team in ("A", "B") else None
    candidates = [
        p["court"] for p in frame_players
        if p.get("court") is not None and (opposing is None or team_by_id.get(p["stable_id"]) == opposing)
    ]
    distance = _nearest_distance(landing_action["ball_court"], candidates)
    if distance is None:
        return None
    return _clamp01(distance / reference_m)


def _weighted_score(factors: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    """Weighted average of whichever factors are actually available,
    renormalized across just those - a missing factor (team data
    unavailable, no camera pose, etc.) drops out cleanly rather than
    zeroing or blocking the whole score. None if nothing is available."""
    total = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        value = factors.get(name)
        if value is None:
            continue
        total += value * weight
        total_weight += weight
    return (total / total_weight) if total_weight > 0 else None


def _instance(action: dict, team: Optional[str], factors: dict[str, Optional[float]], weights: dict[str, float]) -> dict:
    return {
        "frame_idx": action["frame_idx"],
        "timestamp_s": action["timestamp_s"],
        "rally_index": action.get("rally_index"),
        "player_stable_id": action.get("player_stable_id"),
        "team": team,
        "factors": factors,
        "overall_score": _weighted_score(factors, weights),
    }


def _score_serve(action, next_action, players_by_frame, team, team_by_id, frames, heights) -> dict:
    factors: dict[str, Optional[float]] = {}

    speed = action.get("speed_out_m_per_s")
    factors["speed"] = _clamp01(speed / SERVE_SPEED_REFERENCE_MS) if speed is not None else None

    # Bigger gap to the nearest receiving-team player = a harder serve to
    # handle - the server's own ball_court is the contact point, not the
    # landing point, so the NEXT touch's position stands in for "where it
    # landed" (see CAVEATS).
    factors["placement"] = _opposing_player_distance_factor(
        next_action, players_by_frame, team, team_by_id, SERVE_PLACEMENT_REFERENCE_M
    )

    end_frame = next_action["frame_idx"] if next_action is not None else action["frame_idx"]
    apex = _apex_height(action["frame_idx"], end_frame, frames, heights)
    factors["trajectory_height"] = _clamp01(1 - apex / SERVE_HEIGHT_REFERENCE_M) if apex is not None else None

    return _instance(action, team, factors, SERVE_WEIGHTS)


def _score_receive(action, prev_action, team, set_targets) -> dict:
    factors: dict[str, Optional[float]] = {}

    target = set_targets.get(team) if team in ("A", "B") else None
    if target is not None and action.get("ball_court") is not None:
        distance = float(np.linalg.norm(np.array(action["ball_court"]) - target))
        factors["placement"] = _clamp01(1 - distance / RECEIVE_PLACEMENT_REFERENCE_M)
    else:
        factors["placement"] = None

    # How much longer this touch took than the ball's own approach speed
    # alone would predict, given how far it actually travelled since the
    # previous touch - a receive that lands close to the physically
    # expected arrival time scores well; a much longer gap suggests
    # scrambling/poor positioning rather than a clean, prompt play.
    factors["reaction"] = None
    speed_in = action.get("speed_in_m_per_s")
    gap = action.get("time_since_prev_touch_s")
    if prev_action is not None and speed_in and gap is not None \
            and prev_action.get("ball_court") is not None and action.get("ball_court") is not None:
        travel_distance = float(np.linalg.norm(
            np.array(action["ball_court"]) - np.array(prev_action["ball_court"])
        ))
        expected_s = travel_distance / speed_in
        excess_s = max(0.0, gap - expected_s)
        factors["reaction"] = _clamp01(1 - excess_s / RECEIVE_REACTION_EXCESS_REFERENCE_S)

    # A controlled pass goes back out SLOWER than it came in - lower is
    # better here, unlike every other "speed" factor in this module.
    speed_out = action.get("speed_out_m_per_s")
    factors["handling_speed"] = (
        _clamp01(1 - speed_out / RECEIVE_SPEED_REFERENCE_MS) if speed_out is not None else None
    )

    return _instance(action, team, factors, RECEIVE_WEIGHTS)


def _score_set(action, team, spike_targets, next_action, frames, heights) -> dict:
    factors: dict[str, Optional[float]] = {}

    target = spike_targets.get(team) if team in ("A", "B") else None
    if target is not None and action.get("ball_court") is not None:
        distance = float(np.linalg.norm(np.array(action["ball_court"]) - target))
        factors["placement"] = _clamp01(1 - distance / SET_PLACEMENT_REFERENCE_M)
    else:
        factors["placement"] = None

    # Closeness to an ideal middle band, not "lower is better" - a set
    # that's too low OR too high both score worse than one comfortably
    # in between (see SET_HEIGHT_IDEAL_M/SET_HEIGHT_TOLERANCE_M).
    end_frame = next_action["frame_idx"] if next_action is not None else action["frame_idx"]
    apex = _apex_height(action["frame_idx"], end_frame, frames, heights)
    factors["height"] = (
        _clamp01(1 - abs(apex - SET_HEIGHT_IDEAL_M) / SET_HEIGHT_TOLERANCE_M) if apex is not None else None
    )

    # Literally "how much time the setter had" - more time before this
    # touch (since whatever preceded it) scores higher, per the plainest
    # reading of "time given to you to set".
    time_given = action.get("time_since_prev_touch_s")
    factors["time_given"] = _clamp01(time_given / SET_TIME_REFERENCE_S) if time_given is not None else None

    return _instance(action, team, factors, SET_WEIGHTS)


def _score_spike(action, next_action, players_by_frame, team, team_by_id) -> dict:
    factors: dict[str, Optional[float]] = {}

    court = action.get("ball_court")
    if court is not None:
        distance_from_net = abs(court[0] - NET_X_M)
        factors["positioning"] = _clamp01(1 - distance_from_net / SPIKE_POSITION_REFERENCE_M)
    else:
        factors["positioning"] = None

    speed = action.get("speed_out_m_per_s")
    factors["speed"] = _clamp01(speed / SPIKE_SPEED_REFERENCE_MS) if speed is not None else None

    factors["placement"] = _opposing_player_distance_factor(
        next_action, players_by_frame, team, team_by_id, SPIKE_PLACEMENT_REFERENCE_M
    )

    # Difficulty credit, not a "beat the block" success signal - see
    # CAVEATS. Counts opposing players near the net at the spike's own
    # frame (unlike placement above, which looks at the NEXT touch).
    factors["blockers"] = None
    if team in ("A", "B"):
        frame_players = _players_near_frame(players_by_frame, action["frame_idx"])
        if frame_players is not None:
            opposing = "B" if team == "A" else "A"
            blocker_count = sum(
                1 for p in frame_players
                if p.get("court") is not None and team_by_id.get(p["stable_id"]) == opposing
                and abs(p["court"][0] - NET_X_M) <= BLOCKER_PROXIMITY_M
            )
            factors["blockers"] = _clamp01(blocker_count / SPIKE_BLOCKER_REFERENCE_COUNT)

    return _instance(action, team, factors, SPIKE_WEIGHTS)


def _average(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def _summarize(instances: list[dict]) -> dict:
    by_player: dict[int, list[float]] = {}
    for inst in instances:
        if inst["player_stable_id"] is None or inst["overall_score"] is None:
            continue
        by_player.setdefault(inst["player_stable_id"], []).append(inst["overall_score"])

    players = [
        {"stable_id": stable_id, "average_score": _average(scores), "count": len(scores)}
        for stable_id, scores in sorted(by_player.items())
    ]

    return {
        "count": len(instances),
        "average_score": _average([inst["overall_score"] for inst in instances]),
        "players": players,
        "instances": sorted(instances, key=lambda inst: inst["frame_idx"]),
    }


def _team_mean_location(actions: list[dict], action_type: str, team_by_id: dict[int, str]) -> dict[str, Optional[np.ndarray]]:
    """canonical stable_id's team -> mean ball_court of that team's own
    `action_type` touches so far - the "learned target" receive/set
    placement scores against, rather than a fixed hand-picked zone."""
    sums = {"A": np.zeros(2), "B": np.zeros(2)}
    counts = {"A": 0, "B": 0}
    for action in actions:
        if action["action_type"] != action_type or action.get("ball_court") is None:
            continue
        team = team_by_id.get(action.get("player_stable_id"))
        if team not in ("A", "B"):
            continue
        sums[team] += np.array(action["ball_court"], dtype=float)
        counts[team] += 1

    return {team: (sums[team] / counts[team]) if counts[team] else None for team in ("A", "B")}


def compute_action_quality(job_id: str, output_path: Path) -> dict:
    actions = _load_actions(output_path)
    players_by_frame = _players_by_frame(output_path)
    team_records = teams.assign_teams(output_path)
    team_by_id = {stable_id: record["team"] for stable_id, record in team_records.items()}
    frames, heights = _build_height_lookup(output_path)

    ordered = sorted(actions, key=lambda a: a["frame_idx"])
    by_rally: dict[Optional[int], list[dict]] = {}
    for action in ordered:
        by_rally.setdefault(action.get("rally_index"), []).append(action)

    prev_by_id: dict[int, Optional[dict]] = {}
    next_by_id: dict[int, Optional[dict]] = {}
    for rally_actions in by_rally.values():
        for i, action in enumerate(rally_actions):
            prev_by_id[id(action)] = rally_actions[i - 1] if i > 0 else None
            next_by_id[id(action)] = rally_actions[i + 1] if i + 1 < len(rally_actions) else None

    set_targets = _team_mean_location(actions, "set", team_by_id)
    spike_targets = _team_mean_location(actions, "spike", team_by_id)

    serve_instances, receive_instances, set_instances, spike_instances = [], [], [], []

    for action in ordered:
        team = team_by_id.get(action.get("player_stable_id"))
        next_action = next_by_id.get(id(action))
        prev_action = prev_by_id.get(id(action))
        action_type = action["action_type"]

        if action_type == "serve":
            serve_instances.append(
                _score_serve(action, next_action, players_by_frame, team, team_by_id, frames, heights)
            )
        elif action_type == "dig":
            receive_instances.append(_score_receive(action, prev_action, team, set_targets))
        elif action_type == "set":
            set_instances.append(_score_set(action, team, spike_targets, next_action, frames, heights))
        elif action_type == "spike":
            spike_instances.append(_score_spike(action, next_action, players_by_frame, team, team_by_id))

    return {
        "job_id": job_id,
        "teams_available": bool(team_by_id),
        "height_available": bool(frames),
        "serve": {"weights": SERVE_WEIGHTS, **_summarize(serve_instances)},
        "receive": {"weights": RECEIVE_WEIGHTS, **_summarize(receive_instances)},
        "set": {"weights": SET_WEIGHTS, **_summarize(set_instances)},
        "spike": {"weights": SPIKE_WEIGHTS, **_summarize(spike_instances)},
        "caveats": CAVEATS,
    }
