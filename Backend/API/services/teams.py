"""Best-effort team/side and rally-outcome inference for the Overall tab's
win/loss timeline and action radar chart.

None of this comes from a real scoreboard or whistle - the pipeline has no
such signal (see consolidate.py's CAVEATS about rally_ending_touches). What
it does have is a calibrated court (an 18m x 9m rectangle - see
CourtDetection/court.py - with the net crossing the middle of the 18m axis
at x=9) and a tracked ball position in those same real-world coordinates.
So the approach here is geometric rather than statistical: whichever half
of the court a player spends most of their tracked time on decides their
side, and each rally's winner is inferred from serving side rather than
where the ball ended up - rally-point scoring means whoever serves a rally
is always whoever won the previous one, so the side serving rally i+1 (read
from the ball's position soon after that rally starts, a clean signal since
a serve always originates from behind the server's own baseline) is
rally i's winner. This is more robust than watching where the ball last
landed, which can't tell a genuine unforced error apart from a kill shot
the other side simply couldn't reach, and degrades whenever tracking is
lost near the end of a rally rather than its start. The very last rally of
the video has no "next" rally to read a server from, so it falls back to
that same last-position method. Every number this module produces should be
read as "probably", not as ground truth, and callers should surface the
caveats below alongside it.

This also doubles as the "is this a sideline view" signal the Setup tab
wants: the court calibration workflow only ever marks a single court
rectangle with the net crossing its middle, which is only meaningful if the
camera can see (roughly) the whole 18m length - i.e. a sideline-ish angle.
Rather than classify the raw footage, we just check whether tracking
actually picked up a healthy number of players on *both* halves; if it
didn't (an end-on/behind-the-baseline shot would mostly track one team
well and barely see the other), team splitting is reported unavailable.
"""

import bisect
import json
from pathlib import Path
from typing import Optional

from .. import config
from ..resultcache import ttl_cache
from .players import build_canonical_mapping, load_ignored, load_names, load_player_positions

BALL_SPEED_NAME = "ball_speed.json"

# Where the net crosses the calibrated court's 18m length axis (half of
# CourtDetection.court.COURT_LENGTH) - left of this is side A, right is B.
NET_X_M = 9.0

# A player barely tracked at all (a couple of stray frames) shouldn't get a
# confident side - and a whole side with too few tracked players isn't
# trustworthy enough to build a two-team comparison out of. Kept low on
# purpose: this heuristic is meant to give its best guess rather than fall
# back to "unavailable" whenever tracking is merely imperfect - a human
# reviews/corrects every automatic result downstream anyway (see score.py).
MIN_SAMPLES_PER_PLAYER = 5
MIN_PLAYERS_PER_SIDE = 1

# "hit" is consolidate.py's catch-all bucket for anything the action-type
# heuristic isn't confident about, not a real technique - it doesn't belong
# on a skills radar next to actual actions like spike/set/dig/block/serve.
ACTION_TYPES = ["serve", "spike", "set", "dig", "block"]

CAVEATS = [
    "Teams are guessed from which half of the court each player spends most "
    "of their tracked time on, using the calibrated court geometry - not a "
    "jersey color, roster, or anything the pipeline was told directly.",

    "The winner of each rally is guessed from which side serves the next "
    "rally (rally-point scoring means the previous rally's winner always "
    "serves next), read from where the ball was tracked soon after that "
    "next rally starts - since there's no real scoreboard or whistle signal "
    "available anywhere in this pipeline. This is less reliable whenever "
    "ball tracking is lost near the start of a rally, and the last rally of "
    "the video (no next rally to read a server from) falls back to "
    "whichever side the ball was last tracked on before it ended, which "
    "can't tell a genuine unforced error apart from a kill shot the other "
    "side simply couldn't reach.",
]


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def assign_teams(output_path: Path, frame_range: Optional[tuple[int, int]] = None) -> dict[int, dict]:
    """canonical stable_id -> {"team": "A"|"B", "avg_court_x": float, "samples": int},
    for every named-or-not player with enough tracked positions to trust.
    Ignored players (ref, coach, false detections) are left out entirely.

    frame_range, if given, restricts the aggregation to
    [start_frame, end_frame] inclusive - used by score.py to work out each
    side's occupant one game/set at a time, since teams can swap sides
    between sets and a whole-video average would blur the two together.

    Called once per set by compute_team_stats and once per job by
    compute_player_radar/build_matchup - load_player_positions (not a raw
    _load_json here) is what keeps that from re-parsing this job's
    (100-250+ MB) player_positions.json from scratch every single time."""
    positions = load_player_positions(output_path)
    if not positions:
        return {}

    if frame_range is not None:
        start_frame, end_frame = frame_range
        positions = [f for f in positions if start_frame <= f["frame_idx"] <= end_frame]

    mapping = build_canonical_mapping(load_names(output_path))
    ignored = load_ignored(output_path)

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for frame in positions:
        for p in frame.get("players", []):
            # Which side of the net someone stands on is a question about
            # their position on the GROUND, so this wants court_ground (the
            # foot point) rather than `court` (the box centre, which the
            # ground-plane homography reads as further from the camera than
            # the player really is). That bias runs along the court's length
            # on a camera aimed down it - exactly the axis this compares
            # against NET_X_M - and on a real back-view match it put 15 of
            # 41 players on the wrong team. Falls back to `court` for
            # position logs written before court_ground existed.
            court = p.get("court_ground") or p.get("court")
            if not court:
                continue
            stable_id = mapping.get(p["stable_id"], p["stable_id"])
            if stable_id in ignored:
                continue
            sums[stable_id] = sums.get(stable_id, 0.0) + court[0]
            counts[stable_id] = counts.get(stable_id, 0) + 1

    teams: dict[int, dict] = {}
    for stable_id, total in sums.items():
        if counts[stable_id] < MIN_SAMPLES_PER_PLAYER:
            continue
        avg_x = total / counts[stable_id]
        teams[stable_id] = {
            "team": "A" if avg_x < NET_X_M else "B",
            "avg_court_x": avg_x,
            "samples": counts[stable_id],
        }
    return teams


def _other_side(side: str) -> str:
    return "B" if side == "A" else "A"


def infer_rally_winners(output_path: Path, invert_side: bool = False) -> dict[int, Optional[str]]:
    """rally_index -> winning team ("A"/"B"), or None when neither the
    serving-side nor the last-position fallback (see module docstring)
    could be read for that rally.

    invert_side flips which geometric half (x < NET_X_M vs x >= NET_X_M)
    counts as "A" vs "B" wherever a raw ball position is turned into a side
    here - for callers whose own side-to-team mapping runs backwards from
    what this heuristic assumes."""
    status = _load_json(output_path / config.GAME_STATUS_FILE_NAME)
    ball_frames = _load_json(output_path / BALL_SPEED_NAME)
    if not status or not ball_frames:
        return {}

    ball_x_by_frame = {b["frame_idx"]: b["court"][0] for b in ball_frames if b.get("court")}
    frame_list = sorted(ball_x_by_frame)

    def _side_of(court_x: float) -> str:
        side = "A" if court_x < NET_X_M else "B"
        return _other_side(side) if invert_side else side

    def _nearest_ball_frame(target_frame: int) -> Optional[int]:
        """Closest known ball position to target_frame, in either
        direction - deliberately not bounded to the rally the caller is
        asking about. If tracking happens to be missing right at a rally's
        boundary, a sample from just outside it is still a far better guess
        than giving up and leaving the rally unattributed, since every
        automatic result is reviewable/correctable downstream anyway (see
        score.py)."""
        idx = bisect.bisect_left(frame_list, target_frame)
        before = frame_list[idx - 1] if idx > 0 else None
        after = frame_list[idx] if idx < len(frame_list) else None
        candidates = [f for f in (before, after) if f is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda f: abs(f - target_frame))

    def _serving_side(rally: dict) -> Optional[str]:
        """The side serving this rally, read from the ball's position
        nearest to when it starts - a serve always originates from behind
        the server's own baseline, so this is a clean signal even though
        the rally itself may still be in progress."""
        frame = _nearest_ball_frame(rally["start_frame"])
        return _side_of(ball_x_by_frame[frame]) if frame is not None else None

    def _last_seen_losing_side(rally: dict) -> Optional[str]:
        """The old, less reliable signal: whichever side the ball was
        tracked on nearest to when the rally ended - used only as a
        fallback for the final rally of the video, which has no "next"
        rally to read a server from."""
        frame = _nearest_ball_frame(rally["end_frame"])
        return _side_of(ball_x_by_frame[frame]) if frame is not None else None

    rallies = status.get("rallies", [])
    winners: dict[int, Optional[str]] = {}
    for i, rally in enumerate(rallies):
        next_rally = rallies[i + 1] if i + 1 < len(rallies) else None
        winner = _serving_side(next_rally) if next_rally is not None else None
        if winner is None:
            # Falls back to the less reliable last-position signal not just
            # for the final rally (which has no next rally to read a server
            # from) but for any rally where the serve signal itself is
            # missing (e.g. ball tracking dropped right at the next rally's
            # start) - a bold-but-plausible guess beats leaving the rally
            # unattributed, since a human reviews/corrects this either way.
            losing_side = _last_seen_losing_side(rally)
            winner = _other_side(losing_side) if losing_side is not None else None
        winners[rally["rally_index"]] = winner

    return winners


@ttl_cache()
def build_matchup(output_path: Path) -> dict:
    """Everything the Overall tab's win/loss timeline, and the Setup tab's
    team/camera-view readout, need in one shot."""
    teams = assign_teams(output_path)

    side_counts = {"A": 0, "B": 0}
    for record in teams.values():
        side_counts[record["team"]] += 1

    if min(side_counts.values()) < MIN_PLAYERS_PER_SIDE:
        return {
            "available": False,
            "reason": (
                "Couldn't find enough well-tracked players on both sides of the "
                "net to split into two teams - either the video doesn't show "
                "both sides clearly (a behind-the-baseline or otherwise "
                "end-on angle rather than a sideline view), or court "
                "calibration is missing."
            ),
            "team_a": [],
            "team_b": [],
            "rallies": [],
            "wins_a": 0,
            "wins_b": 0,
            "radar": [],
            "caveats": [],
        }

    names = load_names(output_path)

    def _roster(side: str):
        return [
            {"stable_id": sid, "name": names.get(str(sid)), "avg_court_x": rec["avg_court_x"]}
            for sid, rec in sorted(teams.items())
            if rec["team"] == side
        ]

    team_a, team_b = _roster("A"), _roster("B")

    winners = infer_rally_winners(output_path)
    status = _load_json(output_path / config.GAME_STATUS_FILE_NAME) or {"rallies": []}

    rallies_out = []
    wins_a = wins_b = 0
    for rally in status["rallies"]:
        winner = winners.get(rally["rally_index"])
        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        rallies_out.append({
            "rally_index": rally["rally_index"],
            "start_time_s": rally["start_time_s"],
            "end_time_s": rally["end_time_s"],
            "winning_team": winner,
        })

    # Radar: per action type, each team's win rate in the rallies where one
    # of their players performed that action at least once - "when this
    # team did a spike, how often did they end up winning that point".
    actions_file = output_path / "actions_grouped.json"
    if not actions_file.exists():
        actions_file = output_path / "actions.json"
    actions = _load_json(actions_file) or []

    team_by_stable_id = {sid: rec["team"] for sid, rec in teams.items()}
    touched: dict[int, dict[str, set]] = {}
    for action in actions:
        rally_index = action.get("rally_index")
        stable_id = action.get("player_stable_id")
        if rally_index is None or stable_id is None:
            continue
        team = team_by_stable_id.get(stable_id)
        if team is None:
            continue
        action_type = action["action_type"] if action["action_type"] in ACTION_TYPES else "hit"
        touched.setdefault(rally_index, {"A": set(), "B": set()})[team].add(action_type)

    radar = []
    for action_type in ACTION_TYPES:
        per_team = {}
        for team in ("A", "B"):
            wins = total = 0
            for rally_index, by_team in touched.items():
                if action_type not in by_team[team]:
                    continue
                total += 1
                if winners.get(rally_index) == team:
                    wins += 1
            per_team[team] = (wins / total if total else None, total)

        radar.append({
            "action_type": action_type,
            "team_a_win_rate": per_team["A"][0],
            "team_b_win_rate": per_team["B"][0],
            "sample_size_a": per_team["A"][1],
            "sample_size_b": per_team["B"][1],
        })

    return {
        "available": True,
        "reason": None,
        "team_a": team_a,
        "team_b": team_b,
        "rallies": rallies_out,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "radar": radar,
        "caveats": CAVEATS,
    }
