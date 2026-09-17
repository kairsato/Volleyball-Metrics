"""Per-video score tracking: which named roster team (see team_roster.py)
won each rally, and how rallies group into sets. Distinct from both
teams.py's anonymous, whole-video "Team A/B" geometric split and
team_roster.py's persistent named groups - this module is what actually
attributes a geometric side to one of those named teams, one set at a
time, since sides can swap between sets.

Four methods, chosen per video in score_config.json:
  - "none": feature unused for this video.
  - "manual": every rally starts unassigned; a human fills in winners and
    set boundaries entirely by hand via the Setup tab's track editor.
  - "automatic": reuses teams.infer_rally_winners's serving-side heuristic
    (config's invert_side flips its raw A/B mapping), then maps its "A"/"B"
    output onto the two chosen named teams per set (config's
    match_alternating_sides fills in any set that mapping couldn't resolve
    on its own by alternating from the nearest set that could).
  - "ocr" (displayed as "Computer Vision" in the UI, since general OCR
    turned out to be the right tool after two pretrained digit-detection
    models both failed on real footage - see score_cv.py): reads a
    user-marked scoreboard region frame-by-frame near each rally's end and
    diffs the two numbers found there.

Whichever method populates score_result.json, it's always a starting point
a human can review and correct through the same track editor - the user
explicitly wants scrubbing/correction available even for the automatic
methods, since neither is ground truth.
"""

import json
from pathlib import Path
from typing import Optional

from .. import config
from . import teams
from .players import load_names

SCORE_CONFIG_NAME = "score_config.json"
SCORE_RESULT_NAME = "score_result.json"

METHODS = ("none", "manual", "automatic", "ocr")

DEFAULT_CONFIG = {
    "method": "none",
    "team_x_id": None,
    "team_y_id": None,
    "ocr_region": None,
    "cv_reverse_direction": False,
    # Only meaningful for method "automatic": teams.infer_rally_winners's
    # geometric side ("A" for court x < 9m, "B" otherwise) flipped once,
    # for when that raw mapping runs backwards for this video's camera/
    # calibration.
    "invert_side": False,
    # Also "automatic"-only, and on by default: attribute_sides_per_set
    # already re-derives, per set, which named team occupies which
    # geometric side from tracked player positions - which already accounts
    # for teams switching ends of the court between sets, since it's
    # measured fresh each time. But when a set doesn't have enough tracked,
    # named players to resolve that on its own, this fills the gap by
    # assuming sides simply alternate each set from the nearest set that
    # did resolve, rather than leaving every rally in it unattributed.
    "match_alternating_sides": True,
    # Minimum easyocr confidence (0-1) a digit-run detection needs before
    # score_cv counts it at all - a detection below this is treated the
    # same as not having read that side, rather than trusting a low-
    # confidence guess. Defaults to 0 (no filtering) so an existing video's
    # behavior doesn't silently change on upgrade; user-adjustable from the
    # Scoreboard Identification Region dialog.
    "ocr_min_confidence": 0.0,
    "compute_status": "idle",
    "compute_error": None,
    # Set once the user has explicitly signed off on the scoring shown in
    # the Setup tab's track editor - a UI-level lock (see the frontend's
    # ScoreSection/SetsList) that persists here so it survives a reload
    # and drives compute_summary's "needs review" flag below, rather than
    # being lost the moment the page unmounts.
    "confirmed": False,
}

# A side needs at least this many tracked positions from a team's known
# players before that side is trusted as "belonging" to that team for a
# given set. Kept at the lowest meaningful value (any signal beats none) so
# the automatic method commits to its best guess rather than punting to
# "uncertain" whenever a set only has a handful of named-player detections -
# a human reviews/corrects every automatic result downstream anyway. The
# same-team-on-both-sides check right below is what still catches a vote
# that's actually contradictory rather than just thin.
MIN_SIDE_ATTRIBUTION_SAMPLES = 1


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def load_config(output_path: Path) -> dict:
    saved = _load_json(output_path / SCORE_CONFIG_NAME)
    return {**DEFAULT_CONFIG, **(saved or {})}


def save_config(output_path: Path, config_update: dict) -> dict:
    current = load_config(output_path)
    current.update(config_update)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / SCORE_CONFIG_NAME).write_text(json.dumps(current, indent=2))
    return current


def load_result(output_path: Path) -> Optional[dict]:
    return _load_json(output_path / SCORE_RESULT_NAME)


def save_result(output_path: Path, result: dict) -> dict:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / SCORE_RESULT_NAME).write_text(json.dumps(result, indent=2))
    return result


def load_rallies(output_path: Path) -> list[dict]:
    status = _load_json(output_path / config.GAME_STATUS_FILE_NAME)
    return status.get("rallies", []) if status else []


def default_sets(rallies: list[dict]) -> dict:
    """One set containing every rally - the starting point before any
    splitting, and what "manual" starts from with every winner left null."""
    sets = [{"set_index": 0, "start_rally_index": 0, "end_rally_index": len(rallies) - 1}] if rallies else []
    return {
        "sets": sets,
        "rallies": [
            {"rally_index": r["rally_index"], "set_index": 0, "winner": None, "confidence": "manual"}
            for r in rallies
        ],
    }


def build_sets_from_boundaries(rallies: list[dict], split_after: set[int]) -> list[dict]:
    """split_after holds rally_indexes that end a set - the next rally (if
    any) starts a new one."""
    sets = []
    start = 0
    for i, rally in enumerate(rallies):
        if rally["rally_index"] in split_after or i == len(rallies) - 1:
            sets.append({"set_index": len(sets), "start_rally_index": start, "end_rally_index": i})
            start = i + 1
    return sets


def update_set_boundary(output_path: Path, rally_index: int, split: bool) -> dict:
    """Toggles a set split immediately after rally_index. Used directly by
    the manual track editor, and by score_cv.compute_cv to lay down boundaries it
    infers from the scoreboard resetting to 0-0."""
    rallies = load_rallies(output_path)
    result = load_result(output_path) or default_sets(rallies)

    # start/end_rally_index are rally_index values, not list positions - the
    # two always coincide since gameStatusDetection.py assigns rally_index
    # sequentially from 0, but keeping the naming explicit here avoids
    # relying on that silently. Every set's end except the last one is
    # currently a split point.
    split_after = {s["end_rally_index"] for s in result["sets"][:-1]}
    if split:
        split_after.add(rally_index)
    else:
        split_after.discard(rally_index)

    sets = build_sets_from_boundaries(rallies, split_after)

    set_by_rally = {}
    for match_set in sets:
        for idx in range(match_set["start_rally_index"], match_set["end_rally_index"] + 1):
            set_by_rally[rallies[idx]["rally_index"]] = match_set["set_index"]

    winners_by_rally = {r["rally_index"]: r for r in result["rallies"]}
    new_rallies = [
        {
            # Carries forward anything else already on the record (e.g.
            # score_cv.py's cv_left/cv_right reference digits) - only
            # set_index/winner/confidence actually need to change when a
            # boundary moves.
            **winners_by_rally.get(r["rally_index"], {}),
            "rally_index": r["rally_index"],
            "set_index": set_by_rally[r["rally_index"]],
            "winner": winners_by_rally.get(r["rally_index"], {}).get("winner"),
            "confidence": winners_by_rally.get(r["rally_index"], {}).get("confidence", "manual"),
        }
        for r in rallies
    ]

    return save_result(output_path, {"sets": sets, "rallies": new_rallies})


def set_rally_winner(output_path: Path, rally_index: int, winner: Optional[str]) -> dict:
    """winner is "x", "y", or None - a manual override, usable regardless of
    which method originally populated the result (corrections apply to
    automatic/ocr output too, not just "manual" mode)."""
    rallies = load_rallies(output_path)
    result = load_result(output_path) or default_sets(rallies)

    for r in result["rallies"]:
        if r["rally_index"] == rally_index:
            r["winner"] = winner
            r["confidence"] = "manual"
            break

    return save_result(output_path, result)


def reset_winners(output_path: Path) -> dict:
    """"Reset All Scores" - clears every rally's winner (winner=None)
    without touching set boundaries, regardless of which method (manual,
    automatic, ocr) originally populated them. confidence goes to
    "uncertain" rather than "manual": this is a deliberate "needs review"
    state, not an untouched-since-the-start one, and "uncertain" is what
    actually blocks Confirm Scoring (see SetsList's confirmDisabled on the
    frontend) - "manual" wouldn't, letting a reset be immediately
    re-confirmed with nothing actually reviewed. Starts from default_sets
    (one set containing every rally) if there's no result yet, the same
    fallback set_rally_winner/update_set_boundary already use."""
    rallies = load_rallies(output_path)
    result = load_result(output_path) or default_sets(rallies)

    for r in result["rallies"]:
        r["winner"] = None
        r["confidence"] = "uncertain"

    return save_result(output_path, result)


def _team_player_names(team_id: Optional[str]) -> set[str]:
    if team_id is None:
        return set()
    from . import team_roster

    for team in team_roster.load_teams():
        if team["id"] == team_id:
            return set(team["players"])
    return set()


def attribute_sides_per_set(output_path: Path, rallies: list[dict], sets: list[dict], team_x_id: Optional[str], team_y_id: Optional[str]) -> dict[int, dict[str, str]]:
    """set_index -> {"A": "x"|"y"|None, "B": "x"|"y"|None} - which named
    team occupies each geometric side, for each set separately. A side
    attributed by elimination (the other side resolved, so this one is
    whatever's left) still comes back "x"/"y" even with zero votes of its
    own - e.g. team_y_id left unset entirely. Both None means genuinely
    neither side could be placed (no tracked, named players from either
    identified team anywhere in that set's frame range)."""
    names = load_names(output_path)
    team_x_players = _team_player_names(team_x_id)
    team_y_players = _team_player_names(team_y_id)

    attribution: dict[int, dict[str, str]] = {}
    for match_set in sets:
        # start/end_rally_index are positions into `rallies` at the time this
        # set's boundaries were saved, not stable rally_index values - if the
        # job has since been reprocessed with a different rally segmentation
        # (game_status.json regenerated with fewer rallies than it had
        # then), a previously-saved score_result.json's end_rally_index can
        # point past the end of the current list. Rather than write the
        # whole set off, clamp to whatever positions still exist and
        # attribute from those - only genuinely nothing to work with (no
        # rallies at all, or even the set's start is past the end) gives up
        # entirely.
        unusable = not rallies or match_set["start_rally_index"] >= len(rallies)
        if unusable:
            attribution[match_set["set_index"]] = {"A": None, "B": None}
            continue

        end_rally_index = min(match_set["end_rally_index"], len(rallies) - 1)
        start_frame = rallies[match_set["start_rally_index"]]["start_frame"]
        end_frame = rallies[end_rally_index]["end_frame"]
        sides = teams.assign_teams(output_path, frame_range=(start_frame, end_frame))

        votes = {"A": {"x": 0, "y": 0}, "B": {"x": 0, "y": 0}}
        for stable_id, record in sides.items():
            player_name = names.get(str(stable_id))
            if not player_name:
                continue
            side = record["team"]
            if player_name in team_x_players:
                votes[side]["x"] += record["samples"]
            elif player_name in team_y_players:
                votes[side]["y"] += record["samples"]

        set_attribution = {}
        for side in ("A", "B"):
            x_votes, y_votes = votes[side]["x"], votes[side]["y"]
            if max(x_votes, y_votes) < MIN_SIDE_ATTRIBUTION_SAMPLES:
                set_attribution[side] = None
            else:
                set_attribution[side] = "x" if x_votes > y_votes else "y"

        # If both sides landed on the same team (or neither), the vote
        # wasn't clean enough to trust - leave both unattributed rather
        # than risk mislabeling every rally in the set.
        if set_attribution["A"] is not None and set_attribution["A"] == set_attribution["B"]:
            set_attribution = {"A": None, "B": None}

        # A court only has two sides - once votes have confidently placed
        # one of them, the other is whichever label is left over, even with
        # no votes of its own. This is what makes an unregistered second
        # team (team_y_id left null - there's no roster to ever vote "y"
        # in, so that side's own votes always stay 0) still resolve to "y"
        # rather than staying stuck "uncertain": the side isn't attributed
        # by identifying who's on it, but by elimination.
        resolved_sides = [side for side, label in set_attribution.items() if label is not None]
        if len(resolved_sides) == 1:
            known_side = resolved_sides[0]
            other_side = "B" if known_side == "A" else "A"
            set_attribution[other_side] = "y" if set_attribution[known_side] == "x" else "x"

        attribution[match_set["set_index"]] = set_attribution

    return attribution


def _fill_unattributed_sets_by_alternation(
    sets: list[dict], attribution: dict[int, dict[str, str]]
) -> dict[int, dict[str, str]]:
    """Sides switch ends of the court between sets in real volleyball -
    attribute_sides_per_set already captures that on its own (it measures
    tracked player positions fresh for each set), but a set with too few
    tracked, named players from either team can come back unattributed
    ({"A": None, "B": None}). This fills such a gap from the nearest set
    that *did* resolve, on the assumption that sides simply alternate every
    other set - better than leaving every rally in it unattributable."""
    resolved = {idx: attr for idx, attr in attribution.items() if attr.get("A") is not None}
    if not resolved:
        return attribution

    reference_index = min(resolved)
    reference = resolved[reference_index]

    filled = dict(attribution)
    for set_index in attribution:
        if attribution[set_index].get("A") is not None:
            continue
        flip = (set_index - reference_index) % 2 != 0
        filled[set_index] = {
            "A": reference["B" if flip else "A"],
            "B": reference["A" if flip else "B"],
        }
    return filled


def resolve_rally_range(
    output_path: Path, range_type: str, range_start: Optional[int], range_end: Optional[int]
) -> Optional[tuple[int, int]]:
    """Converts the Score tab's range picker (Match / Sets / Rallies, each
    with an inclusive 0-based start/end) into a (start_rally_index,
    end_rally_index) bound that compute_automatic/score_cv.compute_cv restrict
    themselves to - None means the whole match. "sets" is resolved
    against whatever sets currently exist in score_result.json, so
    re-running detection over "Sets 1-3" only touches the rallies that are
    actually in those sets right now."""
    if range_type == "match" or range_start is None or range_end is None:
        return None

    if range_type == "rallies":
        return (range_start, range_end)

    if range_type == "sets":
        result = load_result(output_path)
        sets = result["sets"] if result else []
        start_set = next((s for s in sets if s["set_index"] == range_start), None)
        end_set = next((s for s in sets if s["set_index"] == range_end), None)
        if start_set is None or end_set is None:
            return None
        return (start_set["start_rally_index"], end_set["end_rally_index"])

    return None


def reset_stuck_computations() -> int:
    """A background compute thread (see score_router._run_compute) can't
    survive a server restart - if the server was killed, crashed, or just
    reloaded (e.g. `--reload` picking up a code change) while one was
    running, the job it was working on is left with compute_status stuck at
    "computing" forever: nothing else ever calls save_config to move it
    past that, and the frontend only resumes polling for a job it kicked
    off the compute call for itself in that same page load - so the
    "Analyze" button just reads as permanently disabled with no way to
    recover short of manually editing score_config.json. Called once at
    startup (see main.py) to reset any such job back to "idle" (not
    "error" - there's nothing actionable to tell the user here, a dev
    server restarting mid-run is routine and not their problem to solve,
    so this recovers silently rather than surfacing a message every
    Analyze click has to be retried past). Returns how many jobs were
    reset, for the startup log."""
    if not config.DATA_DIR.exists():
        return 0

    reset_count = 0
    for job_dir in config.DATA_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        output_path = config.output_dir(job_dir.name)
        cfg_file = output_path / SCORE_CONFIG_NAME
        if not cfg_file.exists():
            continue
        try:
            data = json.loads(cfg_file.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("compute_status") == "computing":
            save_config(output_path, {"compute_status": "idle", "compute_error": None})
            reset_count += 1

    return reset_count


def compute_summary(output_path: Path) -> dict:
    """A lightweight, video-list-friendly summary of this video's scoring
    state - whether anything still needs a human's review, and, once a
    registered team's win is actually determinable, that team's name (plus
    the other team's name, when it's also registered - see loser_team_name
    below). Only reads score_config.json/score_result.json (and, if a
    winner is found, the small global team_roster.json) - no tracking data
    or video access, so it's cheap enough to compute for every video in a
    list rather than needing to be precomputed/cached anywhere.

    needs_review reflects whether the user has explicitly confirmed the
    scoring (config["confirmed"], set via the Setup tab's "Confirm
    Scoring"/"Redo Scoring" toggle) rather than directly re-deriving it
    from "uncertain" rallies - confirming is itself gated on there being no
    uncertain rallies left (see SetsList on the frontend), so "confirmed"
    is a strictly stronger signal: not just "nothing's uncertain right
    now" but "a human actually looked at this and signed off".

    A set's winner is whichever team won more rallies within it (mirrors
    the frontend's SetsList); the match winner is whichever team won more
    sets. Either comparison being tied resolves to "no winner yet" rather
    than guessing."""
    cfg = load_config(output_path)
    if cfg["method"] == "none":
        return {"needs_review": False, "winner_team_name": None, "loser_team_name": None}

    result = load_result(output_path)
    if result is None:
        return {"needs_review": False, "winner_team_name": None, "loser_team_name": None}

    needs_review = not cfg.get("confirmed", False)

    set_wins = {"x": 0, "y": 0}
    for match_set in result["sets"]:
        set_rallies = [r for r in result["rallies"] if r["set_index"] == match_set["set_index"]]
        wins_x = sum(1 for r in set_rallies if r["winner"] == "x")
        wins_y = sum(1 for r in set_rallies if r["winner"] == "y")
        if wins_x > wins_y:
            set_wins["x"] += 1
        elif wins_y > wins_x:
            set_wins["y"] += 1

    winner_team_id = None
    loser_team_id = None
    if set_wins["x"] > set_wins["y"]:
        winner_team_id, loser_team_id = cfg["team_x_id"], cfg["team_y_id"]
    elif set_wins["y"] > set_wins["x"]:
        winner_team_id, loser_team_id = cfg["team_y_id"], cfg["team_x_id"]

    winner_team_name = None
    loser_team_name = None
    if winner_team_id is not None:
        from . import team_roster

        teams_by_id = {team["id"]: team["name"] for team in team_roster.load_teams()}
        winner_team_name = teams_by_id.get(winner_team_id)
        # Only set once the *other* side is also a registered team - a None
        # here (scoring configured with just one named team) means "not
        # determinable", not "no loser", so the frontend's win/loss badge
        # and filters can tell the two apart rather than treating an
        # unregistered opponent as a same-named win.
        loser_team_name = teams_by_id.get(loser_team_id) if loser_team_id is not None else None

    return {"needs_review": needs_review, "winner_team_name": winner_team_name, "loser_team_name": loser_team_name}


def compute_automatic(output_path: Path, rally_range: Optional[tuple[int, int]] = None) -> dict:
    """rally_range, if given, restricts which rallies' winner/confidence get
    overwritten - everything else in the existing result (other rallies,
    and every set boundary) is left exactly as it was, so re-running
    Automatic over just "Sets 1-3" can't clobber sets 4+ that a human may
    have already reviewed and corrected."""
    cfg = load_config(output_path)
    rallies = load_rallies(output_path)
    existing = load_result(output_path)
    result = existing if existing is not None else default_sets(rallies)
    sets = result["sets"]

    ab_winners = teams.infer_rally_winners(output_path, invert_side=cfg.get("invert_side", False))
    attribution = attribute_sides_per_set(output_path, rallies, sets, cfg["team_x_id"], cfg["team_y_id"])
    if cfg.get("match_alternating_sides", True):
        attribution = _fill_unattributed_sets_by_alternation(sets, attribution)

    set_by_rally = {}
    for match_set in sets:
        # end_rally_index (and, in principle, start_rally_index) are
        # positions into `rallies` at the time this set was saved, not
        # stable rally_index values - see attribute_sides_per_set's own
        # staleness comment above. If the job has since been reprocessed
        # with fewer rallies than it had then, a stale set can reference
        # positions past the end of the current list; skip those rather
        # than index off the end of `rallies` and crash the whole compute
        # (the rallies inside a stale set just come back uncertain below,
        # same as attribute_sides_per_set already treats them).
        if match_set["start_rally_index"] >= len(rallies):
            continue
        end = min(match_set["end_rally_index"], len(rallies) - 1)
        for idx in range(match_set["start_rally_index"], end + 1):
            set_by_rally[rallies[idx]["rally_index"]] = match_set["set_index"]

    for r in result["rallies"]:
        if rally_range is not None and not (rally_range[0] <= r["rally_index"] <= rally_range[1]):
            continue
        ab_winner = ab_winners.get(r["rally_index"])
        set_index = set_by_rally.get(r["rally_index"])
        set_attribution = attribution.get(set_index, {}) if set_index is not None else {}
        r["winner"] = set_attribution.get(ab_winner) if ab_winner else None
        r["confidence"] = "auto" if r["winner"] is not None else "uncertain"

    return save_result(output_path, result)
