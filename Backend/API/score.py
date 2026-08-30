"""Per-video score tracking: which named roster team (see team_roster.py)
won each rally, and how rallies group into games/sets. Distinct from both
teams.py's anonymous, whole-video "Team A/B" geometric split and
team_roster.py's persistent named groups - this module is what actually
attributes a geometric side to one of those named teams, one game at a
time, since sides can swap between sets.

Four methods, chosen per video in score_config.json:
  - "none": feature unused for this video.
  - "manual": every rally starts unassigned; a human fills in winners and
    game boundaries entirely by hand via the Setup tab's track editor.
  - "automatic": reuses teams.infer_rally_winners's ball-out-of-bounds
    heuristic, then maps its "A"/"B" output onto the two chosen named
    teams per game.
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

from . import config, teams
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
    "compute_status": "idle",
    "compute_error": None,
    # Set once the user has explicitly signed off on the scoring shown in
    # the Setup tab's track editor - a UI-level lock (see the frontend's
    # ScoreSection/GamesList) that persists here so it survives a reload
    # and drives compute_summary's "needs review" flag below, rather than
    # being lost the moment the page unmounts.
    "confirmed": False,
}

# A side needs at least this many tracked positions from a team's known
# players before that side is trusted as "belonging" to that team for a
# given game - a stray misidentified detection or two shouldn't be able to
# flip an entire game's attribution.
MIN_SIDE_ATTRIBUTION_SAMPLES = 5


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


def default_games(rallies: list[dict]) -> dict:
    """One game containing every rally - the starting point before any
    splitting, and what "manual" starts from with every winner left null."""
    games = [{"game_index": 0, "start_rally_index": 0, "end_rally_index": len(rallies) - 1}] if rallies else []
    return {
        "games": games,
        "rallies": [
            {"rally_index": r["rally_index"], "game_index": 0, "winner": None, "confidence": "manual"}
            for r in rallies
        ],
    }


def build_games_from_boundaries(rallies: list[dict], split_after: set[int]) -> list[dict]:
    """split_after holds rally_indexes that end a game - the next rally (if
    any) starts a new one."""
    games = []
    start = 0
    for i, rally in enumerate(rallies):
        if rally["rally_index"] in split_after or i == len(rallies) - 1:
            games.append({"game_index": len(games), "start_rally_index": start, "end_rally_index": i})
            start = i + 1
    return games


def set_game_boundary(output_path: Path, rally_index: int, split: bool) -> dict:
    """Toggles a game split immediately after rally_index. Used directly by
    the manual track editor, and by score_cv.compute_cv to lay down boundaries it
    infers from the scoreboard resetting to 0-0."""
    rallies = load_rallies(output_path)
    result = load_result(output_path) or default_games(rallies)

    # start/end_rally_index are rally_index values, not list positions - the
    # two always coincide since gameStatusDetection.py assigns rally_index
    # sequentially from 0, but keeping the naming explicit here avoids
    # relying on that silently. Every game's end except the last one is
    # currently a split point.
    split_after = {g["end_rally_index"] for g in result["games"][:-1]}
    if split:
        split_after.add(rally_index)
    else:
        split_after.discard(rally_index)

    games = build_games_from_boundaries(rallies, split_after)

    game_by_rally = {}
    for game in games:
        for idx in range(game["start_rally_index"], game["end_rally_index"] + 1):
            game_by_rally[rallies[idx]["rally_index"]] = game["game_index"]

    winners_by_rally = {r["rally_index"]: r for r in result["rallies"]}
    new_rallies = [
        {
            "rally_index": r["rally_index"],
            "game_index": game_by_rally[r["rally_index"]],
            "winner": winners_by_rally.get(r["rally_index"], {}).get("winner"),
            "confidence": winners_by_rally.get(r["rally_index"], {}).get("confidence", "manual"),
        }
        for r in rallies
    ]

    return save_result(output_path, {"games": games, "rallies": new_rallies})


def set_rally_winner(output_path: Path, rally_index: int, winner: Optional[str]) -> dict:
    """winner is "x", "y", or None - a manual override, usable regardless of
    which method originally populated the result (corrections apply to
    automatic/ocr output too, not just "manual" mode)."""
    rallies = load_rallies(output_path)
    result = load_result(output_path) or default_games(rallies)

    for r in result["rallies"]:
        if r["rally_index"] == rally_index:
            r["winner"] = winner
            r["confidence"] = "manual"
            break

    return save_result(output_path, result)


def reset_winners(output_path: Path) -> dict:
    """"Reset All Scores" - clears every rally's winner (winner=None)
    without touching game boundaries, regardless of which method (manual,
    automatic, ocr) originally populated them. confidence goes to
    "uncertain" rather than "manual": this is a deliberate "needs review"
    state, not an untouched-since-the-start one, and "uncertain" is what
    actually blocks Confirm Scoring (see GamesList's confirmDisabled on the
    frontend) - "manual" wouldn't, letting a reset be immediately
    re-confirmed with nothing actually reviewed. Starts from default_games
    (one game containing every rally) if there's no result yet, the same
    fallback set_rally_winner/set_game_boundary already use."""
    rallies = load_rallies(output_path)
    result = load_result(output_path) or default_games(rallies)

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


def attribute_sides_per_game(output_path: Path, rallies: list[dict], games: list[dict], team_x_id: Optional[str], team_y_id: Optional[str]) -> dict[int, dict[str, str]]:
    """game_index -> {"A": "x"|"y"|None, "B": "x"|"y"|None} - which named
    team occupies each geometric side, for each game separately. None means
    that side couldn't be confidently attributed (not enough tracked,
    named players from either identified team in that game's frame range)."""
    names = load_names(output_path)
    team_x_players = _team_player_names(team_x_id)
    team_y_players = _team_player_names(team_y_id)

    attribution: dict[int, dict[str, str]] = {}
    for game in games:
        idxs = range(game["start_rally_index"], game["end_rally_index"] + 1)
        if not idxs or not rallies:
            attribution[game["game_index"]] = {"A": None, "B": None}
            continue

        start_frame = rallies[game["start_rally_index"]]["start_frame"]
        end_frame = rallies[game["end_rally_index"]]["end_frame"]
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

        game_attribution = {}
        for side in ("A", "B"):
            x_votes, y_votes = votes[side]["x"], votes[side]["y"]
            if max(x_votes, y_votes) < MIN_SIDE_ATTRIBUTION_SAMPLES:
                game_attribution[side] = None
            else:
                game_attribution[side] = "x" if x_votes > y_votes else "y"

        # If both sides landed on the same team (or neither), the vote
        # wasn't clean enough to trust - leave both unattributed rather
        # than risk mislabeling every rally in the game.
        if game_attribution["A"] is not None and game_attribution["A"] == game_attribution["B"]:
            game_attribution = {"A": None, "B": None}

        attribution[game["game_index"]] = game_attribution

    return attribution


def resolve_rally_range(
    output_path: Path, range_type: str, range_start: Optional[int], range_end: Optional[int]
) -> Optional[tuple[int, int]]:
    """Converts the Score tab's range picker (Match / Games / Rallies, each
    with an inclusive 0-based start/end) into a (start_rally_index,
    end_rally_index) bound that compute_automatic/score_cv.compute_cv restrict
    themselves to - None means the whole match. "games" is resolved
    against whatever games currently exist in score_result.json, so
    re-running detection over "Games 1-3" only touches the rallies that are
    actually in those games right now."""
    if range_type == "match" or range_start is None or range_end is None:
        return None

    if range_type == "rallies":
        return (range_start, range_end)

    if range_type == "games":
        result = load_result(output_path)
        games = result["games"] if result else []
        start_game = next((g for g in games if g["game_index"] == range_start), None)
        end_game = next((g for g in games if g["game_index"] == range_end), None)
        if start_game is None or end_game is None:
            return None
        return (start_game["start_rally_index"], end_game["end_rally_index"])

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
    registered team's win is actually determinable, that team's name. Only
    reads score_config.json/score_result.json (and, if a winner is found,
    the small global team_roster.json) - no tracking data or video access,
    so it's cheap enough to compute for every video in a list rather than
    needing to be precomputed/cached anywhere.

    needs_review reflects whether the user has explicitly confirmed the
    scoring (config["confirmed"], set via the Setup tab's "Confirm
    Scoring"/"Redo Scoring" toggle) rather than directly re-deriving it
    from "uncertain" rallies - confirming is itself gated on there being no
    uncertain rallies left (see GamesList on the frontend), so "confirmed"
    is a strictly stronger signal: not just "nothing's uncertain right
    now" but "a human actually looked at this and signed off".

    A game's winner is whichever team won more rallies within it (mirrors
    the frontend's GamesList); the match winner is whichever team won more
    games. Either comparison being tied resolves to "no winner yet" rather
    than guessing."""
    cfg = load_config(output_path)
    if cfg["method"] == "none":
        return {"needs_review": False, "winner_team_name": None}

    result = load_result(output_path)
    if result is None:
        return {"needs_review": False, "winner_team_name": None}

    needs_review = not cfg.get("confirmed", False)

    game_wins = {"x": 0, "y": 0}
    for game in result["games"]:
        game_rallies = [r for r in result["rallies"] if r["game_index"] == game["game_index"]]
        wins_x = sum(1 for r in game_rallies if r["winner"] == "x")
        wins_y = sum(1 for r in game_rallies if r["winner"] == "y")
        if wins_x > wins_y:
            game_wins["x"] += 1
        elif wins_y > wins_x:
            game_wins["y"] += 1

    winner_team_id = None
    if game_wins["x"] > game_wins["y"]:
        winner_team_id = cfg["team_x_id"]
    elif game_wins["y"] > game_wins["x"]:
        winner_team_id = cfg["team_y_id"]

    winner_team_name = None
    if winner_team_id is not None:
        from . import team_roster

        for team in team_roster.load_teams():
            if team["id"] == winner_team_id:
                winner_team_name = team["name"]
                break

    return {"needs_review": needs_review, "winner_team_name": winner_team_name}


def compute_automatic(output_path: Path, rally_range: Optional[tuple[int, int]] = None) -> dict:
    """rally_range, if given, restricts which rallies' winner/confidence get
    overwritten - everything else in the existing result (other rallies,
    and every game boundary) is left exactly as it was, so re-running
    Automatic over just "Games 1-3" can't clobber games 4+ that a human may
    have already reviewed and corrected."""
    cfg = load_config(output_path)
    rallies = load_rallies(output_path)
    existing = load_result(output_path)
    result = existing if existing is not None else default_games(rallies)
    games = result["games"]

    ab_winners = teams.infer_rally_winners(output_path)
    attribution = attribute_sides_per_game(output_path, rallies, games, cfg["team_x_id"], cfg["team_y_id"])

    game_by_rally = {}
    for game in games:
        for idx in range(game["start_rally_index"], game["end_rally_index"] + 1):
            game_by_rally[rallies[idx]["rally_index"]] = game["game_index"]

    for r in result["rallies"]:
        if rally_range is not None and not (rally_range[0] <= r["rally_index"] <= rally_range[1]):
            continue
        ab_winner = ab_winners.get(r["rally_index"])
        game_attribution = attribution.get(game_by_rally[r["rally_index"]], {})
        r["winner"] = game_attribution.get(ab_winner) if ab_winner else None
        r["confidence"] = "auto" if r["winner"] is not None else "uncertain"

    return save_result(output_path, result)
