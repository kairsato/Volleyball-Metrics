"""Cross-game aggregate stats for a named team (team_roster.py) - win/loss
record and a win-rate-by-action-type radar - plus the per-player analog
(PlayerStatsPage's radar). team_roster.py's own docstring is explicit that
a named team has "no bearing on any per-game analysis" by itself; the only
place a named team is ever linked to a geometric court side at all is
score.py's per-game Scoring config (team_x_id/team_y_id), so win/loss and
radar here only ever draw from games where that link exists. Hit-count/
action-mix totals are looser - they only need a name match against
player_stats.json, so every complete game contributes those regardless of
whether Scoring was ever configured for it.
"""
import json
from pathlib import Path
from typing import Optional

from .. import config, jobs
from ..resultcache import ttl_cache
from . import action_quality, heuristics, score, team_roster, teams
from .players import STATS_NAME as PLAYER_STATS_NAME
from .players import load_names

ACTION_TYPES = teams.ACTION_TYPES
QUALITY_CATEGORY_KEYS = ("serve", "receive", "set", "spike", "block")


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _actions_for_job(output_path: Path) -> list[dict]:
    """Prefers actions_grouped.json (canonical stable_ids, ignored players'
    actions already dropped) over raw actions.json - the same preference
    teams.build_matchup already uses, and required here since
    teams.assign_teams's stable_ids are canonical too."""
    grouped = output_path / "actions_grouped.json"
    path = grouped if grouped.exists() else output_path / "actions.json"
    return _load_json(path) or []


def _complete_jobs():
    return [job for job in jobs.store.list() if job.status == jobs.STATUS_COMPLETE]


def _team_or_none(team_id: str) -> Optional[dict]:
    for team in team_roster.load_teams():
        if team["id"] == team_id:
            return team
    return None


def _roster_stats(team_players: set[str]) -> tuple[list[dict], dict[str, int], int, int]:
    """Sums each roster member's total_hits/hits_by_type across every
    complete job's player_stats.json - matched by name (the same field
    PlayersPage's own client-side cross-game aggregation already keys on),
    not stable_id, since stable_ids aren't stable across games. Returns
    (per_player_summaries, team_hits_by_type, team_total_hits,
    games_with_a_team_player)."""
    per_player: dict[str, dict] = {name: {"total_hits": 0, "hits_by_type": {}} for name in team_players}
    hits_by_type_team: dict[str, int] = {}
    total_hits_team = 0
    games_total = 0

    for job in _complete_jobs():
        output_path = config.output_dir(job.id)
        stats = _load_json(output_path / PLAYER_STATS_NAME)
        if not stats:
            continue

        job_has_team_player = False
        for record in stats.get("players", {}).values():
            name = record.get("name")
            if not name or name not in team_players:
                continue
            job_has_team_player = True
            hits = record.get("total_hits", 0)
            per_player[name]["total_hits"] += hits
            total_hits_team += hits
            for action_type, count in record.get("hits_by_type", {}).items():
                per_player[name]["hits_by_type"][action_type] = (
                    per_player[name]["hits_by_type"].get(action_type, 0) + count
                )
                hits_by_type_team[action_type] = hits_by_type_team.get(action_type, 0) + count

        if job_has_team_player:
            games_total += 1

    players_out = [
        {"name": name, "total_hits": data["total_hits"], "hits_by_type": data["hits_by_type"]}
        for name, data in sorted(per_player.items())
    ]
    return players_out, hits_by_type_team, total_hits_team, games_total


def _accumulate_set_radar(
    output_path: Path,
    match_set: dict,
    rallies: list[dict],
    actions: list[dict],
    ab_winners: dict[int, Optional[str]],
    my_side: Optional[str],
    opp_side: Optional[str],
    radar: dict[str, dict[str, dict[str, int]]],
) -> None:
    """Mutates radar (`{"mine": {"wins": {...}, "totals": {...}}, "opp":
    {...}}`) with this set's contribution - my_side/opp_side are which
    geometric side ("A"/"B", or None if unattributed) this team/its
    opponent occupy for THIS set specifically (sides can swap between
    sets, hence per-set rather than a single whole-game split)."""
    if my_side is None and opp_side is None:
        return

    start_frame = rallies[match_set["start_rally_index"]]["start_frame"]
    end_frame = rallies[match_set["end_rally_index"]]["end_frame"]
    sides = teams.assign_teams(output_path, frame_range=(start_frame, end_frame))
    rally_indices = {
        rallies[i]["rally_index"] for i in range(match_set["start_rally_index"], match_set["end_rally_index"] + 1)
    }

    touched: dict[int, dict[str, set]] = {}
    for action in actions:
        rally_index = action.get("rally_index")
        stable_id = action.get("player_stable_id")
        if rally_index is None or stable_id is None or rally_index not in rally_indices:
            continue
        record = sides.get(stable_id)
        if record is None:
            continue
        geo_side = record["team"]
        role = "mine" if geo_side == my_side else "opp" if geo_side == opp_side else None
        if role is None:
            continue
        action_type = action["action_type"] if action["action_type"] in ACTION_TYPES else "hit"
        touched.setdefault(rally_index, {"mine": set(), "opp": set()})[role].add(action_type)

    for rally_index, by_role in touched.items():
        winner_side = ab_winners.get(rally_index)
        for role, geo_side in (("mine", my_side), ("opp", opp_side)):
            if geo_side is None:
                continue
            won = winner_side == geo_side
            for action_type in by_role[role]:
                if action_type not in ACTION_TYPES:
                    continue
                radar[role]["totals"][action_type] = radar[role]["totals"].get(action_type, 0) + 1
                if won:
                    radar[role]["wins"][action_type] = radar[role]["wins"].get(action_type, 0) + 1


def _radar_points(mine_wins, mine_totals, opp_wins, opp_totals) -> list[dict]:
    points = []
    for action_type in ACTION_TYPES:
        mine_total = mine_totals.get(action_type, 0)
        opp_total = opp_totals.get(action_type, 0)
        points.append({
            "action_type": action_type,
            "team_a_win_rate": (mine_wins.get(action_type, 0) / mine_total) if mine_total else None,
            "team_b_win_rate": (opp_wins.get(action_type, 0) / opp_total) if opp_total else None,
            "sample_size_a": mine_total,
            "sample_size_b": opp_total,
        })
    return points


@ttl_cache()
def compute_team_stats(team_id: str) -> Optional[dict]:
    # Loops every complete job, and for each one re-derives per-set team/
    # side attribution (teams.assign_teams, itself called once per set) -
    # even with load_player_positions's own parse-level cache, that's still
    # real per-request work across N jobs. ttl_cache makes repeat visits to
    # the Teams page near-instant; see resultcache.py for the staleness
    # tradeoff this accepts.
    team = _team_or_none(team_id)
    if team is None:
        return None

    team_players = set(team["players"])
    players_out, hits_by_type, total_hits, games_total = _roster_stats(team_players)

    match_wins = match_losses = 0
    set_wins = set_losses = 0
    games_with_scoring = 0
    game_summaries = []
    radar = {"mine": {"wins": {}, "totals": {}}, "opp": {"wins": {}, "totals": {}}}

    for job in _complete_jobs():
        output_path = config.output_dir(job.id)
        cfg = score.load_config(output_path)
        if cfg["method"] == "none" or team_id not in (cfg["team_x_id"], cfg["team_y_id"]):
            continue
        result = score.load_result(output_path)
        if result is None:
            continue

        my_letter = "x" if cfg["team_x_id"] == team_id else "y"
        opp_letter = "y" if my_letter == "x" else "x"
        games_with_scoring += 1

        game_set_wins = game_set_losses = 0
        for s in result["sets"]:
            set_rallies = [r for r in result["rallies"] if r["set_index"] == s["set_index"]]
            wins_x = sum(1 for r in set_rallies if r["winner"] == "x")
            wins_y = sum(1 for r in set_rallies if r["winner"] == "y")
            my_wins = wins_x if my_letter == "x" else wins_y
            opp_wins_count = wins_y if my_letter == "x" else wins_x
            if my_wins > opp_wins_count:
                set_wins += 1
                game_set_wins += 1
            elif opp_wins_count > my_wins:
                set_losses += 1
                game_set_losses += 1

        if game_set_wins > game_set_losses:
            match_wins += 1
        elif game_set_losses > game_set_wins:
            match_losses += 1

        rallies = score.load_rallies(output_path)
        attribution = score.attribute_sides_per_set(
            output_path, rallies, result["sets"], cfg["team_x_id"], cfg["team_y_id"]
        )
        ab_winners = teams.infer_rally_winners(output_path)
        actions = _actions_for_job(output_path)

        for s in result["sets"]:
            set_attribution = attribution.get(s["set_index"], {})
            my_side = next((side for side, letter in set_attribution.items() if letter == my_letter), None)
            opp_side = next((side for side, letter in set_attribution.items() if letter == opp_letter), None)
            _accumulate_set_radar(output_path, s, rallies, actions, ab_winners, my_side, opp_side, radar)

        game_summaries.append({
            "job_id": job.id,
            "original_filename": job.original_filename,
            "set_wins": game_set_wins,
            "set_losses": game_set_losses,
        })

    return {
        "team_id": team_id,
        "team_name": team["name"],
        "games_total": games_total,
        "games_with_scoring": games_with_scoring,
        "match_wins": match_wins,
        "match_losses": match_losses,
        "set_wins": set_wins,
        "set_losses": set_losses,
        "radar": _radar_points(radar["mine"]["wins"], radar["mine"]["totals"], radar["opp"]["wins"], radar["opp"]["totals"]),
        "total_hits": total_hits,
        "hits_by_type": hits_by_type,
        "players": players_out,
        "games": game_summaries,
    }


def compute_player_radar(name: str) -> dict:
    """Per-player analog of compute_team_stats's radar - "when THIS player
    (not just their whole team) touched the ball with a given action type,
    how often did their own side go on to win the rally", aggregated across
    every complete game. Unlike the team version, there's no named-team
    indirection needed (no Scoring config required) since a player's own
    geometric side for a whole game is enough - so every complete game
    with a usable team split contributes, not just scored ones. Reuses the
    same RadarPointOut shape as the team radar for one consistent chart
    component, with team_b left empty (a single player has no natural
    "opponent" series to plot against)."""
    radar_wins: dict[str, int] = {}
    radar_totals: dict[str, int] = {}
    games_with_data = 0

    for job in _complete_jobs():
        output_path = config.output_dir(job.id)
        names = load_names(output_path)
        stable_ids = {int(stable_id) for stable_id, player_name in names.items() if player_name == name}
        if not stable_ids:
            continue

        sides = teams.assign_teams(output_path)
        matched = {stable_id: record["team"] for stable_id, record in sides.items() if stable_id in stable_ids}
        if not matched:
            continue

        winners = teams.infer_rally_winners(output_path)
        actions = _actions_for_job(output_path)

        contributed = False
        for action in actions:
            stable_id = action.get("player_stable_id")
            rally_index = action.get("rally_index")
            if stable_id not in matched or rally_index is None:
                continue
            action_type = action["action_type"] if action["action_type"] in ACTION_TYPES else "hit"
            if action_type not in ACTION_TYPES:
                continue
            contributed = True
            radar_totals[action_type] = radar_totals.get(action_type, 0) + 1
            if winners.get(rally_index) == matched[stable_id]:
                radar_wins[action_type] = radar_wins.get(action_type, 0) + 1

        if contributed:
            games_with_data += 1

    return {
        "name": name,
        "games_with_data": games_with_data,
        "radar": _radar_points(radar_wins, radar_totals, {}, {}),
    }


def compute_player_quality(name: str) -> dict:
    """Cross-game rollup of action_quality.compute_action_quality's
    per-player averages for one named player - the action-quality analog of
    compute_player_radar. Gives PlayerStatsPage a per-category score plus a
    match-by-match breakdown, so it can show each game's score against this
    player's own median for that category rather than a bare number."""
    # See results_router.get_action_quality's identical call - keeps this
    # cross-game rollup honoring the Configuration page's active heuristic
    # profile too, not just the single-job endpoint.
    heuristics.apply_overrides("consolidating", action_quality)
    matches_by_category: dict[str, list[dict]] = {key: [] for key in QUALITY_CATEGORY_KEYS}

    for job in _complete_jobs():
        output_path = config.output_dir(job.id)
        names = load_names(output_path)
        stable_ids = {int(stable_id) for stable_id, player_name in names.items() if player_name == name}
        if not stable_ids:
            continue

        quality = action_quality.compute_action_quality(job.id, output_path)
        for key in QUALITY_CATEGORY_KEYS:
            for player in quality[key]["players"]:
                if player["stable_id"] in stable_ids and player["average_score"] is not None:
                    matches_by_category[key].append({
                        "job_id": job.id,
                        "original_filename": job.original_filename,
                        "average_score": player["average_score"],
                        "count": player["count"],
                    })

    categories = {}
    for key in QUALITY_CATEGORY_KEYS:
        entries = matches_by_category[key]
        total_count = sum(e["count"] for e in entries)
        weighted_sum = sum(e["average_score"] * e["count"] for e in entries)
        categories[key] = {
            "average_score": (weighted_sum / total_count) if total_count > 0 else None,
            "median_score": _median([e["average_score"] for e in entries]),
            "count": total_count,
            "matches": entries,
        }

    return {"name": name, "categories": categories}
