"""Cross-video aggregate stats for a named team (team_roster.py) - win/loss
record and a win-rate-by-action-type radar - plus the per-player analog
(PlayerStatsPage's radar). team_roster.py's own docstring is explicit that
a named team has "no bearing on any per-video analysis" by itself; the only
place a named team is ever linked to a geometric court side at all is
score.py's per-video Scoring config (team_x_id/team_y_id), so win/loss and
radar here only ever draw from videos where that link exists. Hit-count/
action-mix totals are looser - they only need a name match against
player_stats.json, so every complete video contributes those regardless of
whether Scoring was ever configured for it.
"""
import json
from pathlib import Path
from typing import Optional

from . import config, jobs, score, team_roster, teams
from .players import STATS_NAME as PLAYER_STATS_NAME
from .players import load_names

ACTION_TYPES = teams.ACTION_TYPES


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
    PlayersPage's own client-side cross-video aggregation already keys on),
    not stable_id, since stable_ids aren't stable across videos. Returns
    (per_player_summaries, team_hits_by_type, team_total_hits,
    videos_with_a_team_player)."""
    per_player: dict[str, dict] = {name: {"total_hits": 0, "hits_by_type": {}} for name in team_players}
    hits_by_type_team: dict[str, int] = {}
    total_hits_team = 0
    videos_total = 0

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
            videos_total += 1

    players_out = [
        {"name": name, "total_hits": data["total_hits"], "hits_by_type": data["hits_by_type"]}
        for name, data in sorted(per_player.items())
    ]
    return players_out, hits_by_type_team, total_hits_team, videos_total


def _accumulate_game_radar(
    output_path: Path,
    game: dict,
    rallies: list[dict],
    actions: list[dict],
    ab_winners: dict[int, Optional[str]],
    my_side: Optional[str],
    opp_side: Optional[str],
    radar: dict[str, dict[str, dict[str, int]]],
) -> None:
    """Mutates radar (`{"mine": {"wins": {...}, "totals": {...}}, "opp":
    {...}}`) with this game's contribution - my_side/opp_side are which
    geometric side ("A"/"B", or None if unattributed) this team/its
    opponent occupy for THIS game specifically (sides can swap between
    games/sets, hence per-game rather than a single whole-video split)."""
    if my_side is None and opp_side is None:
        return

    start_frame = rallies[game["start_rally_index"]]["start_frame"]
    end_frame = rallies[game["end_rally_index"]]["end_frame"]
    sides = teams.assign_teams(output_path, frame_range=(start_frame, end_frame))
    rally_indices = {
        rallies[i]["rally_index"] for i in range(game["start_rally_index"], game["end_rally_index"] + 1)
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


def compute_team_stats(team_id: str) -> Optional[dict]:
    team = _team_or_none(team_id)
    if team is None:
        return None

    team_players = set(team["players"])
    players_out, hits_by_type, total_hits, videos_total = _roster_stats(team_players)

    match_wins = match_losses = 0
    game_wins = game_losses = 0
    videos_with_scoring = 0
    video_summaries = []
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
        videos_with_scoring += 1

        video_game_wins = video_game_losses = 0
        for g in result["games"]:
            game_rallies = [r for r in result["rallies"] if r["game_index"] == g["game_index"]]
            wins_x = sum(1 for r in game_rallies if r["winner"] == "x")
            wins_y = sum(1 for r in game_rallies if r["winner"] == "y")
            my_wins = wins_x if my_letter == "x" else wins_y
            opp_wins_count = wins_y if my_letter == "x" else wins_x
            if my_wins > opp_wins_count:
                game_wins += 1
                video_game_wins += 1
            elif opp_wins_count > my_wins:
                game_losses += 1
                video_game_losses += 1

        if video_game_wins > video_game_losses:
            match_wins += 1
        elif video_game_losses > video_game_wins:
            match_losses += 1

        rallies = score.load_rallies(output_path)
        attribution = score.attribute_sides_per_game(
            output_path, rallies, result["games"], cfg["team_x_id"], cfg["team_y_id"]
        )
        ab_winners = teams.infer_rally_winners(output_path)
        actions = _actions_for_job(output_path)

        for g in result["games"]:
            game_attribution = attribution.get(g["game_index"], {})
            my_side = next((side for side, letter in game_attribution.items() if letter == my_letter), None)
            opp_side = next((side for side, letter in game_attribution.items() if letter == opp_letter), None)
            _accumulate_game_radar(output_path, g, rallies, actions, ab_winners, my_side, opp_side, radar)

        video_summaries.append({
            "job_id": job.id,
            "original_filename": job.original_filename,
            "game_wins": video_game_wins,
            "game_losses": video_game_losses,
        })

    return {
        "team_id": team_id,
        "team_name": team["name"],
        "videos_total": videos_total,
        "videos_with_scoring": videos_with_scoring,
        "match_wins": match_wins,
        "match_losses": match_losses,
        "game_wins": game_wins,
        "game_losses": game_losses,
        "radar": _radar_points(radar["mine"]["wins"], radar["mine"]["totals"], radar["opp"]["wins"], radar["opp"]["totals"]),
        "total_hits": total_hits,
        "hits_by_type": hits_by_type,
        "players": players_out,
        "videos": video_summaries,
    }


def compute_player_radar(name: str) -> dict:
    """Per-player analog of compute_team_stats's radar - "when THIS player
    (not just their whole team) touched the ball with a given action type,
    how often did their own side go on to win the rally", aggregated across
    every complete video. Unlike the team version, there's no named-team
    indirection needed (no Scoring config required) since a player's own
    geometric side for a whole video is enough - so every complete video
    with a usable team split contributes, not just scored ones. Reuses the
    same RadarPointOut shape as the team radar for one consistent chart
    component, with team_b left empty (a single player has no natural
    "opponent" series to plot against)."""
    radar_wins: dict[str, int] = {}
    radar_totals: dict[str, int] = {}
    videos_with_data = 0

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
            videos_with_data += 1

    return {
        "name": name,
        "videos_with_data": videos_with_data,
        "radar": _radar_points(radar_wins, radar_totals, {}, {}),
    }
