"""A persisted, roster-wide rollup of every named player's cross-game stats
- backs the Players list page and each player's own profile page.

This deliberately stores its result on disk (player_profiles.json) rather
than recomputing it - or even just time-caching it (see resultcache.py) -
on every page visit. The naive version of this (what PlayersPage/
PlayerStatsPage used to do) re-fetched and re-aggregated every complete
job's player_stats.json/player_positions.json/actions.json from scratch on
every single page load, which only gets slower as more games get added.
Instead, get_profiles()/get_profile() below serve directly from disk and
only pay the full rebuild cost once, the first time anything reads this
after the underlying data actually changed.

Staleness is detected via _signature() below - for every complete job, the
mtime of each file this module actually reads (player_stats.json,
score_config.json/score_result.json, warmup_config.json, court.json,
actions[_grouped].json) - compared against what the stored file was last
built from. This is deliberately keyed on those files directly rather than
Job.updated_at: naming/scoring/warmup actions persist straight to their own
job-output files without going through jobs.store.update (see
players_router.py/score_router.py/warmup_router.py), so updated_at doesn't
actually bump for most of what this module cares about. Keying on the real
files means no explicit "invalidate after this specific action" hooks are
needed scattered across every mutation site - any change that could affect
a player's stats necessarily touches one of these files, which this
notices on the next read.
"""
import json
from pathlib import Path
from typing import Optional

from .. import config
from . import players, score, team_roster, team_stats, warmup
from .players import STATS_NAME as PLAYER_STATS_NAME

PROFILES_FILE_NAME = "player_profiles.json"

# Every file _build() below actually reads for a given job - see this
# module's docstring for why the staleness signature is keyed on these
# directly instead of Job.updated_at.
_SIGNATURE_FILES = (
    PLAYER_STATS_NAME,
    warmup.WARMUP_CONFIG_NAME,
    score.SCORE_CONFIG_NAME,
    score.SCORE_RESULT_NAME,
    config.COURT_FILE_NAME,
    "actions_grouped.json",
    "actions.json",
)


def _profiles_file() -> Path:
    return config.DATA_DIR / PROFILES_FILE_NAME


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _job_signature(job) -> list:
    output_path = config.output_dir(job.id)
    stamps = []
    for filename in _SIGNATURE_FILES:
        f = output_path / filename
        stamps.append(f.stat().st_mtime_ns if f.exists() else 0)
    return [job.id, *stamps]


def _signature() -> list[list]:
    return sorted((_job_signature(job) for job in team_stats._complete_jobs()), key=lambda s: s[0])


def _teams_by_id() -> dict[str, dict]:
    return {team["id"]: team for team in team_roster.load_teams()}


def _player_team(name: str, cfg: dict, teams_by_id: dict[str, dict]) -> tuple[Optional[str], Optional[str]]:
    """Which of this game's two configured teams (if either) has `name` on
    its roster - the affiliation a per-player win/loss trend needs, since
    score.compute_summary only resolves a whole game's winner/loser team
    names, not which side any particular player was actually on."""
    for team_id in (cfg.get("team_x_id"), cfg.get("team_y_id")):
        team = teams_by_id.get(team_id) if team_id else None
        if team and name in team.get("players", []):
            return team["id"], team["name"]
    return None, None


def _build() -> dict:
    signature = _signature()
    teams_by_id = _teams_by_id()
    players_acc: dict[str, dict] = {}

    def acc_for(name: str) -> dict:
        return players_acc.setdefault(name, {
            "name": name,
            "thumbnail_base64": None,
            "total_hits": 0,
            "rallies_participated": 0,
            "rally_ending_touches": 0,
            "hits_by_type": {},
            "games": [],
        })

    for job in team_stats._complete_jobs():
        output_path = config.output_dir(job.id)
        stats = _load_json(output_path / PLAYER_STATS_NAME)
        if not stats:
            continue

        # Same warmup trimming GET /results applies (see
        # results_router._get_results_sync) - player_stats.json itself is
        # never rewritten when a warmup period is confirmed, only rebased
        # on the fly, so reading it raw here would disagree with what every
        # other stats view in the app already shows for this job.
        warmup_cfg = warmup.load_config(output_path)
        if warmup.is_active(warmup_cfg):
            start_s, end_s = warmup.effective_range(warmup_cfg, job.duration_s)
            stats = warmup.rebase_stats(stats, start_s, end_s)

        player_records = stats.get("players", {})
        if not player_records:
            continue

        cfg = score.load_config(output_path)
        summary = score.compute_summary(output_path)

        thumbnails_by_stable_id: dict[str, Optional[str]] = {}
        video_path = config.find_input_video(job.id)
        if video_path is not None:
            try:
                for p in players.list_players(video_path, output_path):
                    thumbnails_by_stable_id[str(p["stable_id"])] = p.get("thumbnail_base64")
            except Exception:
                pass

        for stable_id_str, record in player_records.items():
            name = (record.get("name") or "").strip()
            if not name:
                continue

            entry = acc_for(name)
            entry["total_hits"] += record.get("total_hits", 0)
            entry["rallies_participated"] += record.get("rallies_participated", 0)
            entry["rally_ending_touches"] += record.get("rally_ending_touches", 0)
            for action_type, count in record.get("hits_by_type", {}).items():
                entry["hits_by_type"][action_type] = entry["hits_by_type"].get(action_type, 0) + count

            if entry["thumbnail_base64"] is None:
                thumb = thumbnails_by_stable_id.get(stable_id_str)
                if thumb:
                    entry["thumbnail_base64"] = thumb

            team_id, team_name = _player_team(name, cfg, teams_by_id)
            result = None
            if team_name is not None and summary["winner_team_name"] is not None:
                if team_name == summary["winner_team_name"]:
                    result = "win"
                elif team_name == summary["loser_team_name"]:
                    result = "loss"

            entry["games"].append({
                "job_id": job.id,
                "original_filename": job.original_filename,
                "date_played": job.date_played,
                "total_hits": record.get("total_hits", 0),
                "rallies_participated": record.get("rallies_participated", 0),
                "rally_ending_touches": record.get("rally_ending_touches", 0),
                "hits_by_type": record.get("hits_by_type", {}),
                "team_id": team_id,
                "team_name": team_name,
                "result": result,
            })

    # Radar (win-rate by action) and action-quality both need their own
    # full pass over every complete job per player (team splits, actions,
    # weighted factor scores) - genuinely more expensive than the hit-count
    # rollup above, which is exactly why this whole module exists: that
    # cost is paid once here, during a rebuild, not on every page visit.
    for name, entry in players_acc.items():
        radar = team_stats.compute_player_radar(name)
        entry["radar"] = radar["radar"]
        entry["games_with_radar_data"] = radar["games_with_data"]

        quality = team_stats.compute_player_quality(name)
        entry["quality_categories"] = quality["categories"]

        entry["games"].sort(key=lambda g: g["date_played"] or "", reverse=True)

    return {"signature": signature, "players": players_acc}


def _read_or_build() -> dict:
    stored = _load_json(_profiles_file())
    if stored is not None and stored.get("signature") == _signature():
        return stored

    fresh = _build()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _profiles_file().write_text(json.dumps(fresh))
    return fresh


def warm() -> None:
    """Forces a rebuild-if-stale right now, discarding the result - meant to
    be scheduled as a BackgroundTask right after an action that just
    changed a job's player-visible stats (confirming names/scoring,
    deleting a job). Without this, the *next* visit to the Players list or
    a profile page would be the one to pay the rebuild cost instead;
    calling this eagerly means that visit already finds a fresh, fast
    store waiting on disk."""
    _read_or_build()


def get_profiles() -> list[dict]:
    """Lightweight roster-wide summary - what the Players list page's grid
    needs, and nothing more (no per-game breakdown, radar, or quality)."""
    stored = _read_or_build()
    return [
        {
            "name": p["name"],
            "thumbnail_base64": p["thumbnail_base64"],
            "total_hits": p["total_hits"],
            "rallies_participated": p["rallies_participated"],
            "game_count": len(p["games"]),
        }
        for p in stored["players"].values()
    ]


_EMPTY_PROFILE_CATEGORIES = {
    key: {"average_score": None, "median_score": None, "count": 0, "matches": []}
    for key in team_stats.QUALITY_CATEGORY_KEYS
}


def get_profile(name: str) -> dict:
    """Always returns a profile, even for a roster-only name with no games
    yet (a zero-stat placeholder, same as the Players list page already
    shows for one) - a player who exists but has nothing recorded isn't an
    error case."""
    stored = _read_or_build()
    existing = stored["players"].get(name)
    if existing is not None:
        return existing
    return {
        "name": name,
        "thumbnail_base64": None,
        "total_hits": 0,
        "rallies_participated": 0,
        "rally_ending_touches": 0,
        "hits_by_type": {},
        "games": [],
        "radar": [],
        "games_with_radar_data": 0,
        "quality_categories": _EMPTY_PROFILE_CATEGORIES,
    }
