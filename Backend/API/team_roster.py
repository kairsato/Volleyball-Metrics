"""A global list of user-defined teams, each a named set of player names
from the roster - not to be confused with teams.py, which infers a
per-video Team A / Team B split geometrically from tracked court position.
This is a persistent, user-authored grouping (e.g. "Varsity", "JV") that
exists purely to organize the frontend's Teams page; it has no bearing on
any per-video analysis."""

import json
import uuid

from . import config

TEAM_ROSTER_NAME = "team_roster.json"


def _teams_file():
    return config.DATA_DIR / TEAM_ROSTER_NAME


def load_teams() -> list[dict]:
    teams_file = _teams_file()
    if not teams_file.exists():
        return []
    return json.loads(teams_file.read_text())


def _save(teams: list[dict]) -> list[dict]:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _teams_file().write_text(json.dumps(teams, indent=2))
    return teams


def add_team(name: str, players: list[str]) -> list[dict]:
    teams = load_teams()
    teams.append({"id": uuid.uuid4().hex[:12], "name": name.strip(), "players": players})
    return _save(teams)


def update_team(team_id: str, name: str, players: list[str]) -> list[dict]:
    teams = load_teams()
    for team in teams:
        if team["id"] == team_id:
            team["name"] = name.strip()
            team["players"] = players
    return _save(teams)


def remove_team(team_id: str) -> list[dict]:
    return _save([t for t in load_teams() if t["id"] != team_id])
