"""A global roster of real people's names, shared across every job - not to
be confused with player_names.json, which maps one job's stable_ids to a
name. The roster exists so a name only has to be typed once: it seeds the
naming dropdown on every future job's Setup/Verify players screen, and picks
up any new name typed there too."""

import json

from . import config


def _roster_file():
    return config.DATA_DIR / config.PLAYER_ROSTER_NAME


def load_roster() -> list[str]:
    roster_file = _roster_file()
    if not roster_file.exists():
        return []
    return json.loads(roster_file.read_text())


def save_roster(names: list[str]) -> list[str]:
    # Case-insensitive de-dup ("Kai" and "kai" typed on different jobs
    # shouldn't become two separate roster entries) while preserving
    # whichever casing was seen first.
    seen: dict[str, str] = {}
    for name in names:
        name = name.strip()
        if name and name.lower() not in seen:
            seen[name.lower()] = name

    result = sorted(seen.values(), key=str.lower)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _roster_file().write_text(json.dumps(result, indent=2))
    return result


def add_to_roster(name: str) -> list[str]:
    return save_roster([*load_roster(), name])


def remove_from_roster(name: str) -> list[str]:
    name = name.strip().lower()
    return save_roster([n for n in load_roster() if n.lower() != name])
