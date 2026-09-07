"""Per-video "warmup period" - the [start_s, end_s] window of the uploaded
footage that actually counts as the match. Once confirmed, this becomes the
single boundary every viewing/results surface respects: the video player
can't be scrubbed outside it, thumbnails/frame grabs are drawn from inside
it, and rallies/hits/stats detected outside it are excluded entirely rather
than just hidden in the player. Displayed/seekable time everywhere outside
this module is always relative to start_s (0:00 == start_s), so nothing
downstream has to know the absolute video position at all.

Deliberately does NOT touch the Setup/configuration tools themselves (Court
Calibration, the Scoring Determination OCR region picker, or this feature's
own picker) - those need to see the whole uploaded file to be usable,
including to set this boundary in the first place.

Same load/save/confirm shape as score.py's ScoreConfig - see that module's
docstring for why the confirm flag lives separately from the range itself.
"""

import json
from pathlib import Path
from typing import Optional

WARMUP_CONFIG_NAME = "warmup_config.json"

DEFAULT_CONFIG = {
    "start_s": 0.0,
    # None means "the end of the video" - resolved against the job's actual
    # duration by effective_range rather than baked in here, since duration
    # isn't always known yet when a default config is first read.
    "end_s": None,
    "confirmed": False,
}


def _load_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None


def load_config(output_path: Path) -> dict:
    saved = _load_json(output_path / WARMUP_CONFIG_NAME)
    return {**DEFAULT_CONFIG, **(saved or {})}


def save_config(output_path: Path, config_update: dict) -> dict:
    current = load_config(output_path)
    current.update(config_update)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / WARMUP_CONFIG_NAME).write_text(json.dumps(current, indent=2))
    return current


def effective_range(cfg: dict, duration_s: Optional[float]) -> tuple[float, float]:
    """Resolves the config's possibly-None end_s against the video's actual
    duration - callers always get a concrete [start_s, end_s] pair."""
    start_s = cfg.get("start_s") or 0.0
    end_s = cfg.get("end_s")
    if end_s is None:
        end_s = duration_s if duration_s is not None else float("inf")
    return start_s, end_s


def is_active(cfg: dict) -> bool:
    return bool(cfg.get("confirmed"))


def _in_range(timestamp_s: float, start_s: float, end_s: float) -> bool:
    return start_s <= timestamp_s <= end_s


def rebase_rallies(rallies: list[dict], start_s: float, end_s: float) -> list[dict]:
    """Drops any rally that isn't fully inside [start_s, end_s] and shifts
    the rest so start_time_s/end_time_s become relative to start_s."""
    kept = []
    for rally in rallies:
        if rally["start_time_s"] < start_s or rally["end_time_s"] > end_s:
            continue
        kept.append({
            **rally,
            "start_time_s": rally["start_time_s"] - start_s,
            "end_time_s": rally["end_time_s"] - start_s,
        })
    return kept


def rebase_stats(stats: dict, start_s: float, end_s: float) -> dict:
    """Filters every player's events to [start_s, end_s] and recomputes
    every aggregate field (total_hits, hits_by_type, rallies_participated,
    rally_ending_touches/_rate) from exactly that filtered set - mirrors
    Analysis/PostProcessing/consolidate.py's own aggregation algorithm
    exactly (including its per-rally "last touch" tracking for
    rally_ending_touches), just re-run against the trimmed event list
    instead of the full one, so a touch that only counted as rally-ending
    because of a later touch now excluded doesn't stay miscounted."""
    players = stats.get("players", {})

    filtered_events_by_player: dict[str, list[dict]] = {}
    last_touch_frame_by_rally: dict[int, int] = {}

    for player_id, record in players.items():
        filtered = [e for e in record.get("events", []) if _in_range(e["timestamp_s"], start_s, end_s)]
        filtered_events_by_player[player_id] = filtered
        for event in filtered:
            rally_index = event.get("rally_index")
            if rally_index is None:
                continue
            if event["frame_idx"] > last_touch_frame_by_rally.get(rally_index, -1):
                last_touch_frame_by_rally[rally_index] = event["frame_idx"]

    rebased_players = {}
    for player_id, record in players.items():
        filtered = filtered_events_by_player[player_id]
        hits_by_type = {action_type: 0 for action_type in record.get("hits_by_type", {})}
        rallies_participated: set[int] = set()
        rally_ending_touches = 0
        rebased_events = []

        for event in filtered:
            action_type = event.get("action_type", "hit")
            hits_by_type[action_type] = hits_by_type.get(action_type, 0) + 1
            rally_index = event.get("rally_index")
            if rally_index is not None:
                rallies_participated.add(rally_index)
                if last_touch_frame_by_rally.get(rally_index) == event["frame_idx"]:
                    rally_ending_touches += 1
            rebased_events.append({**event, "timestamp_s": event["timestamp_s"] - start_s})

        total_hits = len(filtered)
        rebased_players[player_id] = {
            **record,
            "total_hits": total_hits,
            "hits_by_type": hits_by_type,
            "rallies_participated": len(rallies_participated),
            "rally_ending_touches": rally_ending_touches,
            "rally_ending_touch_rate": (rally_ending_touches / total_hits) if total_hits else None,
            "events": rebased_events,
        }

    return {**stats, "players": rebased_players}
