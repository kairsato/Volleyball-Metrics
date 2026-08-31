import json
from collections import defaultdict
from pathlib import Path

ACTIONS_LOG_NAME = "actions.json"
GAME_STATUS_LOG_NAME = "game_status.json"
STATS_LOG_NAME = "player_stats.json"

ACTION_TYPES = ["serve", "spike", "set", "dig", "block", "hit"]

CAVEATS = [
    "action_type comes from a rough geometric heuristic (ball trajectory shape "
    "and net proximity), not a trained action classifier - expect real "
    "misclassification, especially between set and dig, which look similar "
    "from trajectory alone.",

    "rally_ending_touches counts how often a player's touch was the LAST one "
    "before their rally ended. This is NOT a true miss/error rate: it counts "
    "both genuine errors AND winning kill shots the opposing team couldn't "
    "return, which look identical from hit-sequence data alone. Telling those "
    "apart needs to know which side of the net the ball ended up on, or an "
    "actual score signal - this pipeline has neither yet.",

    "player_stable_id comes from PlayerDetection's appearance+motion "
    "re-identification system, not a jersey number or other ground-truth "
    "identity - a long occlusion or a player leaving and re-entering frame "
    "can still occasionally split one real player into two stable_ids."
]


def consolidateStats(output_path, actions_filename=ACTIONS_LOG_NAME):
    """
    Reads actions.json (+ game_status.json for rally boundaries) and rolls
    hit events up into per-player statistics, associated with their original
    timestamps so any number here can be traced back to the moment it came
    from. See CAVEATS above for what these numbers do and don't actually mean.

    actions_filename lets a caller point this at a variant of the actions
    log instead - e.g. one with player_stable_id values already remapped to
    merge stable_ids a human has confirmed are the same person.
    """
    actions_file = Path(output_path) / actions_filename

    if not actions_file.exists():
        print(f"No actions log found at {actions_file}; run action detection first.")
        return

    with open(actions_file) as f:
        actions = json.load(f)

    rallies_by_index = {}
    status_file = Path(output_path) / GAME_STATUS_LOG_NAME
    if status_file.exists():
        with open(status_file) as f:
            for rally in json.load(f)["rallies"]:
                rallies_by_index[rally["rally_index"]] = rally

    # Last hit of each rally, regardless of who made it - needed to know
    # whether a given hit was "the one that ended the rally".
    last_hit_frame_by_rally = {}
    for action in actions:
        rally_index = action["rally_index"]
        if rally_index is None:
            continue
        current = last_hit_frame_by_rally.get(rally_index)
        if current is None or action["frame_idx"] > current:
            last_hit_frame_by_rally[rally_index] = action["frame_idx"]

    players = defaultdict(lambda: {
        "total_hits": 0,
        "hits_by_type": {t: 0 for t in ACTION_TYPES},
        "rallies_participated": set(),
        "rally_ending_touches": 0,
        "events": []
    })

    for action in actions:

        player_id = action["player_stable_id"]
        if player_id is None:
            continue

        record = players[player_id]
        record["total_hits"] += 1

        action_type = action["action_type"] if action["action_type"] in ACTION_TYPES else "hit"
        record["hits_by_type"][action_type] += 1

        rally_index = action["rally_index"]
        if rally_index is not None:
            record["rallies_participated"].add(rally_index)

            if last_hit_frame_by_rally.get(rally_index) == action["frame_idx"]:
                record["rally_ending_touches"] += 1

        record["events"].append({
            "frame_idx": action["frame_idx"],
            "timestamp_s": action["timestamp_s"],
            "action_type": action_type,
            "rally_index": rally_index,
            # .get(...) rather than direct indexing - actions_filename can
            # point at actions_grouped.json (see write_grouped_actions),
            # which only remaps player_stable_id/drops ignored players and
            # otherwise copies actions.json's records verbatim, so these are
            # always present there too; .get keeps this tolerant of a stale
            # actions.json from before these fields existed.
            "speed_in_m_per_s": action.get("speed_in_m_per_s"),
            "speed_out_m_per_s": action.get("speed_out_m_per_s"),
            "real_units": action.get("real_units", False),
            "ball_height_m": action.get("ball_height_m"),
            "time_since_prev_touch_s": action.get("time_since_prev_touch_s"),
        })

    output = {"players": {}, "caveats": CAVEATS}

    for player_id, record in players.items():
        total_hits = record["total_hits"]

        output["players"][str(player_id)] = {
            "total_hits": total_hits,
            "hits_by_type": record["hits_by_type"],
            "rallies_participated": len(record["rallies_participated"]),
            "rally_ending_touches": record["rally_ending_touches"],
            "rally_ending_touch_rate": (
                record["rally_ending_touches"] / total_hits if total_hits else None
            ),
            "events": sorted(record["events"], key=lambda e: e["frame_idx"])
        }

    stats_file = Path(output_path) / STATS_LOG_NAME

    with open(stats_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Consolidated stats for {len(output['players'])} player(s).")
    print(f"Player stats saved: {stats_file}")
