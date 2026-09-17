"""Stitches every video segment's own per-segment analysis output into one
continuous job-level timeline, for a multi-video job (see jobs.py's
Job.videos / job_videos()). Called once, in-process (pure JSON
manipulation, not GPU/CPU work - same reasoning as players.py's
merge_names_into_stats-style helpers), right after phase one's last stage
finishes (see pipeline._phase_one) and before phase two's job-level stages
("consolidating"/"dashboard") run against the result.

For a 1-segment job (every existing job, and any new job with only one
file), merge_segments is a true no-op early return - segment 0 already
writes directly to output_dir(job_id), which is exactly where phase two
already reads from. This is the central backward-compatibility guarantee:
nothing changes for any job that isn't genuinely multi-segment.

For a genuinely multi-segment job, every segment's own raw per-segment
files (game_status.json, actions.json, ball_speed.json,
player_positions.json, player_names.json, player_ignored.json - all
written into that segment's own output dir by phase one's stages, see
config.resolve_video_and_output) get concatenated into one job-level
version of each, written into segment 0's own output dir (=
output_dir(job_id), the exact path phase two's stages already read from) -
overwriting segment 0's own raw files with the merged result. Segment 0's
own data always occupies the front of every merged timeline; segments 1+
follow, each offset by the cumulative (post-warmup-trim) duration of every
segment before it.

Player identity across segments is handled by namespacing
player_stable_id/stable_id per segment (segment_order * ID_NAMESPACE_STEP +
original_id) so two segments' tracker id 3 never collide as the same
untracked person, then merging player_names.json/player_ignored.json with
the same remap. Cross-segment identity then works for free through the
existing name-based grouping (players.build_canonical_mapping/
write_grouped_actions, unchanged) - a player named the same way in every
segment they appear in (whether by auto_identify_from_gallery or a human
via the Player Identification tab) gets merged stats automatically.
KNOWN LIMITATION, accepted: a player not named consistently across every
segment they appear in shows up as separate per-segment rows until named
consistently.
"""

import json
from pathlib import Path
from typing import Optional

from .. import config
from . import players, warmup

# Local, not imported from the analysis modules that own the canonical
# constant - avoids pulling their heavy torch/cv2 imports into a module
# that's pure JSON manipulation. Values must stay in sync with
# BallDetection.ballDetection.SPEED_LOG_NAME and
# ActionDetection.actionDetection.ACTIONS_LOG_NAME/PLAYER_POSITIONS_LOG_NAME.
BALL_SPEED_FILE_NAME = "ball_speed.json"
ACTIONS_FILE_NAME = "actions.json"

# Namespaces a segment's own tracker ids (player_stable_id/stable_id) so
# segment 5's id 3 and segment 0's id 3 never collide as the same
# untracked person once merged. Comfortably above any realistic per-segment
# tracker id count.
ID_NAMESPACE_STEP = 100_000

DEFAULT_FPS_ASSUMPTION = 30.0


def _load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _segment_fps(game_status: dict, entry_count: int, duration_s: Optional[float]) -> float:
    """This segment's own fps, preferring game_status.json's recorded value
    (set once by gameStatusDetection.py from the real video) over the
    len(entries)/duration_s approximation results_router.py's endpoints use
    when nothing better is available."""
    fps = game_status.get("fps")
    if fps:
        return float(fps)
    if duration_s and entry_count > 0:
        return entry_count / duration_s
    return DEFAULT_FPS_ASSUMPTION


def _rebase_dense_log(entries: list[dict], start_s: float, end_s: float, fps: float) -> list[dict]:
    """Filters a per-frame log (ball_speed.json/player_positions.json) to
    [start_s, end_s] by each entry's own timestamp (frame_idx/fps for
    ball_speed, which has no timestamp_s of its own; the entry's own
    timestamp_s for player_positions) - filtering by value rather than by
    list position, since neither file's density relative to frame_idx is
    guaranteed. frame_idx/timestamp_s on every kept entry are reassigned
    from scratch (0-based, relative to start_s) - callers add their own
    cumulative offset on top afterward."""
    kept = []
    for i, entry in enumerate(entries):
        timestamp_s = entry.get("timestamp_s")
        if timestamp_s is None:
            timestamp_s = entry.get("frame_idx", i) / fps
        if timestamp_s < start_s or timestamp_s > end_s:
            continue
        kept.append({**entry, "timestamp_s": timestamp_s - start_s})

    # Reassign frame_idx sequentially now that gaps (if any) are closed up -
    # matches ball_speed.json's own "frame_idx == position in file" shape,
    # and is at worst a harmless renumbering for player_positions.json if it
    # ever turns out to have gaps of its own.
    for i, entry in enumerate(kept):
        entry["frame_idx"] = i
    return kept


def merge_segments(job_id: str, segments: list[dict]) -> None:
    if len(segments) <= 1:
        return  # true no-op - see module docstring

    job0_video, job0_output = config.resolve_video_and_output(job_id, segments[0])

    merged_rallies: list[dict] = []
    merged_actions: list[dict] = []
    merged_ball_speed: list[dict] = []
    merged_player_positions: list[dict] = []
    merged_names: dict[str, str] = {}
    merged_ignored: set[int] = set()

    cumulative_offset_s = 0.0
    next_rally_index = 0
    job_fps = None

    for segment in segments:
        _video_path, output_path = config.resolve_video_and_output(job_id, segment)

        game_status = _load_json(output_path / config.GAME_STATUS_FILE_NAME, {})
        rallies = game_status.get("rallies", [])
        actions = _load_json(output_path / ACTIONS_FILE_NAME, [])
        ball_speed = _load_json(output_path / BALL_SPEED_FILE_NAME, [])
        player_positions = _load_json(output_path / players.PLAYER_POSITIONS_NAME, [])

        fps = _segment_fps(game_status, len(ball_speed), segment.get("duration_s"))
        if job_fps is None:
            job_fps = fps

        # Per-segment warmup trimming happens first, against this segment's
        # OWN raw data, before its contribution to the cumulative time
        # offset is computed - so the merged timeline has no dead gaps
        # where trimmed footage used to be.
        duration_s = segment.get("duration_s") or (len(ball_speed) / fps if ball_speed else 0.0)
        warmup_cfg = warmup.load_config(output_path)
        if warmup.is_active(warmup_cfg):
            start_s, end_s = warmup.effective_range(warmup_cfg, duration_s)
            rallies = warmup.rebase_rallies(rallies, start_s, end_s)
            actions = warmup.rebase_actions(actions, start_s, end_s)
            ball_speed = _rebase_dense_log(ball_speed, start_s, end_s, fps)
            player_positions = _rebase_dense_log(player_positions, start_s, end_s, fps)
            duration_s = end_s - start_s

        segment_order = segment["order"]
        id_offset = segment_order * ID_NAMESPACE_STEP

        # Rallies: append at their final global index, offset in time -
        # build the local->global rally_index map from this pass so
        # actions below can follow the same remap.
        rally_index_map: dict[int, int] = {}
        for local_index, rally in enumerate(rallies):
            new_index = next_rally_index + local_index
            rally_index_map[rally.get("rally_index")] = new_index
            merged_rallies.append({
                **rally,
                "rally_index": new_index,
                "start_time_s": rally["start_time_s"] + cumulative_offset_s,
                "end_time_s": rally["end_time_s"] + cumulative_offset_s,
                "start_frame": round((rally["start_time_s"] + cumulative_offset_s) * job_fps),
                "end_frame": round((rally["end_time_s"] + cumulative_offset_s) * job_fps),
                # Unused today - the seam for a future video-boundary-aware
                # auto game-split feature.
                "segment_id": segment["id"],
                "segment_order": segment_order,
            })
        next_rally_index += len(rallies)

        for action in actions:
            player_id = action.get("player_stable_id")
            merged_actions.append({
                **action,
                "timestamp_s": action["timestamp_s"] + cumulative_offset_s,
                "frame_idx": round((action["timestamp_s"] + cumulative_offset_s) * fps),
                "rally_index": rally_index_map.get(action.get("rally_index")),
                "player_stable_id": (player_id + id_offset) if player_id is not None else None,
            })

        for entry in ball_speed:
            merged_ball_speed.append({
                **entry,
                "frame_idx": len(merged_ball_speed),
                "timestamp_s": entry["timestamp_s"] + cumulative_offset_s,
            })

        for frame in player_positions:
            merged_player_positions.append({
                **frame,
                "frame_idx": len(merged_player_positions),
                "timestamp_s": frame["timestamp_s"] + cumulative_offset_s,
                "players": [
                    {**p, "stable_id": p["stable_id"] + id_offset}
                    for p in frame.get("players", [])
                ],
            })

        for stable_id_str, name in players.load_names(output_path).items():
            merged_names[str(int(stable_id_str) + id_offset)] = name
        for stable_id in players.load_ignored(output_path):
            merged_ignored.add(stable_id + id_offset)

        cumulative_offset_s += duration_s

    # Written directly rather than through save_names/save_ignored - those
    # merge INTO whatever's already on disk (right for an incremental UI
    # edit, wrong here where merged_names/merged_ignored are already the
    # complete, definitive result and should replace segment 0's own raw
    # file outright).
    job0_output.mkdir(parents=True, exist_ok=True)
    (job0_output / config.GAME_STATUS_FILE_NAME).write_text(json.dumps(
        {"fps": job_fps, "total_frames": len(merged_ball_speed), "rallies": merged_rallies}, indent=2,
    ))
    (job0_output / ACTIONS_FILE_NAME).write_text(json.dumps(merged_actions, indent=2))
    (job0_output / BALL_SPEED_FILE_NAME).write_text(json.dumps(merged_ball_speed, indent=2))
    (job0_output / players.PLAYER_POSITIONS_NAME).write_text(json.dumps(merged_player_positions, indent=2))
    (job0_output / config.PLAYER_NAMES_NAME).write_text(json.dumps(merged_names, indent=2))
    (job0_output / config.PLAYER_IGNORED_NAME).write_text(json.dumps(sorted(merged_ignored), indent=2))
