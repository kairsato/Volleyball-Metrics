"""Which stretches of a video are worth running expensive per-frame
detection on - derived from game_status.json's own rally windows, shared by
PlayerDetection.tracker and BallDetection.ballDetection so both skip the
same dead time identically, rather than each computing its own version of
"close enough" and disagreeing about where a rally actually starts. Kept
here rather than duplicated in each (the usual convention for a small
constant elsewhere in this codebase - see e.g. COURT_LENGTH_M's own
comment on why IT stays duplicated) because this is real logic, not a
number: two independently-maintained copies of a buffer-and-clamp
computation are two chances to drift apart and silently track different
frames for the same job.
"""

import json
from pathlib import Path
from typing import Optional

GAME_STATUS_LOG_NAME = "game_status.json"

# How much earlier than the classifier's own detected start each rally's
# tracked window begins - a serve's toss/windup routinely starts before the
# model's own "play" call does (see gameStatusDetection.py's own
# MIN_RALLY_DURATION_SECONDS comment on how fast a real serve action runs),
# so tracking only from the detected boundary itself was cutting off real
# lead-in action. Deliberately a pre-roll only, not also a post-roll: the
# reported problem was rallies starting late, not ending early, and padding
# the end too would just be guessing at a second failure mode nothing has
# actually shown yet.
PRE_ROLL_BUFFER_S = 1.0


def _load_rallies(output_path) -> Optional[list[dict]]:
    status_file = Path(output_path) / GAME_STATUS_LOG_NAME
    if not status_file.exists():
        return None
    try:
        return json.loads(status_file.read_text()).get("rallies", [])
    except (json.JSONDecodeError, OSError):
        return None


def compute_track_windows(output_path, fps: float) -> Optional[list[tuple[int, int]]]:
    """Merged, buffered [start_frame, end_frame] ranges (inclusive, sorted,
    non-overlapping) worth running player/ball detection over - everywhere
    else is dead time between rallies, per game_status.json.

    Returns None (meaning "no restriction, process every frame") when
    game_status.json doesn't exist yet - the normal case for anything that
    isn't the production pipeline itself (a direct call from
    RunEverything.py or a REPL, say), and the same "missing data ->
    don't filter" tradeoff every other optional-input gate in this codebase
    already uses (see e.g. PlayerDetection.tracker.is_in_play_area's own
    docstring). An EMPTY rallies list (game_status ran but found no rallies
    at all) is NOT treated the same way - that is real information, not
    missing data, so it correctly produces an empty window list rather than
    silently falling back to "track everything".
    """
    rallies = _load_rallies(output_path)
    if rallies is None:
        return None

    rallies = sorted(rallies, key=lambda r: r["start_time_s"])

    windows = []
    previous_end_s = 0.0
    for rally in rallies:
        # Never earlier than 0, and never earlier than the previous rally's
        # own end - "this should never cut the previous play" - so a buffer
        # can only reach into genuine dead time, never into frames the
        # previous rally's own window already claims.
        buffered_start_s = max(0.0, rally["start_time_s"] - PRE_ROLL_BUFFER_S, previous_end_s)
        end_s = rally["end_time_s"]
        if end_s <= buffered_start_s:
            # A rally shorter than the gap already claimed by the previous
            # one's own end - pathological (game_status.json enforces
            # MIN_RALLY_DURATION_SECONDS well above this), but skip rather
            # than emit a backwards/empty range if it ever happens.
            previous_end_s = max(previous_end_s, end_s)
            continue

        start_frame = int(buffered_start_s * fps)
        end_frame = int(round(end_s * fps))

        # Merge into the previous window instead of starting a new one if
        # they touch/overlap once buffered - keeps the seek count down and
        # guarantees the output stays non-overlapping without a separate
        # cleanup pass.
        if windows and start_frame <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end_frame))
        else:
            windows.append((start_frame, end_frame))

        previous_end_s = end_s

    return windows


def in_windows(frame_idx: int, windows: Optional[list[tuple[int, int]]]) -> bool:
    """True if frame_idx falls in one of `windows` - or always True when
    windows is None (see compute_track_windows's own "no restriction"
    case). Linear scan: `windows` is one entry per rally, at most a few
    hundred even for a long match, and this is only ever called from a
    frame-by-frame read loop that already does far more per-frame work than
    this comparison costs."""
    if windows is None:
        return True
    return any(start <= frame_idx <= end for start, end in windows)
