import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from . import config, players
from .jobs import (
    STATUS_CANCELLED,
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_FINALIZING,
    STATUS_PROCESSING,
    store,
)
from .players import merge_names_into_stats

BACKEND_DIR = Path(__file__).resolve().parent.parent

_processes_lock = threading.Lock()
_active_processes: dict[str, subprocess.Popen] = {}
_cancelled_jobs: set[str] = set()

# Only MAX_CONCURRENT_PIPELINE_RUNS heavy pipeline runs (GPU/CPU bound) ever
# actually execute at once, but a "Start processing"/redo click beyond that
# no longer just fails outright with "Another job is already processing" -
# it's queued instead, picked up by whichever of the fixed worker pool
# below frees up first. A job waiting its turn sits at status
# "processing"/"finalizing" with stage=None (the same "Starting..." state
# the UI already shows for the brief real gap before the first stage
# begins), and starts for real the moment a worker reaches it - including
# picking up a cancel requested while it was still waiting, since
# _run_stage's very first check is _cancelled_jobs.
_pipeline_queue: "queue.Queue[tuple[str, str, Callable[[], None]]]" = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False

# Mirrors which job_ids are sitting in _pipeline_queue, in order - a plain
# queue.Queue doesn't support peeking at its contents, and the UI needs to
# say *something* more useful than "Starting..." for a job that's actually
# waiting behind others, not just in the brief real gap before its first
# stage begins (see queue_position below, surfaced as JobOut.queue_position).
_queue_state_lock = threading.Lock()
_queued_job_ids: list[str] = []


def queue_position(job_id: str) -> Optional[int]:
    """1-based position in the pipeline queue if job_id is still waiting for
    a worker to pick it up, None once it's actually running (or if it was
    never queued at all)."""
    with _queue_state_lock:
        if job_id not in _queued_job_ids:
            return None
        return _queued_job_ids.index(job_id) + 1


def _worker_loop():
    while True:
        job_id, target_status_on_success, work = _pipeline_queue.get()
        with _queue_state_lock:
            if job_id in _queued_job_ids:
                _queued_job_ids.remove(job_id)
        try:
            work()
            store.update(job_id, status=target_status_on_success, stage=None, error=None)
        except JobCancelled:
            store.update(job_id, status=STATUS_CANCELLED, stage=None, error=None)
        except Exception as exc:  # noqa: BLE001 - surfaced on the job, not swallowed
            traceback.print_exc()
            store.update(job_id, status=STATUS_ERROR, error=str(exc))
        finally:
            _cancelled_jobs.discard(job_id)
            _pipeline_queue.task_done()


def _enqueue(job_id: str, target_status_on_success: str, work: Callable[[], None]):
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            _worker_started = True
            for _ in range(config.MAX_CONCURRENT_PIPELINE_RUNS):
                threading.Thread(target=_worker_loop, daemon=True).start()

    with _queue_state_lock:
        _queued_job_ids.append(job_id)
    _pipeline_queue.put((job_id, target_status_on_success, work))


class JobCancelled(Exception):
    pass


def cancel(job_id: str) -> None:
    """Marks a job cancelled and kills whatever stage subprocess is
    currently running for it, if any. Safe to call even in the brief gap
    between two stages, where no subprocess is registered yet - the next
    _run_stage() call checks _cancelled_jobs before launching anything."""
    with _processes_lock:
        _cancelled_jobs.add(job_id)
        process = _active_processes.get(job_id)

    if process is not None:
        process.terminate()


def _run_stage(job_id: str, stage: str, video_path: Path, output_path: Path):
    if job_id in _cancelled_jobs:
        raise JobCancelled()

    store.update(job_id, stage=stage)

    started = time.monotonic()

    process = subprocess.Popen(
        [sys.executable, "-m", "API.stage_runner", stage, str(video_path), str(output_path)],
        cwd=str(BACKEND_DIR),
        stderr=subprocess.PIPE,
        text=True,
    )

    with _processes_lock:
        _active_processes[job_id] = process

    _, stderr = process.communicate()

    with _processes_lock:
        _active_processes.pop(job_id, None)

    if job_id in _cancelled_jobs:
        raise JobCancelled()

    if process.returncode != 0:
        raise RuntimeError(stderr.strip() or f"Stage '{stage}' failed (exit code {process.returncode})")

    store.mark_stage_complete(job_id, stage, time.monotonic() - started)


def _phase_one(job_id: str, video_path: Path, output_path: Path):
    # Court calibration is a post-processing Setup tab step (see
    # calibration_router.py), so court.json may not exist yet the first time
    # this runs - player_tracking/ball_detection fall back to pixel-space
    # positions/speeds and a full-frame ball search in that case. Once
    # calibration is saved through the web UI, CalibrationPanel.handleSave
    # calls redo/process to re-run this phase against the saved court.json.
    output_path.mkdir(parents=True, exist_ok=True)

    _run_stage(job_id, "player_tracking", video_path, output_path)

    # Best-effort: before anyone's even looked at this video's players,
    # check whether any of them are confidently recognizable from a person
    # already named in a *different* video (see player_gallery.py) and
    # write those names in now. Never fatal - a video with the appearance
    # gallery still empty, or no players tracked at all, just gets zero
    # matches and moves on exactly as before.
    try:
        players.auto_identify_from_gallery(video_path, output_path)
    except Exception:  # noqa: BLE001 - genuinely best-effort, never blocks processing
        traceback.print_exc()

    _run_stage(job_id, "ball_detection", video_path, output_path)
    _run_stage(job_id, "game_status", video_path, output_path)
    _run_stage(job_id, "action_detection", video_path, output_path)


def _phase_two(job_id: str, video_path: Path, output_path: Path):
    _run_stage(job_id, "consolidating", video_path, output_path)
    merge_names_into_stats(output_path)

    _run_stage(job_id, "dashboard", video_path, output_path)
    _run_stage(job_id, "rendering", video_path, output_path)


def start_phase_one(job_id: str):
    video_path = config.find_input_video(job_id)
    if video_path is None:
        store.update(job_id, status=STATUS_ERROR, error="No uploaded video found for this job.")
        return

    output_path = config.output_dir(job_id)
    store.update(job_id, status=STATUS_PROCESSING, error=None)

    # Phase two runs right after phase one instead of waiting on a manual
    # "finalize" click - results should be viewable as soon as processing
    # finishes, even before anyone's assigned player names. Whatever names
    # exist yet (usually none) get stamped in; player_stats.json falls back
    # to "Player <id>" labels until someone visits the Setup tab and
    # re-finalizes with real names.
    def work():
        _phase_one(job_id, video_path, output_path)
        store.update(job_id, status=STATUS_FINALIZING, stage=None)
        _phase_two(job_id, video_path, output_path)

    _enqueue(job_id, STATUS_COMPLETE, work)


def start_phase_two(job_id: str):
    video_path = config.find_input_video(job_id)
    if video_path is None:
        store.update(job_id, status=STATUS_ERROR, error="No uploaded video found for this job.")
        return

    output_path = config.output_dir(job_id)
    store.update(job_id, status=STATUS_FINALIZING, error=None)

    _enqueue(job_id, STATUS_COMPLETE, lambda: _phase_two(job_id, video_path, output_path))


def start_recalibration(job_id: str):
    """
    The cheap path for a calibration change on an already-"complete" job:
    re-picks the ball from its already-saved raw candidates and re-derives
    player court coordinates/auto-ignores against the new court.json (the
    "recalibrate" stage - see stage_runner._recalibrate), then re-runs phase
    two to refresh stats/dashboard/video - all without re-running
    player_tracking's or ball_detection's actual (expensive) detection
    passes. Caller (jobs_router.recalibrate_job) has already confirmed
    ball_candidates.json exists - there's nothing for "recalibrate" to
    re-pick the ball from without it.
    """
    video_path = config.find_input_video(job_id)
    if video_path is None:
        store.update(job_id, status=STATUS_ERROR, error="No uploaded video found for this job.")
        return

    output_path = config.output_dir(job_id)
    store.update(job_id, status=STATUS_FINALIZING, error=None)

    def work():
        _run_stage(job_id, "recalibrate", video_path, output_path)
        _phase_two(job_id, video_path, output_path)

    _enqueue(job_id, STATUS_COMPLETE, work)
