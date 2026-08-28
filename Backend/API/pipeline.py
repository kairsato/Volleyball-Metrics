import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import config
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
    # Court calibration happens up front through the web UI (see
    # api/calibration.py) instead of the desktop courtDefine() GUI - by the
    # time this runs, court.json already exists and "court_calibration" is
    # already marked complete.
    output_path.mkdir(parents=True, exist_ok=True)

    _run_stage(job_id, "player_tracking", video_path, output_path)
    _run_stage(job_id, "ball_detection", video_path, output_path)
    _run_stage(job_id, "game_status", video_path, output_path)
    _run_stage(job_id, "action_detection", video_path, output_path)


def _phase_two(job_id: str, video_path: Path, output_path: Path):
    _run_stage(job_id, "consolidating", video_path, output_path)
    merge_names_into_stats(output_path)

    _run_stage(job_id, "dashboard", video_path, output_path)
    _run_stage(job_id, "rendering", video_path, output_path)


def _run_locked(job_id: str, target_status_on_success: str, work):
    if not store.pipeline_lock.acquire(blocking=False):
        store.update(job_id, status=STATUS_ERROR, error="Another job is already processing.")
        return

    try:
        work()
        store.update(job_id, status=target_status_on_success, stage=None, error=None)
    except JobCancelled:
        store.update(job_id, status=STATUS_CANCELLED, stage=None, error=None)
    except Exception as exc:
        traceback.print_exc()
        store.update(job_id, status=STATUS_ERROR, error=str(exc))
    finally:
        _cancelled_jobs.discard(job_id)
        store.pipeline_lock.release()


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

    thread = threading.Thread(
        target=_run_locked,
        args=(job_id, STATUS_COMPLETE, work),
        daemon=True,
    )
    thread.start()


def start_phase_two(job_id: str):
    video_path = config.find_input_video(job_id)
    if video_path is None:
        store.update(job_id, status=STATUS_ERROR, error="No uploaded video found for this job.")
        return

    output_path = config.output_dir(job_id)
    store.update(job_id, status=STATUS_FINALIZING, error=None)

    thread = threading.Thread(
        target=_run_locked,
        args=(job_id, STATUS_COMPLETE, lambda: _phase_two(job_id, video_path, output_path)),
        daemon=True,
    )
    thread.start()
