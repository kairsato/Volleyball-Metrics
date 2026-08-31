import json
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config

# Ordered so the frontend can render a progress list. Court calibration
# used to run up front and lead this list; it's now a post-processing
# Setup tab step (see calibration_router.py) that doesn't run as part of
# processing, so it's no longer one of these stages.
PHASE_ONE_STAGES = [
    "player_tracking",
    "ball_detection",
    "game_status",
    "action_detection",
]
PHASE_TWO_STAGES = ["consolidating", "dashboard", "rendering", "transcoding"]
ALL_STAGES = PHASE_ONE_STAGES + PHASE_TWO_STAGES

STATUS_UPLOADED = "uploaded"
STATUS_PROCESSING = "processing"
STATUS_AWAITING_PLAYER_REVIEW = "awaiting_player_review"
STATUS_FINALIZING = "finalizing"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    original_filename: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    status: str = STATUS_UPLOADED
    stage: Optional[str] = None
    completed_stages: list = field(default_factory=list)
    # stage name -> how long it took to run, in seconds. Populated as each
    # stage finishes so a slow run can be attributed to a specific stage
    # (e.g. player_tracking) instead of just a slow job overall.
    stage_durations_s: dict = field(default_factory=dict)
    error: Optional[str] = None
    # The uploaded video's own length, not a processing duration - computed
    # once (see jobs_router._ensure_duration) and cached here so the video
    # list never has to re-open the file just to show it. None for jobs
    # uploaded before this field existed, or if the video's length couldn't
    # be read at all; either way it's backfilled the next time the job is
    # listed or fetched.
    duration_s: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


class JobStore:
    """In-memory job registry, mirrored to a job.json file per job so state
    survives a server restart even though an in-flight pipeline run does not
    (the caller has to re-trigger /process if the server restarted mid-run)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._load_existing()

    def _load_existing(self):
        if not config.DATA_DIR.exists():
            return
        for entry in config.DATA_DIR.iterdir():
            meta_file = entry / config.JOB_METADATA_NAME
            if meta_file.exists():
                try:
                    data = json.loads(meta_file.read_text())
                    job = Job(**data)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

                # A fresh process can't have a background thread still
                # working on a job that was mid-run when the server last
                # stopped (a crash, or a --reload restart) - surface that as
                # an error rather than leaving it stuck with no way to retry.
                if job.status in (STATUS_PROCESSING, STATUS_FINALIZING):
                    job.status = STATUS_ERROR
                    job.stage = None
                    job.error = "Processing was interrupted (the server restarted mid-run). Retry to resume."
                    self._persist(job)

                self._jobs[job.id] = job

    def _persist(self, job: Job):
        meta_file = config.job_dir(job.id) / config.JOB_METADATA_NAME
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.write_text(json.dumps(job.as_dict(), indent=2))

    def create(self, original_filename: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, original_filename=original_filename)
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update(self, job_id: str, **changes) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()
            self._persist(job)
            return job

    def mark_stage_complete(self, job_id: str, stage: str, duration_s: Optional[float] = None) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            if stage not in job.completed_stages:
                job.completed_stages.append(stage)
            # court_calibration completes synchronously from a user API call
            # with no subprocess to time - there's nothing meaningful to
            # record for it (it would just measure how long the user took to
            # click corners, not processing time), so it's the one stage
            # that's marked complete without a duration.
            if duration_s is not None:
                job.stage_durations_s[stage] = round(duration_s, 1)
            job.updated_at = _now()
            self._persist(job)
            return job

    def delete(self, job_id: str):
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(config.job_dir(job_id), ignore_errors=True)


store = JobStore()
