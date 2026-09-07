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


def now_iso() -> str:
    """Current UTC time, ISO 8601 - every timestamp this store (and
    pipeline.py, the one other module that needs one - see Job.processed_at)
    writes goes through this, so they're always in the same format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    original_filename: str
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    status: str = STATUS_UPLOADED
    stage: Optional[str] = None
    completed_stages: list = field(default_factory=list)
    # stage name -> how long it took to run, in seconds. Populated as each
    # stage finishes so a slow run can be attributed to a specific stage
    # (e.g. player_tracking) instead of just a slow job overall.
    stage_durations_s: dict = field(default_factory=dict)
    # When the pipeline last actually finished running (whichever of
    # start_phase_one/start_phase_two/start_recalibration it was - see
    # pipeline.py's _worker_loop, the one success point all three funnel
    # through). Deliberately separate from updated_at, which bumps on
    # essentially every write this store makes (a calibration save, a player
    # rename, a score correction) and so can't tell "the analysis itself ran
    # at this time" apart from "someone edited something about this job
    # afterward." None for a job that's never completed a run, or one
    # processed before this field existed.
    processed_at: Optional[str] = None
    error: Optional[str] = None
    # The uploaded video's own length, not a processing duration - computed
    # once (see jobs_router._ensure_duration) and cached here so the video
    # list never has to re-open the file just to show it. None for jobs
    # uploaded before this field existed, or if the video's length couldn't
    # be read at all; either way it's backfilled the next time the job is
    # listed or fetched. For a multi-video job this mirrors segment 0's own
    # duration_s, same as original_filename/date_played below - see
    # job_videos().
    duration_s: Optional[float] = None
    # Ordered video segments making up this job - {id, order,
    # original_filename, suffix, duration_s, date_played} per entry. Always
    # populated by JobStore.create from here on (even a single-video job
    # gets a length-1 list); empty only for a job.json written before
    # multi-video support existed. Call sites should go through
    # job_videos(job) below rather than reading this field directly, so
    # that pre-existing-job case is handled in exactly one place.
    videos: list = field(default_factory=list)
    # Segment 0's own date_played, mirrored here (same reasoning as
    # original_filename/duration_s above) so job listings/search don't need
    # to reach into `videos` for the common case. User-provided at upload
    # time, defaulting to the video's own recording-date metadata when
    # readable, else the upload date - see video_metadata.py.
    date_played: Optional[str] = None
    # Only meaningful for a job with more than one video. Defaults to True
    # for a multi-video job at creation (JobStore.create) and stays False,
    # unused, for a single-video job. See calibration_router.py /
    # CalibrationPanel.tsx's "segmented court selections" toggle.
    segmented_calibration: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def job_videos(job: Job) -> list[dict]:
    """The job's video segments, in order - always use this instead of
    reading job.videos directly. A job.json written before multi-video
    support existed round-trips through Job(**data) with videos still at
    its [] default, so this synthesizes the one implicit segment such a job
    always had, from its own legacy top-level fields, rather than making
    every segment-aware call site special-case "no videos list" itself."""
    if job.videos:
        return job.videos
    return [
        {
            "id": "segment-0",
            "order": 0,
            "original_filename": job.original_filename,
            "suffix": None,
            "duration_s": job.duration_s,
            "date_played": job.date_played,
        }
    ]


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

    def create(self, videos: list[dict]) -> Job:
        """videos: one dict per uploaded file, in upload order - each
        {original_filename, suffix, duration_s, date_played}. Segment 0's
        own values get mirrored onto the job's top-level fields (the same
        fields every existing single-video consumer already reads), and
        segmented_calibration defaults on iff there's more than one video."""
        job_id = uuid.uuid4().hex[:12]
        segments = [
            {
                "id": "segment-0" if i == 0 else uuid.uuid4().hex[:12],
                "order": i,
                "original_filename": v["original_filename"],
                "suffix": v["suffix"],
                "duration_s": v.get("duration_s"),
                "date_played": v.get("date_played"),
            }
            for i, v in enumerate(videos)
        ]
        first = segments[0]
        job = Job(
            id=job_id,
            original_filename=first["original_filename"],
            videos=segments,
            date_played=first["date_played"],
            duration_s=first["duration_s"],
            segmented_calibration=len(segments) > 1,
        )
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
            job.updated_at = now_iso()
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
            job.updated_at = now_iso()
            self._persist(job)
            return job

    def delete(self, job_id: str):
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(config.job_dir(job_id), ignore_errors=True)


store = JobStore()
