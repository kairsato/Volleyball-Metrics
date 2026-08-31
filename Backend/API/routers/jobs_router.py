import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from .. import calibration, config, pipeline, players, score
from ..jobs import (
    STATUS_AWAITING_PLAYER_REVIEW,
    STATUS_CANCELLED,
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_FINALIZING,
    STATUS_PROCESSING,
    STATUS_UPLOADED,
    Job,
    store,
)
from ..schemas import JobOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _get_job_or_404(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _ensure_duration(job: Job) -> Job:
    """Backfills duration_s for a job uploaded before that field existed (or
    if reading it failed at upload time) - lazily, on whichever request
    happens to touch this job next, rather than needing a one-off migration
    script. A job that already has a duration never re-opens its video."""
    if job.duration_s is not None:
        return job
    video_path = config.find_input_video(job.id)
    if video_path is None:
        return job
    duration_s = calibration.video_duration_s(video_path)
    if duration_s is None:
        return job
    return store.update(job.id, duration_s=round(duration_s, 1))


def _needs_player_id(output_path: Path) -> bool:
    """Mirrors score.compute_summary's needs_review: reflects whether the
    user has explicitly confirmed player identification (see
    players.load_player_confirmed), not just "is anyone currently
    unnamed" - confirming is itself gated on nobody being unresolved (see
    the frontend), so "confirmed" is the stronger signal of the two: not
    just "nothing's unnamed right now" but "a human actually looked at
    this and signed off". A video with no detected players at all needs no
    review regardless."""
    stats_file = output_path / config.STATS_FILE_NAME
    if not stats_file.exists():
        return False
    stats = json.loads(stats_file.read_text())
    if not stats.get("players"):
        return False
    return not players.load_player_confirmed(output_path)


def _job_out(job: Job) -> JobOut:
    job = _ensure_duration(job)

    needs_player_id = False
    needs_scoring_review = False
    winner_team_name = None
    if job.status == STATUS_COMPLETE:
        output_path = config.output_dir(job.id)
        needs_player_id = _needs_player_id(output_path)
        summary = score.compute_summary(output_path)
        needs_scoring_review = summary["needs_review"]
        winner_team_name = summary["winner_team_name"]

    return JobOut(
        **job.as_dict(),
        needs_player_id=needs_player_id,
        needs_scoring_review=needs_scoring_review,
        winner_team_name=winner_team_name,
        queue_position=pipeline.queue_position(job.id),
    )


@router.post("", response_model=JobOut)
async def upload_video(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_VIDEO_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    job = store.create(original_filename=file.filename or "video")

    destination = config.input_video_path(job.id, suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    return _job_out(job)


@router.get("", response_model=list[JobOut])
async def list_jobs():
    return [_job_out(job) for job in store.list()]


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    return _job_out(_get_job_or_404(job_id))


@router.post("/{job_id}/process", response_model=JobOut)
async def process_job(job_id: str):
    # Court calibration is no longer required up front - it's a post-
    # processing Setup tab step like player identification and scoring, not
    # a gate on starting the pipeline at all. Player/ball tracking still use
    # court.json *while tracking runs* if it exists yet (see
    # tracker_offline.py/ballDetection.py), but both fall back gracefully
    # (pixel-space positions/speeds, full-frame ball search) when it
    # doesn't - calibrating afterward means recalibrating the job (see
    # recalibrate_job below) to get real-world court coordinates
    # retroactively, which CalibrationPanel.handleSave (frontend) triggers
    # automatically on save.
    job = _get_job_or_404(job_id)

    if job.status not in (STATUS_UPLOADED, STATUS_ERROR, STATUS_CANCELLED):
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")

    pipeline.start_phase_one(job_id)
    return _job_out(store.get(job_id))


@router.post("/{job_id}/finalize", response_model=JobOut)
async def finalize_job(job_id: str):
    job = _get_job_or_404(job_id)

    # STATUS_COMPLETE is allowed too - re-finalizing lets a completed job's
    # player names/groupings be edited later and the stats, dashboard and
    # video regenerated from the same tracking data, without re-running the
    # expensive tracking/detection stages.
    if job.status not in (STATUS_AWAITING_PLAYER_REVIEW, STATUS_ERROR, STATUS_CANCELLED, STATUS_COMPLETE):
        raise HTTPException(
            status_code=409,
            detail="Job must finish processing (awaiting player review) before it can be finalized",
        )

    pipeline.start_phase_two(job_id)
    return _job_out(store.get(job_id))


def _reset_for_full_reprocess(job_id: str, output_path: Path) -> Job:
    """Shared by redo_job and recalibrate_job's old-job fallback: any
    existing player names/ignores are cleared rather than carried forward -
    a fresh tracking run assigns new stable_ids, so the old name-to-id
    mapping would silently apply to the wrong people."""
    (output_path / config.PLAYER_NAMES_NAME).unlink(missing_ok=True)
    (output_path / config.PLAYER_IGNORED_NAME).unlink(missing_ok=True)

    return store.update(
        job_id,
        status=STATUS_UPLOADED,
        stage=None,
        completed_stages=[],
        stage_durations_s={},
        error=None,
    )


@router.post("/{job_id}/redo", response_model=JobOut)
async def redo_job(job_id: str):
    """
    Resets a completed job back to "uploaded" so everything can be
    reprocessed from scratch. Prefer recalibrate_job below for a plain
    calibration change on a job that already has ball_candidates.json - this
    full reset is now mainly the fallback path for jobs that predate it.
    """
    job = _get_job_or_404(job_id)

    if job.status != STATUS_COMPLETE:
        raise HTTPException(status_code=409, detail="Only a completed job can be redone")

    updated = _reset_for_full_reprocess(job_id, config.output_dir(job_id))
    return _job_out(updated)


@router.post("/{job_id}/recalibrate", response_model=JobOut)
async def recalibrate_job(job_id: str):
    """
    Triggered after saving a new court calibration (see
    calibration_router.set_points) on an already-complete job. Cheap path:
    re-picks the ball from its saved raw candidates and re-derives player
    court coordinates/auto-ignores, then re-runs action_detection and phase
    two - without re-running player_tracking's or ball_detection's actual
    (expensive) detection passes (see pipeline.start_recalibration). Falls
    back to a full reprocess for a job whose ball detection ran before
    ball_candidates.json existed - there's nothing for the cheap path to
    re-pick the ball from without it.
    """
    job = _get_job_or_404(job_id)

    if job.status != STATUS_COMPLETE:
        raise HTTPException(status_code=409, detail="Only a completed job can be recalibrated")

    output_path = config.output_dir(job_id)
    if not (output_path / config.BALL_CANDIDATES_FILE_NAME).exists():
        _reset_for_full_reprocess(job_id, output_path)
        pipeline.start_phase_one(job_id)
        return _job_out(store.get(job_id))

    pipeline.start_recalibration(job_id)
    return _job_out(store.get(job_id))


@router.post("/{job_id}/cancel", response_model=JobOut)
async def cancel_job(job_id: str):
    job = _get_job_or_404(job_id)

    if job.status not in (STATUS_PROCESSING, STATUS_FINALIZING):
        raise HTTPException(status_code=409, detail="Job isn't currently running")

    pipeline.cancel(job_id)
    return _job_out(store.get(job_id))


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str):
    job = _get_job_or_404(job_id)

    if job.status in (STATUS_PROCESSING, STATUS_FINALIZING):
        raise HTTPException(status_code=409, detail="Cancel the job before deleting it")

    store.delete(job_id)
