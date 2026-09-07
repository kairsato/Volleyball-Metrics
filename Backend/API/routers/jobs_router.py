import json
import shutil
from datetime import date
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import calibration, config, pipeline, players, score, video_metadata, warmup
from ..jobs import (
    STATUS_AWAITING_PLAYER_REVIEW,
    STATUS_CANCELLED,
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_FINALIZING,
    STATUS_PROCESSING,
    STATUS_UPLOADED,
    Job,
    job_videos,
    store,
)
from ..schemas import JobOut, VideoDateIn

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _get_job_or_404(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _ensure_duration(job: Job) -> Job:
    """Backfills duration_s for any segment that doesn't have one yet (a job
    uploaded before this field existed, before multi-video support existed,
    or where reading it failed at upload time) - lazily, on whichever
    request happens to touch this job next, rather than needing a one-off
    migration script. A job whose every segment already has a duration
    never re-opens any video. Only ever writes job.videos back for a job
    that already has a real one (see job_videos()) - a legacy single-video
    job just gets its top-level duration_s backfilled exactly as before,
    with no videos list added to its job.json."""
    segments = job_videos(job)
    updated = []
    changed = False
    for segment in segments:
        if segment.get("duration_s") is not None:
            updated.append(segment)
            continue
        video_path, _ = config.resolve_video_and_output(job.id, segment)
        duration_s = calibration.video_duration_s(video_path) if video_path else None
        if duration_s is None:
            updated.append(segment)
            continue
        updated.append({**segment, "duration_s": round(duration_s, 1)})
        changed = True

    if not changed:
        return job

    changes = {"duration_s": updated[0]["duration_s"]}
    if job.videos:
        changes["videos"] = updated
    return store.update(job.id, **changes)


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
    warmup_confirmed = False
    warmup_start_s = None
    warmup_end_s = None
    if job.status == STATUS_COMPLETE:
        output_path = config.output_dir(job.id)
        needs_player_id = _needs_player_id(output_path)
        summary = score.compute_summary(output_path)
        needs_scoring_review = summary["needs_review"]
        winner_team_name = summary["winner_team_name"]

        warmup_cfg = warmup.load_config(output_path)
        warmup_confirmed = warmup.is_active(warmup_cfg)
        if warmup_confirmed:
            warmup_start_s, warmup_end_s = warmup.effective_range(warmup_cfg, job.duration_s)

    return JobOut(
        **job.as_dict(),
        needs_player_id=needs_player_id,
        needs_scoring_review=needs_scoring_review,
        winner_team_name=winner_team_name,
        warmup_confirmed=warmup_confirmed,
        warmup_start_s=warmup_start_s,
        warmup_end_s=warmup_end_s,
        queue_position=pipeline.queue_position(job.id),
    )


@router.post("", response_model=JobOut)
async def upload_video(files: list[UploadFile] = File(...)):
    """One atomic request creates the whole job with every segment it will
    ever have - there's no separate "add another video to this job later"
    flow (see the multi-video plan's scope decision). files[0] becomes
    segment 0 (the job's own original_filename/duration_s/date_played, and
    the one that lives at the legacy flat data/<job_id>/input<ext> path -
    see config.resolve_video_and_output); files[1:] become segments 1..N
    under data/<job_id>/videos/<segment_id>/."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    video_specs = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in config.ALLOWED_VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(config.ALLOWED_VIDEO_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for {file.filename!r}. Allowed: {allowed}",
            )
        video_specs.append({"original_filename": file.filename or "video", "suffix": suffix})

    # Segment ids/paths only exist once the Job record itself does, so
    # files are streamed to disk in a second pass, then each one's
    # date_played is extracted from the now-on-disk file and patched back -
    # same lazy-backfill shape as _ensure_duration above, just done
    # eagerly here since upload is the one moment a sensible default is
    # actually worth computing.
    job = store.create(video_specs)
    segments = job_videos(job)

    updated_segments = []
    for file, segment in zip(files, segments):
        video_path = config.segment_destination(job.id, segment)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with video_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        played = video_metadata.extract_creation_date(video_path) or date.today()
        updated_segments.append({**segment, "date_played": played.isoformat()})

    job = store.update(job.id, videos=updated_segments, date_played=updated_segments[0]["date_played"])

    return _job_out(job)


@router.put("/{job_id}/videos/{segment_id}", response_model=JobOut)
async def set_video_date(job_id: str, segment_id: str, body: VideoDateIn):
    """Corrects one segment's date_played after upload (auto-detected from
    file metadata or defaulted to the upload date - see upload_video). The
    only user-editable field on a video segment today."""
    job = _get_job_or_404(job_id)
    segments = job_videos(job)
    if not any(s["id"] == segment_id for s in segments):
        raise HTTPException(status_code=404, detail="Video segment not found")

    updated_segments = [
        {**s, "date_played": body.date_played} if s["id"] == segment_id else s for s in segments
    ]
    changes: dict = {"videos": updated_segments}
    if segment_id == updated_segments[0]["id"]:
        changes["date_played"] = body.date_played
    job = store.update(job_id, **changes)
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
