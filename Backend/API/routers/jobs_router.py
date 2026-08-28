import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from .. import config, pipeline
from ..jobs import (
    STATUS_AWAITING_PLAYER_REVIEW,
    STATUS_CANCELLED,
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_FINALIZING,
    STATUS_PROCESSING,
    STATUS_UPLOADED,
    store,
)
from ..schemas import JobOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _get_job_or_404(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


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

    return JobOut(**job.as_dict())


@router.get("", response_model=list[JobOut])
async def list_jobs():
    return [JobOut(**job.as_dict()) for job in store.list()]


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    return JobOut(**_get_job_or_404(job_id).as_dict())


@router.post("/{job_id}/process", response_model=JobOut)
async def process_job(job_id: str):
    job = _get_job_or_404(job_id)

    if job.status not in (STATUS_UPLOADED, STATUS_ERROR, STATUS_CANCELLED):
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")

    if not (config.output_dir(job_id) / config.COURT_FILE_NAME).exists():
        raise HTTPException(status_code=409, detail="Court calibration is required before processing")

    pipeline.start_phase_one(job_id)
    return JobOut(**store.get(job_id).as_dict())


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
    return JobOut(**store.get(job_id).as_dict())


@router.post("/{job_id}/cancel", response_model=JobOut)
async def cancel_job(job_id: str):
    job = _get_job_or_404(job_id)

    if job.status not in (STATUS_PROCESSING, STATUS_FINALIZING):
        raise HTTPException(status_code=409, detail="Job isn't currently running")

    pipeline.cancel(job_id)
    return JobOut(**store.get(job_id).as_dict())


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str):
    job = _get_job_or_404(job_id)

    if job.status in (STATUS_PROCESSING, STATUS_FINALIZING):
        raise HTTPException(status_code=409, detail="Cancel the job before deleting it")

    store.delete(job_id)
