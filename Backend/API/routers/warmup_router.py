from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import calibration, config, warmup
from ..jobs import store
from ..schemas import WarmupConfigIn, WarmupConfigOut, WarmupConfirmIn

router = APIRouter(prefix="/api/jobs/{job_id}/warmup", tags=["warmup"])


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


def _duration_s(job_id: str) -> Optional[float]:
    job = store.get(job_id)
    if job and job.duration_s is not None:
        return job.duration_s
    video_path = config.find_input_video(job_id)
    return calibration.video_duration_s(video_path) if video_path else None


def _config_out(job_id: str, cfg: dict) -> WarmupConfigOut:
    return WarmupConfigOut(job_id=job_id, duration_s=_duration_s(job_id), **cfg)


@router.get("", response_model=WarmupConfigOut)
async def get_warmup(job_id: str):
    _require_job(job_id)
    cfg = warmup.load_config(config.output_dir(job_id))
    return _config_out(job_id, cfg)


# Full-video, unrestricted frame grab for the picker itself to scrub - same
# shape as score_router.get_frame, deliberately not bounded by any
# already-confirmed range, since this is exactly the tool used to set that
# range in the first place.
@router.get("/frame")
async def get_frame(job_id: str, t: Optional[float] = None):
    _require_job(job_id)
    video_path = config.find_input_video(job_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this job")

    try:
        jpeg_bytes, _, _ = calibration.read_calibration_frame(video_path, timestamp_s=t)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.put("", response_model=WarmupConfigOut)
async def set_warmup(job_id: str, body: WarmupConfigIn):
    _require_job(job_id)
    duration_s = _duration_s(job_id)

    if body.start_s < 0:
        raise HTTPException(status_code=400, detail="start_s can't be negative")
    if body.end_s is not None and body.end_s <= body.start_s:
        raise HTTPException(status_code=400, detail="end_s must be after start_s")
    if duration_s is not None and body.start_s >= duration_s:
        raise HTTPException(status_code=400, detail="start_s must be before the video ends")

    output_path = config.output_dir(job_id)
    # Saving a range always marks it confirmed - same one-action save+lock
    # as CourtCalibrationPage's "Set Court Identification" (see
    # calibration.save's docstring for why): a single "Set Warmup Period"
    # button both saves and locks it, rather than needing a separate
    # explicit confirm step for something this simple.
    cfg = warmup.save_config(output_path, {"start_s": body.start_s, "end_s": body.end_s, "confirmed": True})
    return _config_out(job_id, cfg)


@router.put("/confirm", response_model=WarmupConfigOut)
async def set_confirmed(job_id: str, body: WarmupConfirmIn):
    """Separate from PUT /warmup so "Redo Warmup Period" can unlock editing
    (confirmed: false) without resending - and potentially drifting - the
    actual range. Saving a new range (see set_warmup above) always re-locks
    it, same one-action save+lock pattern as Court Calibration."""
    _require_job(job_id)
    output_path = config.output_dir(job_id)
    cfg = warmup.save_config(output_path, {"confirmed": body.confirmed})
    return _config_out(job_id, cfg)
