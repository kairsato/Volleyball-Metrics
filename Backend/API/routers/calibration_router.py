from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import calibration, config
from ..jobs import store
from ..schemas import CalibrationIn, CalibrationPointsOut, Point

router = APIRouter(prefix="/api/jobs/{job_id}/calibration", tags=["calibration"])


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


def _video_or_404(job_id: str) -> Path:
    video_path = config.find_input_video(job_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this job")
    return video_path


@router.get("/frame")
async def get_frame(job_id: str):
    _require_job(job_id)
    video_path = _video_or_404(job_id)

    try:
        jpeg_bytes, _, _ = calibration.read_calibration_frame(video_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("", response_model=CalibrationPointsOut)
async def get_points(job_id: str):
    _require_job(job_id)
    video_path = _video_or_404(job_id)

    output_path = config.output_dir(job_id)
    existing = calibration.existing_points(output_path)

    if existing is not None:
        return CalibrationPointsOut(
            job_id=job_id,
            corners=[Point(**p) for p in existing["corners"]],
            net_points=[Point(**p) for p in existing["net_points"]],
            calibrated=True,
        )

    try:
        _, width, height = calibration.read_calibration_frame(video_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    defaults = calibration.default_points(width, height)
    return CalibrationPointsOut(
        job_id=job_id,
        corners=[Point(**p) for p in defaults["corners"]],
        net_points=[Point(**p) for p in defaults["net_points"]],
        calibrated=False,
    )


@router.put("", response_model=CalibrationPointsOut)
async def set_points(job_id: str, body: CalibrationIn):
    _require_job(job_id)
    output_path = config.output_dir(job_id)

    try:
        calibration.save(
            output_path,
            [(p.x, p.y) for p in body.corners],
            [(p.x, p.y) for p in body.net_points],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.mark_stage_complete(job_id, "court_calibration")

    return CalibrationPointsOut(
        job_id=job_id,
        corners=body.corners,
        net_points=body.net_points,
        calibrated=True,
    )
