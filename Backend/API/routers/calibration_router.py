from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import calibration, config
from ..jobs import store
from ..schemas import CalibrationConfirmIn, CalibrationIn, CalibrationPointsOut, Point

router = APIRouter(prefix="/api/jobs/{job_id}/calibration", tags=["calibration"])


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


def _video_or_404(job_id: str) -> Path:
    video_path = config.find_input_video(job_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this job")
    return video_path


def _points_out(job_id: str, existing: dict) -> CalibrationPointsOut:
    predicted = existing.get("predicted")
    return CalibrationPointsOut(
        job_id=job_id,
        middle_left=Point(**existing["middle_left"]),
        middle_right=Point(**existing["middle_right"]),
        far_left=Point(**existing["far_left"]),
        far_right=Point(**existing["far_right"]),
        net_top_left=Point(**existing["net_top_left"]),
        net_top_right=Point(**existing["net_top_right"]),
        net_height_m=existing["net_height_m"],
        calibrated=True,
        confirmed=existing["confirmed"],
        net_top_calibrated=existing["net_top_calibrated"],
        camera_pose_available=existing["camera_pose_available"],
        camera_pose_reprojection_error_px=existing["camera_pose_reprojection_error_px"],
        predicted={name: Point(**p) for name, p in predicted.items()} if predicted else None,
    )


@router.get("/frame")
async def get_frame(job_id: str):
    _require_job(job_id)
    video_path = _video_or_404(job_id)
    output_path = config.output_dir(job_id)

    try:
        jpeg_bytes, _, _ = calibration.build_clean_court_frame(video_path, output_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("", response_model=CalibrationPointsOut)
async def get_points(job_id: str):
    _require_job(job_id)
    video_path = _video_or_404(job_id)
    output_path = config.output_dir(job_id)

    width, height = calibration.frame_size(video_path)
    existing = calibration.existing_points(output_path, width, height)

    if existing is not None:
        return _points_out(job_id, existing)

    defaults = calibration.default_points(width, height)
    return CalibrationPointsOut(
        job_id=job_id,
        middle_left=Point(**defaults["middle_left"]),
        middle_right=Point(**defaults["middle_right"]),
        far_left=Point(**defaults["far_left"]),
        far_right=Point(**defaults["far_right"]),
        net_top_left=Point(**defaults["net_top_left"]),
        net_top_right=Point(**defaults["net_top_right"]),
        net_height_m=defaults["net_height_m"],
        calibrated=False,
        confirmed=False,
        net_top_calibrated=False,
        camera_pose_available=None,
        camera_pose_reprojection_error_px=None,
    )


@router.put("", response_model=CalibrationPointsOut)
async def set_points(job_id: str, body: CalibrationIn):
    _require_job(job_id)
    video_path = _video_or_404(job_id)
    output_path = config.output_dir(job_id)
    width, height = calibration.frame_size(video_path)

    try:
        calibration.save(
            output_path,
            (body.middle_left.x, body.middle_left.y),
            (body.middle_right.x, body.middle_right.y),
            (body.far_left.x, body.far_left.y),
            (body.far_right.x, body.far_right.y),
            (body.net_top_left.x, body.net_top_left.y),
            (body.net_top_right.x, body.net_top_right.y),
            body.net_height_m,
            width,
            height,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.mark_stage_complete(job_id, "court_calibration")

    # Re-read rather than build the response from `body` + the raw save()
    # result directly - existing_points is the single source of truth for
    # how net_top_calibrated/camera_pose_available get derived, and this
    # keeps that logic in one place instead of duplicating it here.
    existing = calibration.existing_points(output_path, width, height)
    return _points_out(job_id, existing)


@router.put("/confirm", response_model=CalibrationPointsOut)
async def set_confirmed(job_id: str, body: CalibrationConfirmIn):
    """Separate from PUT /calibration so "Redo Court Identification" can
    just unlock editing without needing to resend (and potentially
    overwrite) the actual points."""
    _require_job(job_id)
    video_path = _video_or_404(job_id)
    output_path = config.output_dir(job_id)
    width, height = calibration.frame_size(video_path)

    try:
        calibration.set_confirmed(output_path, body.confirmed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = calibration.existing_points(output_path, width, height)
    if existing is None:
        raise HTTPException(status_code=404, detail="No calibration saved yet")

    return _points_out(job_id, existing)
