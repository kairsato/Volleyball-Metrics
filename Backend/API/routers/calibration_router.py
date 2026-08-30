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
    existing = calibration.existing_points(output_path)

    if existing is not None:
        predicted = existing.get("predicted")
        return CalibrationPointsOut(
            job_id=job_id,
            middle_left=Point(**existing["middle_left"]),
            middle_right=Point(**existing["middle_right"]),
            far_left=Point(**existing["far_left"]),
            far_right=Point(**existing["far_right"]),
            net_height_m=existing["net_height_m"],
            calibrated=True,
            confirmed=existing["confirmed"],
            # None for a job whose calibration predates this field (see
            # calibration.existing_points' legacy fallback) - left as None,
            # not coerced to an empty dict, since the frontend treats
            # "predicted is truthy" as "safe to read predicted.near_left
            # etc." and a `{}` would pass that check only to then crash on
            # the missing fields.
            predicted={name: Point(**p) for name, p in predicted.items()} if predicted else None,
        )

    try:
        _, width, height = calibration.read_calibration_frame(video_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    defaults = calibration.default_points(width, height)
    return CalibrationPointsOut(
        job_id=job_id,
        middle_left=Point(**defaults["middle_left"]),
        middle_right=Point(**defaults["middle_right"]),
        far_left=Point(**defaults["far_left"]),
        far_right=Point(**defaults["far_right"]),
        net_height_m=defaults["net_height_m"],
        calibrated=False,
        confirmed=False,
    )


@router.put("", response_model=CalibrationPointsOut)
async def set_points(job_id: str, body: CalibrationIn):
    _require_job(job_id)
    output_path = config.output_dir(job_id)

    try:
        data = calibration.save(
            output_path,
            (body.middle_left.x, body.middle_left.y),
            (body.middle_right.x, body.middle_right.y),
            (body.far_left.x, body.far_left.y),
            (body.far_right.x, body.far_right.y),
            body.net_height_m,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.mark_stage_complete(job_id, "court_calibration")

    return CalibrationPointsOut(
        job_id=job_id,
        middle_left=body.middle_left,
        middle_right=body.middle_right,
        far_left=body.far_left,
        far_right=body.far_right,
        net_height_m=body.net_height_m,
        calibrated=True,
        confirmed=True,
        predicted={name: Point(**p) for name, p in data["predicted"].items()},
    )


@router.put("/confirm", response_model=CalibrationPointsOut)
async def set_confirmed(job_id: str, body: CalibrationConfirmIn):
    """Separate from PUT /calibration so "Redo Court Identification" can
    just unlock editing without needing to resend (and potentially
    overwrite) the actual points."""
    _require_job(job_id)
    output_path = config.output_dir(job_id)

    try:
        calibration.set_confirmed(output_path, body.confirmed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = calibration.existing_points(output_path)
    if existing is None:
        raise HTTPException(status_code=404, detail="No calibration saved yet")

    predicted = existing.get("predicted")
    return CalibrationPointsOut(
        job_id=job_id,
        middle_left=Point(**existing["middle_left"]),
        middle_right=Point(**existing["middle_right"]),
        far_left=Point(**existing["far_left"]),
        far_right=Point(**existing["far_right"]),
        net_height_m=existing["net_height_m"],
        calibrated=True,
        confirmed=existing["confirmed"],
        predicted={name: Point(**p) for name, p in predicted.items()} if predicted else None,
    )
