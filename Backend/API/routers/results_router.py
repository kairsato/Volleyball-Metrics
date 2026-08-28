import json
import mimetypes

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from .. import calibration, config, teams
from ..jobs import store
from ..schemas import MatchupOut, RallyOut, ResultsOut

router = APIRouter(prefix="/api/jobs/{job_id}", tags=["results"])


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/thumbnail")
async def get_thumbnail(job_id: str):
    _require_job(job_id)

    video_file = config.find_input_video(job_id)
    if video_file is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this job")

    try:
        # Reuses the calibration frame grab - it's just "frame 0, JPEG
        # encoded" and works regardless of the job's calibration state.
        jpeg_bytes, _, _ = calibration.read_calibration_frame(video_file)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/results", response_model=ResultsOut)
async def get_results(job_id: str):
    _require_job(job_id)

    output_path = config.output_dir(job_id)
    stats_file = output_path / config.STATS_FILE_NAME

    if not stats_file.exists():
        raise HTTPException(status_code=409, detail="Job hasn't finalized yet - consolidate first.")

    stats = json.loads(stats_file.read_text())

    rallies: list[RallyOut] = []
    status_file = output_path / config.GAME_STATUS_FILE_NAME
    if status_file.exists():
        status = json.loads(status_file.read_text())
        rallies = [RallyOut(**rally) for rally in status.get("rallies", [])]

    return ResultsOut(
        job_id=job_id,
        players=stats.get("players", {}),
        rallies=rallies,
        caveats=stats.get("caveats", []),
        dashboard_available=(output_path / config.DASHBOARD_FILE_NAME).exists(),
        video_available=(output_path / config.ANNOTATED_VIDEO_NAME).exists(),
    )


@router.get("/matchup", response_model=MatchupOut)
async def get_matchup(job_id: str):
    _require_job(job_id)

    output_path = config.output_dir(job_id)
    stats_file = output_path / config.STATS_FILE_NAME
    if not stats_file.exists():
        raise HTTPException(status_code=409, detail="Job hasn't finalized yet - consolidate first.")

    matchup = teams.build_matchup(output_path)
    return MatchupOut(job_id=job_id, **matchup)


@router.get("/dashboard")
async def get_dashboard(job_id: str):
    _require_job(job_id)

    dashboard_file = config.output_dir(job_id) / config.DASHBOARD_FILE_NAME
    if not dashboard_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard not generated yet")

    return FileResponse(dashboard_file, media_type="text/html")


@router.get("/source")
async def get_source_video(job_id: str):
    _require_job(job_id)

    video_file = config.find_input_video(job_id)
    if video_file is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this job")

    media_type = mimetypes.guess_type(video_file.name)[0] or "application/octet-stream"
    return FileResponse(video_file, media_type=media_type, filename=video_file.name)


@router.get("/video")
async def get_video(job_id: str):
    _require_job(job_id)

    video_file = config.output_dir(job_id) / config.ANNOTATED_VIDEO_NAME
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Annotated video not rendered yet")

    return FileResponse(video_file, media_type="video/mp4", filename=config.ANNOTATED_VIDEO_NAME)
