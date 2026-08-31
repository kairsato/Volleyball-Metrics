import json
import mimetypes
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from .. import action_quality, calibration, config, teams
from ..jobs import store
from ..schemas import ActionQualityOut, MatchupOut, QualitiesOut, RallyOut, ResultsOut

# Same "add Analysis's own dir to sys.path, import its subpackages as
# top-level" pattern calibration.py already uses - keeps TRANSCODE_TIERS'
# definition in one place (transcode.py) instead of duplicating it here.
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for _directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from PostProcessing.transcode import TRANSCODE_TIERS, rendition_filename  # noqa: E402

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

    # Reuses the calibration frame grab, aimed at the middle of the video
    # rather than frame 0 - frame 0 is very often a black/loading/warm-up
    # frame with no play visible at all, which made every thumbnail before
    # the game actually started look the same. Falls back to frame 0 (the
    # calibration frame grab's own default) if duration can't be read for
    # some reason, same as before this change.
    timestamp_s = calibration.video_duration_s(video_file)
    try:
        if timestamp_s is not None:
            jpeg_bytes, _, _ = calibration.read_calibration_frame(video_file, timestamp_s=timestamp_s / 2)
        else:
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


@router.get("/action-quality", response_model=ActionQualityOut)
async def get_action_quality(job_id: str):
    _require_job(job_id)

    output_path = config.output_dir(job_id)
    stats_file = output_path / config.STATS_FILE_NAME
    if not stats_file.exists():
        raise HTTPException(status_code=409, detail="Job hasn't finalized yet - consolidate first.")

    result = action_quality.compute_action_quality(job_id, output_path)
    return ActionQualityOut(**result)


@router.get("/dashboard")
async def get_dashboard(job_id: str):
    _require_job(job_id)

    dashboard_file = config.output_dir(job_id) / config.DASHBOARD_FILE_NAME
    if not dashboard_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard not generated yet")

    return FileResponse(dashboard_file, media_type="text/html")


@router.get("/source")
async def get_source_video(job_id: str, quality: Optional[str] = None):
    _require_job(job_id)

    # quality=None (or "original", or a tier that was never generated - a
    # short source, a still-processing job, a job from before this existed)
    # all fall back to the untouched original file, exactly like before this
    # param existed - existing callers that never pass it are unaffected.
    if quality and quality in TRANSCODE_TIERS:
        rendition_file = config.output_dir(job_id) / rendition_filename(quality)
        if rendition_file.exists():
            return FileResponse(rendition_file, media_type="video/mp4", filename=rendition_file.name)

    video_file = config.find_input_video(job_id)
    if video_file is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this job")

    media_type = mimetypes.guess_type(video_file.name)[0] or "application/octet-stream"
    return FileResponse(video_file, media_type=media_type, filename=video_file.name)


@router.get("/qualities", response_model=QualitiesOut)
async def get_qualities(job_id: str):
    _require_job(job_id)
    output_dir = config.output_dir(job_id)
    generated = [tier for tier in TRANSCODE_TIERS if (output_dir / rendition_filename(tier)).exists()]
    return QualitiesOut(qualities=["original", *generated])


@router.get("/video")
async def get_video(job_id: str):
    _require_job(job_id)

    video_file = config.output_dir(job_id) / config.ANNOTATED_VIDEO_NAME
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Annotated video not rendered yet")

    return FileResponse(video_file, media_type="video/mp4", filename=config.ANNOTATED_VIDEO_NAME)
