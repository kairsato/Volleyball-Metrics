import json
import mimetypes
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from .. import action_quality, calibration, config, players, teams, warmup
from ..jobs import store
from ..schemas import (
    ActionQualityOut,
    BallTrajectoryOut,
    BallTrajectoryPointOut,
    MatchupOut,
    PlayerBoxOut,
    PlayerTrajectoryFrameOut,
    PlayerTrajectoryOut,
    QualitiesOut,
    RallyOut,
    ResultsOut,
)

# Same "add Analysis's own dir to sys.path, import its subpackages as
# top-level" pattern calibration.py already uses - keeps TRANSCODE_TIERS'
# definition in one place (transcode.py) instead of duplicating it here.
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for _directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from BallDetection.ballDetection import SPEED_LOG_NAME, TRAJECTORY_LOG_NAME, load_homography  # noqa: E402
from CourtDefinition.court import COURT_LENGTH, COURT_WIDTH, NET_HEIGHT_M, pixel_to_court  # noqa: E402
from PostProcessing.transcode import TRANSCODE_TIERS, rendition_filename  # noqa: E402

router = APIRouter(prefix="/api/jobs/{job_id}", tags=["results"])

# Keeps roughly 1 in every this-many tracked frames when building the ball
# trajectory below. Full density (1) so the on-video ball-tracking overlay
# actually shows the tracker's own real per-frame positions rather than
# gliding a straight line between samples half a second apart - a real
# rally changes the ball's direction/speed multiple times within that
# window, so anything coarser reads as prediction, not tracking, no matter
# how accurate the underlying per-frame data actually is. A few MB per
# results-page load is a non-issue for a self-hosted app on a local
# network; if a very long match's payload ever becomes a real problem,
# raise this back up for BallMinimap specifically (which never needed
# frame-accuracy - see BallMinimap.tsx) rather than for this on-video
# overlay.
BALL_TRAJECTORY_STRIDE = 1
DEFAULT_FPS_ASSUMPTION = 30.0


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
    # (or the middle of the confirmed warmup range, once one exists - the
    # whole reason that range exists is to skip exactly this kind of
    # black/loading/warm-up frame) rather than frame 0, which is very often
    # a dead frame with no play visible at all. Falls back to frame 0 (the
    # calibration frame grab's own default) if duration can't be read for
    # some reason, same as before this change.
    duration_s = calibration.video_duration_s(video_file)
    warmup_cfg = warmup.load_config(config.output_dir(job_id))
    if warmup.is_active(warmup_cfg):
        start_s, end_s = warmup.effective_range(warmup_cfg, duration_s)
        timestamp_s = (start_s + end_s) / 2
    elif duration_s is not None:
        timestamp_s = duration_s / 2
    else:
        timestamp_s = None

    try:
        if timestamp_s is not None:
            jpeg_bytes, _, _ = calibration.read_calibration_frame(video_file, timestamp_s=timestamp_s)
        else:
            jpeg_bytes, _, _ = calibration.read_calibration_frame(video_file)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(content=jpeg_bytes, media_type="image/jpeg")


def _get_results_sync(job_id: str) -> ResultsOut:
    output_path = config.output_dir(job_id)
    stats_file = output_path / config.STATS_FILE_NAME

    if not stats_file.exists():
        raise HTTPException(status_code=409, detail="Job hasn't finalized yet - consolidate first.")

    stats = json.loads(stats_file.read_text())

    rallies_raw: list[dict] = []
    status_file = output_path / config.GAME_STATUS_FILE_NAME
    if status_file.exists():
        status = json.loads(status_file.read_text())
        rallies_raw = status.get("rallies", [])

    # Once a warmup period is confirmed, this is the one place that filters
    # out rallies/hits detected outside it and rebases everything else to
    # be relative to its start - every other endpoint that returns rallies
    # or player stats (StatsTab, RalliesTab, ActionsTab, PlayerStatsPage,
    # ScoringDeterminationPage, ...) reads through this response rather
    # than the raw stats/game_status files directly, so they all inherit
    # this for free instead of needing their own warmup-awareness.
    warmup_cfg = warmup.load_config(output_path)
    if warmup.is_active(warmup_cfg):
        job = store.get(job_id)
        start_s, end_s = warmup.effective_range(warmup_cfg, job.duration_s if job else None)
        stats = warmup.rebase_stats(stats, start_s, end_s)
        rallies_raw = warmup.rebase_rallies(rallies_raw, start_s, end_s)

    rallies = [RallyOut(**rally) for rally in rallies_raw]

    return ResultsOut(
        job_id=job_id,
        players=stats.get("players", {}),
        rallies=rallies,
        caveats=stats.get("caveats", []),
        dashboard_available=(output_path / config.DASHBOARD_FILE_NAME).exists(),
        video_available=(output_path / config.ANNOTATED_VIDEO_NAME).exists(),
    )


@router.get("/results", response_model=ResultsOut)
async def get_results(job_id: str):
    _require_job(job_id)
    # Reading/parsing stats.json is synchronous disk I/O - run it off the
    # event loop thread so N concurrent /results calls (e.g. the Players/
    # Teams pages, which fetch every completed job's results at once) can
    # actually overlap instead of queueing behind each other one at a time.
    return await run_in_threadpool(_get_results_sync, job_id)


def _net_height_m(output_path: Path) -> float:
    """The calibrated net height (metres) for this job, or the standard
    men's height (CourtDefinition.court.NET_HEIGHT_M) as a fallback -
    same default calibration.py itself uses before one's explicitly set.
    Deliberately a lightweight direct read rather than calibration.
    existing_points (which needs a frame size just to also validate/return
    the four ground + two net-top image points this endpoint has no use
    for)."""
    court_file = output_path / config.COURT_FILE_NAME
    if not court_file.exists():
        return NET_HEIGHT_M
    try:
        data = json.loads(court_file.read_text())
        return float(data.get("net", {}).get("height_m", NET_HEIGHT_M))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return NET_HEIGHT_M


def _source_frame_size(output_path: Path) -> tuple[Optional[int], Optional[int]]:
    """(frame_w, frame_h) of the ORIGINAL uploaded video, as recorded by
    ball_detection.detectBall - the fixed coordinate space every px/py/box
    the ball- and player-trajectory endpoints return is measured in,
    regardless of whichever rendition (see transcode.TRANSCODE_TIERS) the
    client currently has selected for playback. (None, None) if
    ball_trajectory.json doesn't exist yet (ball_detection hasn't run) -
    player detection runs against the same source video, so this is reused
    for both endpoints rather than each tracking its own copy."""
    trajectory_file = output_path / TRAJECTORY_LOG_NAME
    if not trajectory_file.exists():
        return None, None
    try:
        data = json.loads(trajectory_file.read_text())
        return data.get("frame_w"), data.get("frame_h")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None, None


@router.get("/ball-trajectory", response_model=BallTrajectoryOut)
async def get_ball_trajectory(job_id: str):
    """A decimated (see BALL_TRAJECTORY_STRIDE), court-coordinate ball
    trajectory for drawing a live minimap in the web player - e.g.
    VideoPlayer.tsx's top-right ball-position indicator - as it plays.
    Not the full per-frame ball_speed.json (too large to ship to a
    browser wholesale for a full match), and rebased/filtered to the
    warmup range the same way /results already is, so its timestamps
    line up with everything else the player shows."""
    _require_job(job_id)

    output_path = config.output_dir(job_id)
    net_height_m = _net_height_m(output_path)
    frame_w, frame_h = _source_frame_size(output_path)
    speed_file = output_path / SPEED_LOG_NAME
    if not speed_file.exists():
        return BallTrajectoryOut(
            job_id=job_id, points=[], court_length_m=COURT_LENGTH, court_width_m=COURT_WIDTH,
            net_height_m=net_height_m, frame_w=frame_w, frame_h=frame_h,
        )

    entries = json.loads(speed_file.read_text())
    job = store.get(job_id)
    duration_s = job.duration_s if job else None
    fps = (len(entries) / duration_s) if duration_s and len(entries) > 0 else DEFAULT_FPS_ASSUMPTION

    warmup_cfg = warmup.load_config(output_path)
    start_s, end_s = warmup.effective_range(warmup_cfg, duration_s) if warmup.is_active(warmup_cfg) else (0.0, float("inf"))

    points: list[BallTrajectoryPointOut] = []
    for i in range(0, len(entries), BALL_TRAJECTORY_STRIDE):
        entry = entries[i]
        court = entry.get("court")
        pixel = entry.get("pixel")
        # A point needs at least one of court/pixel to be worth sending -
        # but not both: a job with no court calibration yet still has real
        # pixel positions, and the on-video Ball tracking annotation only
        # ever needs those (the minimap is the one that needs court and
        # simply won't have anything to draw for those points - see
        # BallMinimap.tsx). Dropping the whole point here used to mean an
        # uncalibrated job's on-video tracking showed nothing at all.
        if not court and not pixel:
            continue
        t = entry["frame_idx"] / fps
        if t < start_s or t > end_s:
            continue
        points.append(BallTrajectoryPointOut(
            t=t - start_s,
            x=court[0] if court else None,
            y=court[1] if court else None,
            px=pixel[0] if pixel else None,
            py=pixel[1] if pixel else None,
            height_m=entry.get("height_m"),
        ))

    return BallTrajectoryOut(
        job_id=job_id, points=points, court_length_m=COURT_LENGTH, court_width_m=COURT_WIDTH,
        net_height_m=net_height_m, frame_w=frame_w, frame_h=frame_h,
    )


# Denser than BALL_TRAJECTORY_STRIDE deliberately, not tied to it - the
# frontend linearly interpolates player boxes between consecutive served
# frames (see PlayerTrackingOverlay.tsx), and a human diving/changing
# direction mid-rally departs from a straight line far more within any
# given window than the ball's own flight arcs do, so the same 0.5s window
# that looks smooth for the ball reads as the player box visibly trailing
# behind real movement. ~5 frames at a typical 30fps source is ~6
# points/second (a ~0.17s window) - still a fraction of the full per-frame
# player_positions.json this is decimated from, but tight enough that even
# a fast dive gets 2-3 samples instead of 1.
PLAYER_TRAJECTORY_STRIDE = 5


@router.get("/player-trajectory", response_model=PlayerTrajectoryOut)
async def get_player_trajectory(job_id: str):
    """A decimated (see PLAYER_TRAJECTORY_STRIDE), pixel-space player
    position timeline for VideoPlayer.tsx's Player tracking annotation -
    draws each tracked player's box directly on the video as it plays, the
    same idea as /ball-trajectory's px/py but for every player at once
    instead of a single point. Not the full per-frame player_positions.json
    (too large to ship to a browser wholesale for a full match), and
    rebased/filtered to the warmup range the same way /results already is.
    Ignored stable_ids (see players.load_ignored) are left out - the same
    "not actually a player" judgment call the Setup tab already makes."""
    _require_job(job_id)

    output_path = config.output_dir(job_id)
    frame_w, frame_h = _source_frame_size(output_path)
    positions_file = output_path / players.PLAYER_POSITIONS_NAME
    if not positions_file.exists():
        return PlayerTrajectoryOut(job_id=job_id, frames=[], frame_w=frame_w, frame_h=frame_h)

    entries = json.loads(positions_file.read_text())
    job = store.get(job_id)
    duration_s = job.duration_s if job else None
    fps = (len(entries) / duration_s) if duration_s and len(entries) > 0 else DEFAULT_FPS_ASSUMPTION

    warmup_cfg = warmup.load_config(output_path)
    start_s, end_s = warmup.effective_range(warmup_cfg, duration_s) if warmup.is_active(warmup_cfg) else (0.0, float("inf"))

    names = players.load_names(output_path)
    ignored = players.load_ignored(output_path)
    # None for an uncalibrated job - court_x/court_y then stay None and the
    # minimap simply has no players to draw, same as it already handles a
    # ball point with no court position.
    matrix = load_homography(output_path)

    def court_of(box):
        """Court metres under a player's feet - the box's bottom centre, see
        PlayerBoxOut.court_x."""
        if matrix is None:
            return None, None
        x1, _y1, x2, y2 = box
        court_x, court_y = pixel_to_court((x1 + x2) / 2.0, y2, matrix)
        return float(court_x), float(court_y)

    frames: list[PlayerTrajectoryFrameOut] = []
    for i in range(0, len(entries), PLAYER_TRAJECTORY_STRIDE):
        entry = entries[i]
        t = entry["frame_idx"] / fps
        if t < start_s or t > end_s:
            continue

        boxes = []
        for p in entry.get("players", []):
            if p["stable_id"] in ignored:
                continue
            court_x, court_y = court_of(p["box"])
            boxes.append(PlayerBoxOut(
                stable_id=p["stable_id"],
                name=names.get(str(p["stable_id"])),
                box=p["box"],
                court_x=court_x,
                court_y=court_y,
            ))
        frames.append(PlayerTrajectoryFrameOut(t=t - start_s, players=boxes))

    return PlayerTrajectoryOut(job_id=job_id, frames=frames, frame_w=frame_w, frame_h=frame_h)


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
