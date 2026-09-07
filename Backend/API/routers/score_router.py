import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .. import calibration, config, score
from ..jobs import store
from ..schemas import (
    GameBoundaryIn,
    RallyWinnerIn,
    ScoreComputeIn,
    ScoreConfigIn,
    ScoreConfigOut,
    ScoreConfirmIn,
    ScoreOut,
    ScoreRegionTestIn,
    ScoreRegionTestOut,
    ScoreResultOut,
)

router = APIRouter(prefix="/api/jobs/{job_id}/score", tags=["score"])

# Compute runs are serialized per job, not globally - a different video's
# Analyze click is never blocked by this, each still starts its own thread
# the moment it's requested. But a second Analyze click on the *same*
# video while one is already running is queued instead of either being
# rejected or spawning a second thread that would race the first one
# writing score_result.json: _pending holds at most one queued request per
# job_id (a third click just replaces it - only the latest request
# matters), which _run_compute picks up and runs itself once the current
# pass finishes.
_lock = threading.Lock()
_active_jobs: set[str] = set()
_pending: dict[str, tuple[str, Optional[tuple[int, int]]]] = {}


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("", response_model=ScoreOut)
async def get_score(job_id: str):
    _require_job(job_id)
    output_path = config.output_dir(job_id)
    cfg = score.load_config(output_path)
    result = score.load_result(output_path)
    return ScoreOut(
        job_id=job_id,
        config=ScoreConfigOut(**cfg),
        result=ScoreResultOut(**result) if result else None,
    )


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


# A plain (non-async) def, unlike every other route here - FastAPI runs a
# sync route in its worker thread pool rather than the main event loop, and
# OCR inference (easyocr, first call also lazily loads the model) is slow
# and CPU/GPU-bound enough that running it inline in an async route would
# stall every other request on this server for the duration.
@router.post("/test-region", response_model=ScoreRegionTestOut)
def test_region(job_id: str, body: ScoreRegionTestIn):
    _require_job(job_id)
    video_path = config.find_input_video(job_id)
    if video_path is None:
        raise HTTPException(status_code=404, detail="No uploaded video found for this job")

    from .. import score_cv  # lazy - pulls in easyocr/torch, not needed unless OCR is actually used

    try:
        result = score_cv.test_region(video_path, body.t, body.region.model_dump(), body.min_confidence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ScoreRegionTestOut(**result)


@router.put("/config", response_model=ScoreConfigOut)
async def set_config(job_id: str, body: ScoreConfigIn):
    _require_job(job_id)
    if body.method not in score.METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method '{body.method}'")

    output_path = config.output_dir(job_id)
    cfg = score.save_config(output_path, body.model_dump())
    return ScoreConfigOut(**cfg)


@router.put("/confirm", response_model=ScoreConfigOut)
async def set_confirmed(job_id: str, body: ScoreConfirmIn):
    """Separate from set_config above (which replaces the whole
    method/teams/region/reverse-direction bundle) so confirming/redoing
    scoring can never accidentally reset a config field it has nothing to
    do with, and vice versa - saving a team change doesn't touch this."""
    _require_job(job_id)
    output_path = config.output_dir(job_id)
    cfg = score.save_config(output_path, {"confirmed": body.confirmed})
    return ScoreConfigOut(**cfg)


def _run_compute(job_id: str, method: str, rally_range: Optional[tuple[int, int]]):
    output_path = config.output_dir(job_id)
    while True:
        try:
            if method == "automatic":
                score.compute_automatic(output_path, rally_range=rally_range)
            elif method == "ocr":
                from .. import score_cv

                score_cv.compute_cv(output_path, config.find_input_video(job_id), rally_range=rally_range)
            score.save_config(output_path, {"compute_status": "done", "compute_error": None})
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            score.save_config(output_path, {"compute_status": "error", "compute_error": str(exc)})

        with _lock:
            queued = _pending.pop(job_id, None)
            if queued is None:
                _active_jobs.discard(job_id)
                return
            method, rally_range = queued

        # A queued request re-marks "computing" itself, right before this
        # loop runs it - between the previous pass finishing and here, a
        # poll could otherwise briefly see "done"/"error" and stop polling
        # just before the queued run actually starts.
        score.save_config(output_path, {"compute_status": "computing", "compute_error": None})


@router.post("/compute", response_model=ScoreConfigOut)
async def compute(job_id: str, body: ScoreComputeIn = ScoreComputeIn()):
    _require_job(job_id)
    output_path = config.output_dir(job_id)
    cfg = score.load_config(output_path)

    if cfg["method"] not in ("automatic", "ocr"):
        raise HTTPException(status_code=409, detail="Only the Heuristic or Computer Vision method can be computed")

    rally_range = score.resolve_rally_range(output_path, body.range_type, body.range_start, body.range_end)

    with _lock:
        already_running = job_id in _active_jobs
        if already_running:
            _pending[job_id] = (cfg["method"], rally_range)
        else:
            _active_jobs.add(job_id)

    cfg = score.save_config(output_path, {"compute_status": "computing", "compute_error": None})
    if already_running:
        return ScoreConfigOut(**cfg)

    thread = threading.Thread(target=_run_compute, args=(job_id, cfg["method"], rally_range), daemon=True)
    thread.start()

    return ScoreConfigOut(**cfg)


@router.put("/games/{rally_index}", response_model=ScoreResultOut)
async def set_game_boundary(job_id: str, rally_index: int, body: GameBoundaryIn):
    _require_job(job_id)
    result = score.set_game_boundary(config.output_dir(job_id), rally_index, body.split)
    return ScoreResultOut(**result)


@router.put("/rallies/{rally_index}", response_model=ScoreResultOut)
async def set_rally_winner(job_id: str, rally_index: int, body: RallyWinnerIn):
    _require_job(job_id)
    result = score.set_rally_winner(config.output_dir(job_id), rally_index, body.winner)
    return ScoreResultOut(**result)


@router.post("/reset", response_model=ScoreResultOut)
async def reset_scores(job_id: str):
    _require_job(job_id)
    result = score.reset_winners(config.output_dir(job_id))
    return ScoreResultOut(**result)
