from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from .. import config, players, roster
from ..jobs import store
from ..schemas import (
    CandidateMatchOut,
    NamesUpdateIn,
    NamesUpdateOut,
    PlayerConfirmIn,
    PlayerOut,
    PlayersListOut,
)

router = APIRouter(prefix="/api/jobs/{job_id}/players", tags=["players"])


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


def _get_players_sync(job_id: str) -> PlayersListOut:
    video_path = config.find_input_video(job_id)
    output_path = config.output_dir(job_id)

    if video_path is None or not (output_path / players.PLAYER_POSITIONS_NAME).exists():
        raise HTTPException(
            status_code=409,
            detail="Player tracking hasn't finished yet - process the job first.",
        )

    player_list = players.list_players(video_path, output_path)
    candidate_matches = players.load_candidate_matches(output_path)
    return PlayersListOut(
        job_id=job_id,
        players=[PlayerOut(**p) for p in player_list],
        candidate_matches=[CandidateMatchOut(**m) for m in candidate_matches],
        confirmed=players.load_player_confirmed(output_path),
    )


@router.get("", response_model=PlayersListOut)
async def get_players(job_id: str):
    _require_job(job_id)
    # list_players does synchronous disk I/O and, on an uncached thumbnail,
    # a video seek + OpenCV encode - run it off the event loop thread so N
    # concurrent calls (Players/Teams pages fetch every completed job's
    # players at once) can actually overlap instead of queueing behind each
    # other one at a time, blocking the whole server meanwhile.
    return await run_in_threadpool(_get_players_sync, job_id)


@router.put("/names", response_model=NamesUpdateOut)
async def update_names(job_id: str, body: NamesUpdateIn):
    _require_job(job_id)

    output_path = config.output_dir(job_id)
    if not output_path.exists():
        raise HTTPException(status_code=409, detail="Job hasn't produced any output yet.")

    updated_names = players.save_names(output_path, body.names)
    updated_ignored = players.save_ignored(output_path, set(body.ignored))

    # A name typed here for the first time joins the shared roster too, so
    # it's a dropdown pick rather than a retype on every future job.
    if updated_names:
        roster.save_roster([*roster.load_roster(), *updated_names.values()])

    # Every newly-assigned name also joins the cross-video appearance
    # gallery (see player_gallery.py) so a *future* video can auto-suggest
    # this same person - best-effort, a video with no input file left (or
    # a crop that can't be extracted) just skips that one name rather than
    # failing the whole save.
    video_path = config.find_input_video(job_id)
    if video_path is not None:
        from .. import player_gallery

        for stable_id_str, name in body.names.items():
            if not name.strip():
                continue
            embedding = players.embed_player(video_path, output_path, int(stable_id_str))
            if embedding is not None:
                player_gallery.remember(name.strip(), embedding)

    return NamesUpdateOut(job_id=job_id, names=updated_names, ignored=updated_ignored)


@router.put("/confirm", response_model=PlayersListOut)
async def set_confirmed(job_id: str, body: PlayerConfirmIn):
    """Separate from PUT /names above (which persists the actual naming
    work) so confirming/redoing can never accidentally touch a name or
    ignored flag, and vice versa - saving a name doesn't touch this. Same
    split score_router.set_confirmed uses for scoring."""
    _require_job(job_id)
    output_path = config.output_dir(job_id)
    if not output_path.exists():
        raise HTTPException(status_code=409, detail="Job hasn't produced any output yet.")

    players.save_player_confirmed(output_path, body.confirmed)
    return await get_players(job_id)
