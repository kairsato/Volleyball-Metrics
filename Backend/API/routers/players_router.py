from fastapi import APIRouter, HTTPException

from .. import config, players, roster
from ..jobs import store
from ..schemas import CandidateMatchOut, NamesUpdateIn, NamesUpdateOut, PlayerOut, PlayersListOut

router = APIRouter(prefix="/api/jobs/{job_id}/players", tags=["players"])


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("", response_model=PlayersListOut)
async def get_players(job_id: str):
    _require_job(job_id)

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
    )


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

    return NamesUpdateOut(job_id=job_id, names=updated_names, ignored=updated_ignored)
