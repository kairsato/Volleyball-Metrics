from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from ..services import player_profiles
from ..schemas import PlayerProfileOut, PlayerProfileSummaryOut, PlayerProfilesOut

# A separate top-level /api/players/{name} namespace, distinct from
# players_router.py's /api/jobs/{job_id}/players - that one is per-job
# (naming/ignoring detections within a single video), this one is the
# cross-video profile backed by player_profiles.py's persisted store.
router = APIRouter(prefix="/api/players", tags=["player-stats"])


@router.get("/profiles", response_model=PlayerProfilesOut)
async def list_player_profiles():
    # Normally just a disk read - run_in_threadpool matters on the rare
    # request that lands right after something changed, where this pays
    # player_profiles._build()'s full rebuild cost (see its docstring) and
    # would otherwise block the whole event loop while it does.
    profiles = await run_in_threadpool(player_profiles.get_profiles)
    return PlayerProfilesOut(players=[PlayerProfileSummaryOut(**p) for p in profiles])


@router.get("/{name}/profile", response_model=PlayerProfileOut)
async def get_player_profile(name: str):
    profile = await run_in_threadpool(player_profiles.get_profile, name)
    return PlayerProfileOut(**profile)
