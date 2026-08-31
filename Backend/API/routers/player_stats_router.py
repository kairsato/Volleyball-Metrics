from fastapi import APIRouter

from .. import team_stats
from ..schemas import PlayerRadarOut

# A separate top-level /api/players/{name} namespace, distinct from
# players_router.py's /api/jobs/{job_id}/players - that one is per-job
# (naming/ignoring detections within a single video), this one is the
# cross-video profile PlayerStatsPage's own client-side aggregation
# (loadProfile()) doesn't have the data to compute itself.
router = APIRouter(prefix="/api/players", tags=["player-stats"])


@router.get("/{name}/radar", response_model=PlayerRadarOut)
async def get_player_radar(name: str):
    return PlayerRadarOut(**team_stats.compute_player_radar(name))
