from fastapi import APIRouter, HTTPException

from .. import team_roster
from ..schemas import TeamRosterOut, TeamSaveIn

router = APIRouter(prefix="/api/teams", tags=["teams"])


@router.get("", response_model=TeamRosterOut)
async def get_teams():
    return TeamRosterOut(teams=team_roster.load_teams())


@router.post("", response_model=TeamRosterOut)
async def create_team(body: TeamSaveIn):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Team name can't be blank.")
    return TeamRosterOut(teams=team_roster.add_team(body.name, body.players))


@router.put("/{team_id}", response_model=TeamRosterOut)
async def update_team(team_id: str, body: TeamSaveIn):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Team name can't be blank.")
    return TeamRosterOut(teams=team_roster.update_team(team_id, body.name, body.players))


@router.delete("/{team_id}", response_model=TeamRosterOut)
async def delete_team(team_id: str):
    return TeamRosterOut(teams=team_roster.remove_team(team_id))
