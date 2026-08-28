from fastapi import APIRouter, HTTPException

from .. import roster
from ..schemas import RosterAddIn, RosterOut

router = APIRouter(prefix="/api/roster", tags=["roster"])


@router.get("", response_model=RosterOut)
async def get_roster():
    return RosterOut(players=roster.load_roster())


@router.post("", response_model=RosterOut)
async def add_player(body: RosterAddIn):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Name can't be blank.")
    return RosterOut(players=roster.add_to_roster(body.name))


@router.delete("/{name}", response_model=RosterOut)
async def remove_player(name: str):
    return RosterOut(players=roster.remove_from_roster(name))
