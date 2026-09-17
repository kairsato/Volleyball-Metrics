"""Backs the frontend's Configuration page (gear menu -> Configuration): a
static registry of every stage's tunable heuristics, plus CRUD over
persisted profiles (see ../services/heuristics.py). Not job-scoped - this is
global tuning applied to every job processed while a profile is active.
"""
from fastapi import APIRouter, HTTPException

from ..services import heuristics
from ..schemas import (
    HeuristicProfileOut,
    HeuristicsRegistryOut,
    HeuristicsStateOut,
    ProfileCreateIn,
    ProfileUpdateIn,
)

router = APIRouter(prefix="/api/configuration", tags=["configuration"])


def _clear_result_caches() -> None:
    """Best-effort: the "consolidating" stage's action-quality weights are
    read on demand (see action_quality.py's module docstring), through a
    couple of short-TTL caches - drop both so a profile change is visible
    immediately rather than up to DEFAULT_TTL_S later."""
    from ..services import action_quality, team_stats

    action_quality.compute_action_quality.cache_clear()
    team_stats.compute_team_stats.cache_clear()


@router.get("/registry", response_model=HeuristicsRegistryOut)
async def get_registry():
    return HeuristicsRegistryOut(stages=[
        {
            "key": stage.key,
            "label": stage.label,
            "phase": stage.phase,
            "summary": stage.summary,
            "params": [
                {
                    "key": p.key,
                    "label": p.label,
                    "description": p.description,
                    "type": p.type,
                    "default": p.default,
                    "min": p.min,
                    "max": p.max,
                    "step": p.step,
                    "options": p.options,
                    "group": p.group,
                }
                for p in stage.params
            ],
        }
        for stage in heuristics.get_registry()
    ])


@router.get("/profiles", response_model=HeuristicsStateOut)
async def list_profiles():
    return heuristics.list_state()


@router.post("/profiles", response_model=HeuristicProfileOut)
async def create_profile(body: ProfileCreateIn):
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Profile name can't be empty")
    return heuristics.create_profile(body.name.strip(), body.base_profile_id)


@router.patch("/profiles/{profile_id}", response_model=HeuristicProfileOut)
async def update_profile(profile_id: str, body: ProfileUpdateIn):
    try:
        result = heuristics.update_profile(
            profile_id,
            name=body.name.strip() if body.name is not None else None,
            values=body.values,
        )
    except heuristics.HeuristicsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _clear_result_caches()
    return result


@router.delete("/profiles/{profile_id}", response_model=HeuristicsStateOut)
async def delete_profile(profile_id: str):
    try:
        result = heuristics.delete_profile(profile_id)
    except heuristics.HeuristicsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _clear_result_caches()
    return result


@router.post("/profiles/{profile_id}/activate", response_model=HeuristicsStateOut)
async def activate_profile(profile_id: str):
    try:
        result = heuristics.set_active_profile(profile_id)
    except heuristics.HeuristicsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _clear_result_caches()
    return result
