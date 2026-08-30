from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config, score
from .routers import (
    calibration_router,
    jobs_router,
    players_router,
    results_router,
    roster_router,
    score_router,
    team_roster_router,
)

app = FastAPI(title="Volleyball Video Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router.router)
app.include_router(calibration_router.router)
app.include_router(players_router.router)
app.include_router(results_router.router)
app.include_router(roster_router.router)
app.include_router(team_roster_router.router)
app.include_router(score_router.router)


@app.on_event("startup")
async def _reset_stuck_score_computations():
    # No compute thread from a previous process life can possibly still be
    # running once we're here - see score.reset_stuck_computations's
    # docstring for why a stuck "computing" status otherwise never recovers
    # on its own.
    score.reset_stuck_computations()


@app.get("/api/health")
async def health():
    return {"status": "ok"}
