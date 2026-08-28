from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import calibration_router, jobs_router, players_router, results_router, roster_router

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


@app.get("/api/health")
async def health():
    return {"status": "ok"}
