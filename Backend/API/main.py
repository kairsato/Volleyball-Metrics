import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import auth, config, score, share

# How often the background watcher re-checks whether Share's 7-day window
# has expired - see _watch_share_expiry. Short enough that an unattended
# server still closes itself down close to on time.
SHARE_EXPIRY_CHECK_INTERVAL_S = 600
from .routers import (
    auth_router,
    calibration_router,
    jobs_router,
    player_stats_router,
    players_router,
    results_router,
    roster_router,
    score_router,
    share_router,
    team_roster_router,
)

# Reachable with no session at all, even once login is turned on - status/
# captcha/login are what a not-yet-authenticated client needs to actually
# log in; health is just a liveness probe with nothing sensitive in it.
AUTH_PUBLIC_PATHS = {"/api/health", "/api/auth/status", "/api/auth/captcha", "/api/auth/login"}


class AuthMiddleware(BaseHTTPMiddleware):
    """A single gate in front of the whole API, rather than a `Depends()`
    sprinkled across every router - see auth.py's module docstring for why
    this exists at all. A no-op (every request passes straight through)
    whenever auth.status()'s "enabled" is False, which is the default -
    running this app on localhost/a LAN has never needed a login, and still
    doesn't unless a password is actually set up and turned on."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in AUTH_PUBLIC_PATHS:
            return await call_next(request)

        # Cheap on the normal path (one JSON read); only touches UPnP when
        # Share's window has actually just passed - see share.py's
        # module docstring for why this can never leave the app open with
        # no password required.
        share.enforce_expiry()

        if not auth.load_config()["enabled"]:
            return await call_next(request)

        header = request.headers.get("Authorization", "")
        token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
        # <img>/<video> tags can't attach an Authorization header, so the
        # handful of media endpoints (thumbnail/video/dashboard/source, plus
        # the calibration/score frame pickers) accept the same token as a
        # query param instead - see api.ts's thumbnailUrl/videoUrl/etc.
        if not token:
            token = request.query_params.get("token", "")

        if not auth.verify_session(token):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)

        return await call_next(request)


class DynamicCORSMiddleware:
    """Swaps between the LAN-only CORS policy (config.CORS_ORIGIN_REGEX) and
    a fully-open one, based on whether login is currently enabled.

    CORS on its own was never meant to be this API's security boundary -
    login (auth.py) is. The LAN-only restriction exists only to keep the
    *default*, no-login setup safe for its intended LAN-only use case; once
    a password is actually required, a browser can't forge the
    Authorization header a hostile cross-origin page would need anyway, so
    there's nothing left for a stricter CORS policy to usefully protect -
    and keeping it strict would just break the reachable-from-outside setup
    (a port-forward, UPnP) that requiring login was built to make safe.
    """

    def __init__(self, app):
        self._strict = CORSMiddleware(
            app,
            allow_origins=config.CORS_ORIGINS,
            allow_origin_regex=config.CORS_ORIGIN_REGEX,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._open = CORSMiddleware(
            app,
            allow_origin_regex=r".*",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def __call__(self, scope, receive, send):
        target = self._open if auth.load_config()["enabled"] else self._strict
        await target(scope, receive, send)


app = FastAPI(title="Volleyball Video Analytics API")

# Registered before the CORS middleware so CORS ends up OUTERMOST (Starlette
# wraps middleware in reverse-registration order) - a 401 from AuthMiddleware
# still needs the usual CORS headers attached, or the browser reports an
# opaque CORS failure instead of a readable 401.
app.add_middleware(AuthMiddleware)
app.add_middleware(DynamicCORSMiddleware)

app.include_router(jobs_router.router)
app.include_router(calibration_router.router)
app.include_router(players_router.router)
app.include_router(player_stats_router.router)
app.include_router(results_router.router)
app.include_router(roster_router.router)
app.include_router(team_roster_router.router)
app.include_router(score_router.router)
app.include_router(auth_router.router)
app.include_router(share_router.router)


@app.on_event("startup")
async def _reset_stuck_score_computations():
    # No compute thread from a previous process life can possibly still be
    # running once we're here - see score.reset_stuck_computations's
    # docstring for why a stuck "computing" status otherwise never recovers
    # on its own.
    score.reset_stuck_computations()


@app.on_event("startup")
async def _start_share_expiry_watcher():
    # AuthMiddleware already enforces expiry on every request, but that
    # alone can't close a genuinely unattended server (nobody making
    # requests means nobody ever triggers the check) - this background loop
    # is what actually guarantees Share turns itself off close to on time
    # even with zero traffic.
    async def _loop():
        while True:
            try:
                share.enforce_expiry()
            except Exception:
                pass
            await asyncio.sleep(SHARE_EXPIRY_CHECK_INTERVAL_S)

    asyncio.create_task(_loop())


@app.get("/api/health")
async def health():
    return {"status": "ok"}
