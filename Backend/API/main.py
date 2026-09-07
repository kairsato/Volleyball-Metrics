import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import auth, config, network, score, share

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
    warmup_router,
)

# Reachable with no session at all, even once login is turned on - status/
# captcha/login are what a not-yet-authenticated client needs to actually
# log in; health is just a liveness probe with nothing sensitive in it.
AUTH_PUBLIC_PATHS = {"/api/health", "/api/auth/status", "/api/auth/captcha", "/api/auth/login"}

# Blocked for anyone not on the LAN, regardless of session/auth state -
# Share turning the rest of the app on for internet visitors was never
# meant to also hand them the controls for Share itself (or the ability to
# change the login password out from under the owner). See network.py.
LAN_ONLY_PATH_PREFIXES = ("/api/share",)
LAN_ONLY_PATHS = {"/api/auth/set-password", "/api/auth/enable", "/api/auth/generate-password"}


def _is_lan_only_path(path: str) -> bool:
    return path in LAN_ONLY_PATHS or any(path.startswith(prefix) for prefix in LAN_ONLY_PATH_PREFIXES)


class AuthMiddleware(BaseHTTPMiddleware):
    """A single gate in front of the whole API, rather than a `Depends()`
    sprinkled across every router - see auth.py's module docstring for why
    this exists at all. A no-op (every request passes straight through)
    whenever auth.status()'s "enabled" is False, which is the default -
    running this app on localhost/a LAN has never needed a login, and still
    doesn't unless a password is actually set up and turned on.

    Login only ever gates *remote* access - a LAN request never needs a
    session, even while Share is on, matching how this app has always
    worked (see auth.py's own module docstring). That's why
    DynamicCORSMiddleware below never opens itself up for arbitrary
    origins the way it used to: a hostile page loaded in a LAN user's own
    browser could otherwise ride that LAN-trust bypass with no token
    needed at all."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        if _is_lan_only_path(request.url.path) and not network.is_lan_request(request):
            return JSONResponse({"detail": "This is only available on the local network."}, status_code=403)

        if request.url.path in AUTH_PUBLIC_PATHS or network.is_lan_request(request):
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


app = FastAPI(title="Volleyball Video Analytics API")

# Registered before the CORS middleware so CORS ends up OUTERMOST (Starlette
# wraps middleware in reverse-registration order) - a 401 from AuthMiddleware
# still needs the usual CORS headers attached, or the browser reports an
# opaque CORS failure instead of a readable 401.
app.add_middleware(AuthMiddleware)
# Lets a LAN browser reach the API cross-port (frontend on :5173, API on
# :8000, see vite.config.ts) without needing a session, matching
# AuthMiddleware's own LAN bypass above. Deliberately never opens up wider
# than this, even once login is on: a legitimate remote visitor (via
# Share) talks to the API through Caddy on the SAME origin as the frontend
# (see ../../Caddyfile and api.ts's API_BASE), never cross-origin, so
# there's nothing for a wider policy to enable that a real client needs -
# and opening it would let a hostile page loaded in a LAN user's own
# browser ride AuthMiddleware's LAN bypass with no token needed at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_origin_regex=config.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router.router)
app.include_router(calibration_router.router)
app.include_router(players_router.router)
app.include_router(player_stats_router.router)
app.include_router(results_router.router)
app.include_router(roster_router.router)
app.include_router(team_roster_router.router)
app.include_router(score_router.router)
app.include_router(warmup_router.router)
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
