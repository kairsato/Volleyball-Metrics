from fastapi import APIRouter, HTTPException, Request

from .. import auth, share
from ..schemas import AuthStatusOut, CaptchaOut, LoginIn, LoginOut, SetEnabledIn, SetPasswordIn, SuggestedPasswordOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _token_from_request(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header.removeprefix("Bearer ").strip()
    return request.query_params.get("token", "")


@router.get("/status", response_model=AuthStatusOut)
async def get_status():
    return AuthStatusOut(**auth.status())


@router.get("/captcha", response_model=CaptchaOut)
async def get_captcha():
    captcha_id, png_bytes = auth.generate_captcha()
    return CaptchaOut(captcha_id=captcha_id, image_base64=auth.captcha_image_base64(png_bytes))


@router.post("/login", response_model=LoginOut)
async def login(body: LoginIn):
    remaining = auth.is_locked_out()
    if remaining is not None:
        raise HTTPException(status_code=429, detail=f"Too many failed attempts - try again in {remaining}s.")

    if not auth.verify_captcha(body.captcha_id, body.captcha_answer):
        raise HTTPException(status_code=400, detail="Incorrect captcha - try again.")

    if not auth.verify_password(body.password):
        auth.record_failed_attempt()
        raise HTTPException(status_code=401, detail="Incorrect password.")

    auth.record_success()
    return LoginOut(token=auth.create_session())


@router.post("/logout")
async def logout(request: Request):
    token = _token_from_request(request)
    if token:
        auth.revoke_session(token)
    return {"ok": True}


# set-password/enable are only reachable without a session while login is
# still off (auth.AuthMiddleware only starts requiring a token once
# enabled=True) - that's exactly first-time setup, so it needs no special
# casing here. Once login is on, these need a valid session like everything
# else, same as the rest of the app.
@router.post("/set-password", response_model=AuthStatusOut)
async def set_password(body: SetPasswordIn):
    try:
        auth.set_password(body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthStatusOut(**auth.status())


@router.post("/enable", response_model=AuthStatusOut)
async def set_enabled(body: SetEnabledIn):
    try:
        auth.set_enabled(body.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not body.enabled:
        # Closes the "turn off login, leave UPnP mapped" gap - best-effort
        # so a UPnP hiccup never blocks the login-disable response itself.
        try:
            share.disable_upnp()
        except Exception:
            pass

    return AuthStatusOut(**auth.status())


@router.get("/generate-password", response_model=SuggestedPasswordOut)
async def generate_password():
    return SuggestedPasswordOut(password=auth.generate_password())
