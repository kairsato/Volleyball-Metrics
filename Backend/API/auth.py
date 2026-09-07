"""Optional single-shared-password login, off by default. Not a multi-user
system - there's exactly one password, matching every other piece of global
state in this app (team_roster.json, player_roster.json). Exists purely so
the app can be reached safely from outside the LAN (a port-forward, a
tunnel) without leaving an unauthenticated API - create/delete/upload - open
to the internet; running only on localhost/the LAN has no need for this at
all, hence the "off by default".

Session tokens are persisted to disk (auth_sessions.json), not just held in
memory, so a `uvicorn --reload` restart during development doesn't silently
log everyone out - everything else here (the captcha store, the login
lockout counter) is short-lived enough that losing it on a restart is fine.
"""
import base64
import hashlib
import io
import json
import random
import secrets
import string
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from . import config

AUTH_CONFIG_NAME = "auth_config.json"
AUTH_SESSIONS_NAME = "auth_sessions.json"

DEFAULT_CONFIG = {
    "enabled": False,
    "password_hash": None,
    "password_salt": None,
    "failed_attempts": 0,
    "locked_until": None,
    # Set from set_enabled(True)'s duration_days whenever Share turns on,
    # cleared on any disable - bounds how long the app stays
    # reachable-with-a-password rather than relying on someone remembering
    # to turn it back off, unless the user explicitly picked "Forever" (see
    # the Share dialog's duration dropdown), which leaves this None while
    # still enabled. See check_share_expired().
    "share_expires_at": None,
}

# set_enabled(True)'s default when the caller (the Share dialog's duration
# dropdown) doesn't specify one explicitly.
MAX_SHARE_DURATION_DAYS = 7

# A password has to clear this bar before it can be set at all - deliberately
# stricter than a typical single-site password, since this one password
# guards the whole app once login is turned on.
PASSWORD_MIN_LENGTH = 16
PASSWORD_MIN_CHAR_CLASSES = 3

PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16

GENERATED_PASSWORD_LENGTH = 16
# Kept separate from CAPTCHA_ALPHABET (which excludes ambiguous glyphs for a
# *human to read off an image*) - a generated password is only ever copied,
# never read character-by-character, so the full symbol set is fair game.
_GENERATED_PASSWORD_SYMBOLS = "!@#$%^&*()-_=+"

# A week - long enough that a returning visitor doesn't have to re-clear
# the captcha/password every couple hours, short enough that a stolen/leaked
# token (synced localStorage on a shared device, a browser history snoop)
# doesn't grant access indefinitely. The token itself still lives in
# localStorage (persists across a tab close/reopen within this window) -
# logging out (or the admin disabling Share) revokes it immediately either
# way, this is just the outer bound for an otherwise-unused token.
SESSION_LIFETIME_DAYS = 7

# After this many consecutive wrong passwords, login is refused outright for
# LOCKOUT_SECONDS - a fresh captcha is already required per attempt, this is
# a second, coarser layer on top rather than the only defense.
LOCKOUT_THRESHOLD = 5
LOCKOUT_SECONDS = 5 * 60

CAPTCHA_LIFETIME_SECONDS = 5 * 60
CAPTCHA_LENGTH = 5
# Excludes glyphs that are easy to confuse with each other at small size /
# with noise added (0/O, 1/l/I) - a captcha that's ambiguous to a real user
# isn't actually testing anything useful.
CAPTCHA_ALPHABET = "".join(c for c in string.ascii_uppercase + string.digits if c not in "0O1IL")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _config_file() -> Path:
    return config.DATA_DIR / AUTH_CONFIG_NAME


def _sessions_file() -> Path:
    return config.DATA_DIR / AUTH_SESSIONS_NAME


def load_config() -> dict:
    if not _config_file().exists():
        return dict(DEFAULT_CONFIG)
    try:
        return {**DEFAULT_CONFIG, **json.loads(_config_file().read_text())}
    except json.JSONDecodeError:
        return dict(DEFAULT_CONFIG)


def _save_config(cfg: dict) -> dict:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _config_file().write_text(json.dumps(cfg, indent=2))
    return cfg


def status() -> dict:
    cfg = load_config()
    return {
        "enabled": cfg["enabled"],
        "password_set": cfg["password_hash"] is not None,
        "expires_at": cfg["share_expires_at"],
    }


def _password_strength_error(password: str) -> Optional[str]:
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters."

    classes_present = sum([
        any(c.islower() for c in password),
        any(c.isupper() for c in password),
        any(c.isdigit() for c in password),
        any(not c.isalnum() for c in password),
    ])
    if classes_present < PASSWORD_MIN_CHAR_CLASSES:
        return (
            f"Password must include at least {PASSWORD_MIN_CHAR_CLASSES} of: "
            "lowercase letters, uppercase letters, digits, symbols."
        )
    return None


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return digest.hex()


def set_password(password: str) -> None:
    """Raises ValueError with a human-readable reason if the password is too
    weak - callers should surface that message directly to the user."""
    error = _password_strength_error(password)
    if error:
        raise ValueError(error)

    salt = secrets.token_bytes(SALT_BYTES)
    cfg = load_config()
    cfg["password_hash"] = _hash_password(password, salt)
    cfg["password_salt"] = salt.hex()
    # A new password invalidates whatever the lockout counter was tracking.
    cfg["failed_attempts"] = 0
    cfg["locked_until"] = None
    _save_config(cfg)


def generate_password() -> str:
    """A random password that's always guaranteed to pass
    _password_strength_error - one character from each required class,
    the rest random from the combined pool, order shuffled so the fixed
    classes aren't predictably in the first few positions."""
    rng = secrets.SystemRandom()
    pools = [string.ascii_lowercase, string.ascii_uppercase, string.digits, _GENERATED_PASSWORD_SYMBOLS]
    chars = [rng.choice(pool) for pool in pools]
    all_chars = "".join(pools)
    chars += [rng.choice(all_chars) for _ in range(GENERATED_PASSWORD_LENGTH - len(chars))]
    rng.shuffle(chars)
    return "".join(chars)


def is_locked_out() -> Optional[int]:
    """Seconds remaining in the current lockout, or None if login attempts
    are currently allowed."""
    cfg = load_config()
    if not cfg["locked_until"]:
        return None
    locked_until = datetime.fromisoformat(cfg["locked_until"])
    remaining = (locked_until - _now()).total_seconds()
    return max(1, round(remaining)) if remaining > 0 else None


def record_failed_attempt() -> None:
    cfg = load_config()
    cfg["failed_attempts"] = cfg.get("failed_attempts", 0) + 1
    if cfg["failed_attempts"] >= LOCKOUT_THRESHOLD:
        cfg["locked_until"] = (_now() + timedelta(seconds=LOCKOUT_SECONDS)).isoformat()
        cfg["failed_attempts"] = 0
    _save_config(cfg)


def record_success() -> None:
    cfg = load_config()
    cfg["failed_attempts"] = 0
    cfg["locked_until"] = None
    _save_config(cfg)


def verify_password(password: str) -> bool:
    cfg = load_config()
    if not cfg["password_hash"] or not cfg["password_salt"]:
        return False
    salt = bytes.fromhex(cfg["password_salt"])
    # Constant-time comparison - a plain == on the hex digests would leak
    # timing information about how many leading characters matched.
    return secrets.compare_digest(_hash_password(password, salt), cfg["password_hash"])


def set_enabled(enabled: bool, duration_days: Optional[int] = MAX_SHARE_DURATION_DAYS) -> None:
    """duration_days is only meaningful when enabling: how long until Share
    auto-disables itself (see check_share_expired) - None means it never
    auto-expires. Ignored when disabling, which always clears the expiry."""
    if enabled and not load_config()["password_hash"]:
        raise ValueError("Set a password before enabling login.")
    if enabled and duration_days is not None and duration_days <= 0:
        raise ValueError("duration_days must be positive, or omitted/null for no expiry.")

    cfg = {**load_config(), "enabled": enabled}
    if enabled:
        cfg["share_expires_at"] = (
            (_now() + timedelta(days=duration_days)).isoformat() if duration_days is not None else None
        )
    else:
        cfg["share_expires_at"] = None
    _save_config(cfg)


def check_share_expired() -> bool:
    """Auto-disables login once its share window has passed - called on
    every request (see main.py's AuthMiddleware) and from a periodic
    background task (see share.enforce_expiry), so exposure doesn't outlive
    an unattended browser tab. A missing share_expires_at while enabled is
    True (a config saved before this field existed) is treated as "never
    expires" rather than an immediate forced logout. Returns True the
    moment it actually flips enabled off, so callers know to also tear down
    any UPnP port mapping."""
    cfg = load_config()
    if not cfg["enabled"] or not cfg["share_expires_at"]:
        return False
    if _now() < datetime.fromisoformat(cfg["share_expires_at"]):
        return False
    cfg["enabled"] = False
    cfg["share_expires_at"] = None
    _save_config(cfg)
    return True


# --- Sessions ---

def _load_sessions() -> dict[str, str]:
    if not _sessions_file().exists():
        return {}
    try:
        return json.loads(_sessions_file().read_text())
    except json.JSONDecodeError:
        return {}


def _save_sessions(sessions: dict[str, str]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _sessions_file().write_text(json.dumps(sessions))


def _prune_expired(sessions: dict[str, str]) -> dict[str, str]:
    now = _now()
    return {token: expiry for token, expiry in sessions.items() if datetime.fromisoformat(expiry) > now}


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    sessions = _prune_expired(_load_sessions())
    sessions[token] = (_now() + timedelta(days=SESSION_LIFETIME_DAYS)).isoformat()
    _save_sessions(sessions)
    return token


def verify_session(token: str) -> bool:
    if not token:
        return False
    sessions = _prune_expired(_load_sessions())
    valid = token in sessions
    _save_sessions(sessions)
    return valid


def revoke_session(token: str) -> None:
    sessions = _load_sessions()
    if token in sessions:
        del sessions[token]
        _save_sessions(sessions)


# --- CAPTCHA ---
# In-memory only, deliberately: a captcha is only ever meant to survive a
# few minutes, so losing the store on a dev-server reload just means the
# in-progress attempt has to fetch a new one - not worth persisting to disk.
_captcha_store: dict[str, tuple[str, float]] = {}

CAPTCHA_IMAGE_SIZE = (200, 80)


def _draw_noise_lines(draw: ImageDraw.ImageDraw, count: int) -> None:
    w, h = CAPTCHA_IMAGE_SIZE
    for _ in range(count):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        draw.line([(x1, y1), (x2, y2)], fill=(random.randint(150, 200),) * 3, width=1)


def generate_captcha() -> tuple[str, bytes]:
    """Returns (captcha_id, png_bytes). The expected answer is never sent to
    the client - only the rendered image and an opaque id."""
    text = "".join(random.choice(CAPTCHA_ALPHABET) for _ in range(CAPTCHA_LENGTH))

    image = Image.new("RGB", CAPTCHA_IMAGE_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    _draw_noise_lines(draw, count=6)

    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except OSError:
        font = ImageFont.load_default()

    char_width = CAPTCHA_IMAGE_SIZE[0] // (CAPTCHA_LENGTH + 1)
    for i, char in enumerate(text):
        char_img = Image.new("RGBA", (60, 60), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((10, 5), char, font=font, fill=(random.randint(0, 90),) * 3)
        rotated = char_img.rotate(random.randint(-30, 30), expand=True, resample=Image.BICUBIC)
        x = char_width * (i + 1) - 30 + random.randint(-6, 6)
        y = (CAPTCHA_IMAGE_SIZE[1] - rotated.height) // 2 + random.randint(-8, 8)
        image.paste(rotated, (x, y), rotated)

    _draw_noise_lines(draw, count=4)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    captcha_id = secrets.token_urlsafe(16)
    _captcha_store[captcha_id] = (text, time.monotonic() + CAPTCHA_LIFETIME_SECONDS)

    # Best-effort cleanup of anything else that's expired, so the in-memory
    # store doesn't grow unbounded over a long-running dev server.
    now = time.monotonic()
    for key in [k for k, (_, expiry) in _captcha_store.items() if expiry < now]:
        del _captcha_store[key]

    return captcha_id, buffer.getvalue()


def verify_captcha(captcha_id: str, answer: str) -> bool:
    """One-time use: the entry is removed whether the answer was right or
    wrong, so a captcha can never be retried or brute-forced."""
    entry = _captcha_store.pop(captcha_id, None)
    if entry is None:
        return False
    expected, expiry = entry
    if time.monotonic() > expiry:
        return False
    return answer.strip().upper() == expected


def captcha_image_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")
