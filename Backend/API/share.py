"""Orchestrates the "Share" settings panel's UPnP auto-port-forwarding
toggle, optional custom hostname, and the share URL shown alongside them -
see upnp.py for the actual IGD protocol client this wraps, and auth.py for
the login/expiry state this coordinates with.

Forwards 80+443 to this machine's Caddy reverse proxy (see ../../Caddyfile),
not the frontend/backend dev ports (5173/8000) directly - Caddy terminates
HTTPS and forwards to both of those internally from there, so a visitor's
browser only ever talks to one HTTPS origin instead of two raw HTTP ports.
Enabling/disabling UPnP always acts on both ports together rather than
exposing them independently - though each port's own success/failure is
tracked separately (see enable_upnp), since a router can happily map one and
refuse the other.

The hostname (set_hostname) is for anyone fronting this with their own
dynamic-DNS domain (e.g. a free DuckDNS subdomain) instead of a bare IP -
it's what makes the Caddyfile's `{$DEV_HOSTNAME}` resolve to something real
without hardcoding anyone's personal domain into the repo: saving it here
also mirrors it to a plain-text file (_hostname_file) that _run_proxy.bat
reads into that env var before launching Caddy.
"""
import json
import re
import urllib.request
from pathlib import Path
from typing import Optional

from . import auth, config, upnp

# Plain-text public-IP echo service - used as a fallback when UPnP itself
# can't tell us the external IP (not enabled, no IGD, or the router's own
# GetExternalIPAddress call failed even though port mapping worked). This
# is the only outbound call this app makes to the open internet; it's a
# well-known no-signup service and the request carries nothing but "what's
# my own public IP" - see get_share_status.
PUBLIC_IP_LOOKUP_URL = "https://api.ipify.org"
PUBLIC_IP_LOOKUP_TIMEOUT_S = 3

SHARE_CONFIG_NAME = "share_config.json"

# Mirrors config["hostname"] as plain text (no JSON parsing needed) so
# _run_proxy.bat can read it with a bare `set /p` before starting Caddy.
HOSTNAME_FILE_NAME = "hostname.txt"

# Lenient DNS-hostname check - just enough to reject obvious junk (a URL
# with a scheme/path, whitespace, an empty string) before it ends up in the
# Share URL or gets written out for Caddy to use as a site address.
_HOSTNAME_RE = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))+$")

# 80 (plain HTTP, redirects to HTTPS) + 443 (HTTPS) - what Caddy listens on,
# not the frontend/backend's own 5173/8000 (see this module's docstring).
# External port == internal port for both - keeps the share URL predictable
# (no separate "what port did the router actually give me" step).
SHARE_PORTS = [80, 443]

DEFAULT_CONFIG = {
    "upnp_enabled": False,
    "external_ip": None,
    # {"<port>": "open" | "error"} - only meaningful while upnp_enabled is
    # True; get_share_status() reports "not_forwarded" for every port
    # otherwise, regardless of what's left over here.
    "port_status": {},
    "last_error": None,
    # User-supplied public hostname (e.g. "example.duckdns.org"), or None to
    # fall back to the detected IP - see set_hostname().
    "hostname": None,
}

INVALID_HOSTNAME_ERROR = "That doesn't look like a valid hostname (e.g. example.duckdns.org)."

NO_ROUTER_ERROR = (
    "Your router doesn't support UPnP, or it's turned off. "
    f"Forward these ports manually instead: {', '.join(str(p) for p in SHARE_PORTS)}."
)

NOT_SHARE_ENABLED_ERROR = "Turn on Enable Share first - automatic port forwarding requires login to be turned on."


def _config_file() -> Path:
    return config.DATA_DIR / SHARE_CONFIG_NAME


def _hostname_file() -> Path:
    return config.DATA_DIR / HOSTNAME_FILE_NAME


def _load_config() -> dict:
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


def enable_upnp() -> dict:
    # Enforced here, not just in the UI - the actual guarantee that "open
    # port, no password" can never happen through this feature.
    if not auth.load_config()["enabled"]:
        return _save_config({**DEFAULT_CONFIG, "last_error": NOT_SHARE_ENABLED_ERROR})

    igd = upnp.discover_igd()
    if igd is None:
        return _save_config({**_load_config(), "upnp_enabled": False, "port_status": {}, "last_error": NO_ROUTER_ERROR})

    local = upnp.local_ip()
    port_status: dict[str, str] = {}
    for port in SHARE_PORTS:
        ok = upnp.add_port_mapping(igd["control_url"], igd["service_type"], port, local, "Volleyball Video Analytics")
        port_status[str(port)] = "open" if ok else "error"

    any_ok = any(v == "open" for v in port_status.values())
    all_ok = all(v == "open" for v in port_status.values())
    external_ip = upnp.get_external_ip(igd["control_url"], igd["service_type"]) if any_ok else None

    return _save_config({
        "upnp_enabled": any_ok,
        "external_ip": external_ip,
        "port_status": port_status,
        "last_error": None if all_ok else NO_ROUTER_ERROR,
    })


def disable_upnp() -> dict:
    igd = upnp.discover_igd()
    if igd is not None:
        # Best-effort - a mapping that can't be removed (router rebooted,
        # UPnP toggled off since) shouldn't block clearing our own local
        # "enabled" state, or the UI would be stuck showing it as on.
        for port in SHARE_PORTS:
            upnp.delete_port_mapping(igd["control_url"], igd["service_type"], port)

    return _save_config({**DEFAULT_CONFIG})


def set_hostname(hostname: Optional[str]) -> dict:
    """Saves (or clears, if hostname is None/blank) the user's own public
    hostname. Raises ValueError on anything that isn't a bare DNS name -
    the caller is responsible for turning that into a 4xx response."""
    cleaned = (hostname or "").strip().lower()
    if not cleaned:
        _hostname_file().unlink(missing_ok=True)
        return _save_config({**_load_config(), "hostname": None})

    if not _HOSTNAME_RE.match(cleaned):
        raise ValueError(INVALID_HOSTNAME_ERROR)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _hostname_file().write_text(cleaned)
    return _save_config({**_load_config(), "hostname": cleaned})


def enforce_expiry() -> None:
    """Called on every request (see main.py's AuthMiddleware) and from a
    periodic background task - if login's share window just expired, also
    tears down any UPnP mapping, so expiry can never leave the app open
    with no password required."""
    if auth.check_share_expired():
        disable_upnp()


def _detect_public_ip() -> Optional[str]:
    """Best-effort fallback for when UPnP hasn't told us the external IP
    (off, no IGD, or GetExternalIPAddress specifically failed) - without
    this, the Share URL is only ever correct while UPnP is on, which makes
    it useless for the "forward the ports manually" path. Returns None on
    any failure (no internet, service unreachable) rather than raising -
    callers fall back to the LAN IP same as before."""
    try:
        with urllib.request.urlopen(PUBLIC_IP_LOOKUP_URL, timeout=PUBLIC_IP_LOOKUP_TIMEOUT_S) as resp:
            ip = resp.read().decode("ascii").strip()
    except Exception:
        return None

    parts = ip.split(".")
    if len(parts) != 4 or not all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
        return None
    return ip


def get_share_status() -> dict:
    enforce_expiry()
    cfg = _load_config()
    local = upnp.local_ip()
    external_ip = cfg["external_ip"] or _detect_public_ip()
    # A hostname the user set beats a raw IP - it's what gets Caddy a real,
    # browser-trusted cert (see ../../Caddyfile) instead of just being a
    # prettier URL.
    host = cfg["hostname"] or external_ip or local
    ports = [
        # A config saved before per-port tracking existed only ever set
        # upnp_enabled True when every port succeeded (old all-or-nothing
        # behavior) - "open" is the honest fallback for a missing entry
        # there, not "error".
        {"port": port, "status": cfg["port_status"].get(str(port), "open") if cfg["upnp_enabled"] else "not_forwarded"}
        for port in SHARE_PORTS
    ]
    return {
        "upnp_enabled": cfg["upnp_enabled"],
        "external_ip": external_ip,
        "local_ip": local,
        "hostname": cfg["hostname"],
        "ports": ports,
        # No port suffix - 443 is HTTPS's default, and that's what's actually
        # forwarded now (see SHARE_PORTS above / the Caddyfile).
        "share_url": f"https://{host}",
        "last_error": cfg["last_error"],
    }
