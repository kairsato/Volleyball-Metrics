"""Orchestrates the "Share" settings panel's UPnP auto-port-forwarding
toggle and the share URL shown alongside it - see upnp.py for the actual
IGD protocol client this wraps, and auth.py for the login/expiry state this
coordinates with.

Both the frontend (5173, what a visitor's browser loads) and the backend
(8000, what that browser's own JS then calls directly) need to be reachable
from outside the LAN, so enabling/disabling UPnP always acts on both ports
together rather than exposing them independently - though each port's own
success/failure is tracked separately (see enable_upnp), since a router can
happily map one and refuse the other.
"""
import json
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

# External port == internal port for both - keeps the share URL predictable
# (no separate "what port did the router actually give me" step) and matches
# the ports this app already listens on.
SHARE_PORTS = [5173, 8000]

DEFAULT_CONFIG = {
    "upnp_enabled": False,
    "external_ip": None,
    # {"<port>": "open" | "error"} - only meaningful while upnp_enabled is
    # True; get_share_status() reports "not_forwarded" for every port
    # otherwise, regardless of what's left over here.
    "port_status": {},
    "last_error": None,
}

NO_ROUTER_ERROR = (
    "Your router doesn't support UPnP, or it's turned off. "
    f"Forward these ports manually instead: {', '.join(str(p) for p in SHARE_PORTS)}."
)

NOT_SHARE_ENABLED_ERROR = "Turn on Enable Share first - automatic port forwarding requires login to be turned on."


def _config_file() -> Path:
    return config.DATA_DIR / SHARE_CONFIG_NAME


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
    host = external_ip or local
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
        "ports": ports,
        "share_url": f"http://{host}:{SHARE_PORTS[0]}",
        "last_error": cfg["last_error"],
    }
