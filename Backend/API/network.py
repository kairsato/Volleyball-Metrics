"""Distinguishes a request that genuinely originated on the local network
(or this same machine) from one that came in over the internet - used to
keep admin-ish endpoints (Share's own settings, changing the login
password) reachable only from the LAN even while Share makes the rest of
the app reachable from the internet. See main.py's AuthMiddleware, which is
where this actually gets enforced.
"""
import ipaddress
from typing import Optional

from fastapi import Request


def _client_ip(request: Request) -> Optional[str]:
    """The request's real origin IP - trusting X-Forwarded-For only when the
    direct TCP peer is localhost, meaning it genuinely came through this
    project's own Caddy instance (see ../../Caddyfile, whose reverse_proxy
    always sets this header to the real client's address). X-Forwarded-For
    is otherwise attacker-controlled - a request arriving any other way
    (including straight from the internet, if 8000 were ever exposed
    directly) is judged by its own direct peer address instead, which can't
    be spoofed the same way."""
    direct_ip = request.client.host if request.client else None

    if direct_ip in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # First entry is the original client - Caddy may append others
            # if there were further hops.
            direct_ip = forwarded.split(",")[0].strip()

    return direct_ip


def is_lan_request(request: Request) -> bool:
    ip = _client_ip(request)
    if not ip:
        return False
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False
