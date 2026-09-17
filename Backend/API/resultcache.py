"""Small in-process TTL cache for expensive, purely-derived-from-disk
computations (team stats, per-job matchup/action-quality) - this is a
self-hosted, single-user local tool (see root README.md's "What it gives
you"), so a short TTL trading a little staleness immediately after
reprocessing/rescoring for every repeat page visit within that window
feeling instant is the right tradeoff. This is deliberately separate from
players.load_player_positions's own mtime-keyed cache (which stays correct
forever, at the cost of parsing at least once per file change) - the
functions here recombine several inputs (rosters, scoring config, multiple
jobs at once) where a single clean mtime key isn't available, so a plain
time-based expiry is the pragmatic middle ground.

Cross-game PLAYER stats (the Players list, each player's own profile page)
use a different, stronger approach instead - see player_profiles.py. Those
pages are visited far more often than a time-based cache can comfortably
stay warm for, so their result is persisted to disk and only recomputed
when the underlying jobs actually changed, rather than re-derived from
scratch on a timer.
"""
import functools
import time
from threading import Lock

DEFAULT_TTL_S = 60.0


def ttl_cache(ttl_s: float = DEFAULT_TTL_S):
    """Like functools.lru_cache, but entries expire after ttl_s seconds
    instead of living until evicted by size - every argument must be
    hashable, same as lru_cache. wrapper.cache_clear() drops everything
    immediately, for call sites that know they just invalidated the
    underlying data (e.g. right after a reprocess/rescore/redo)."""

    def decorator(fn):
        cache: dict[tuple, tuple[float, object]] = {}
        lock = Lock()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            with lock:
                cached = cache.get(key)
                if cached is not None and now - cached[0] < ttl_s:
                    return cached[1]
            result = fn(*args, **kwargs)
            with lock:
                cache[key] = (now, result)
            return result

        wrapper.cache_clear = cache.clear
        return wrapper

    return decorator
