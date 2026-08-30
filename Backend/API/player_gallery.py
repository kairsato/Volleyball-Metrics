"""Best-effort cross-video player recognition: whenever a human assigns a
name to a detected player (in any video), a small appearance-embedding
snapshot of them is saved into a single global gallery keyed by name - not
tied to any one job. When a *new* video finishes its player-tracking stage
(see pipeline._phase_one), every still-unnamed detection is compared
against that gallery and auto-named if the match is confident and
unambiguous enough (see auto_identify_from_gallery in players.py, which
drives this).

This is deliberately conservative: a wrong auto-name silently corrupts that
person's stats, which is worse than just leaving them unidentified for a
human to name normally, so both AUTO_MATCH_MAX_DISTANCE and
AUTO_MATCH_MIN_MARGIN below are tuned to only fire on genuinely
unambiguous matches. Reuses PlayerDetection.tracker's AppearanceEncoder -
the same ResNet18-as-fixed-feature-extractor approach the within-job
"might be the same person" candidate-match system (tracker_offline.py)
already relies on, just applied across jobs instead of within one.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# PlayerDetection lives under Backend/Analysis/, a sibling of this API/
# package - same sys.path setup calibration.py already uses to reach
# CourtDefinition.
BACKEND_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for _directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from PlayerDetection.tracker import AppearanceEncoder  # noqa: E402

from . import config

GALLERY_NAME = "player_gallery.json"

# Per-name embedding cap - a handful of real snapshots beats one blended-
# together average (which tends to blur different lighting/jerseys across
# videos into something that matches nobody well), same idea as the
# within-job appearance gallery (tracker.APPEARANCE_GALLERY_SIZE).
MAX_EMBEDDINGS_PER_NAME = 10

# Cosine distance (1 - similarity) below which a match is trusted enough to
# auto-apply with no human looking at it first. Deliberately tighter than
# the within-job CANDIDATE_MAX_APPEARANCE_DISTANCE (0.45) - a within-job
# candidate is always still reviewed by a human before it does anything to
# stats; this one isn't.
AUTO_MATCH_MAX_DISTANCE = 0.30

# The best match also has to beat the second-best candidate (from a
# *different* name) by at least this much distance to auto-apply -
# otherwise two different people who both resemble the detection somewhat
# could get an arbitrary, low-confidence tiebreak silently baked into
# stats.
AUTO_MATCH_MIN_MARGIN = 0.08

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _encoder = AppearanceEncoder(device)
    return _encoder


def _gallery_file() -> Path:
    return config.DATA_DIR / GALLERY_NAME


def load_gallery() -> dict[str, list[list[float]]]:
    gallery_file = _gallery_file()
    if not gallery_file.exists():
        return {}
    try:
        return json.loads(gallery_file.read_text())
    except json.JSONDecodeError:
        return {}


def _save_gallery(gallery: dict[str, list[list[float]]]):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _gallery_file().write_text(json.dumps(gallery))


def embed_crop(crop_bgr: np.ndarray) -> Optional[list[float]]:
    """A single BGR crop -> its L2-normalised appearance embedding, or None
    if the crop is unusable (missing/too small) - see AppearanceEncoder."""
    embedding = _get_encoder().encode([crop_bgr])[0]
    return embedding.tolist() if embedding is not None else None


def remember(name: str, embedding: list[float]):
    """Adds one more appearance snapshot of `name` to the global gallery -
    called whenever a human assigns a name to a detection (players.py's
    save_names, via the /players/names endpoint), regardless of which video
    it came from. The oldest snapshot is dropped once a name has more than
    MAX_EMBEDDINGS_PER_NAME."""
    gallery = load_gallery()
    entries = gallery.setdefault(name, [])
    entries.append(embedding)
    if len(entries) > MAX_EMBEDDINGS_PER_NAME:
        del entries[: len(entries) - MAX_EMBEDDINGS_PER_NAME]
    _save_gallery(gallery)


def match(embedding: list[float]) -> Optional[tuple[str, float]]:
    """The gallery name this embedding most confidently, unambiguously
    matches - None unless it clears both the absolute-distance and
    margin-over-second-best bars above. The returned distance is the raw
    cosine distance to that best match (only meaningful for logging, since
    by the time a match is returned at all it's already confident).

    Compares per-*name* best distance, not per-embedding - two snapshots of
    the same (correct) person being close together isn't ambiguity, it's
    exactly what a real match should look like. The margin check only ever
    compares the best name against the best distance from every *other*
    name."""
    gallery = load_gallery()
    if not gallery:
        return None

    vec = np.array(embedding, dtype=np.float32)
    per_name_best: dict[str, float] = {}
    for name, entries in gallery.items():
        if not entries:
            continue
        per_name_best[name] = min(
            float(1.0 - np.dot(vec, np.array(stored, dtype=np.float32))) for stored in entries
        )

    if not per_name_best:
        return None

    ranked = sorted(per_name_best.items(), key=lambda item: item[1])
    best_name, best_dist = ranked[0]

    if best_dist > AUTO_MATCH_MAX_DISTANCE:
        return None
    if len(ranked) > 1 and ranked[1][1] - best_dist < AUTO_MATCH_MIN_MARGIN:
        return None

    return best_name, best_dist
