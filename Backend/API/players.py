import base64
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

PLAYER_POSITIONS_NAME = "player_positions.json"
STATS_NAME = "player_stats.json"
ACTIONS_NAME = "actions.json"
GROUPED_ACTIONS_NAME = "actions_grouped.json"
CANDIDATE_MATCHES_NAME = "player_candidate_matches.json"
THUMBNAIL_CACHE_NAME = "player_thumbnails_cache.json"
PLAYER_CONFIG_NAME = "player_config.json"

THUMBNAIL_MAX_DIM = 220
JPEG_QUALITY = 85
# Bumped whenever a change would make an already-cached thumbnail wrong even
# though its frame_idx hasn't changed (e.g. reworking the spotlight overlay
# below) - a cache entry stamped with an older version is treated as a miss
# and regenerated once, rather than serving a stale thumbnail indefinitely.
THUMBNAIL_CACHE_VERSION = 4

# A detector's box is often a touch tight around the actual person - pad it
# outward proportionally (same fraction for everyone, so crops stay
# consistent) rather than cropping exactly to the box, which regularly
# clips the top of someone's head or their feet.
THUMBNAIL_PADDING_FRACTION = 0.18

# How many of a stable_id's best-ranked candidate appearances (see
# _ranked_candidates_per_player) are actually worth decoding and scoring for
# motion blur when picking its thumbnail. Comparing every appearance a
# player has across a whole video would make an uncached thumbnail request
# far too slow - this keeps it to a handful of already-good (unclipped,
# same conflict tier, similarly-sized) candidates.
BLUR_CANDIDATE_COUNT = 6


def _box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _load_player_positions(output_path: Path) -> list[dict]:
    positions_file = output_path / PLAYER_POSITIONS_NAME
    if not positions_file.exists():
        return []
    return json.loads(positions_file.read_text())


# Bounds on height/width for a box to plausibly be a single standing/moving
# person rather than a spurious detection - a fused box covering two
# players, a shadow, a sideline ad board, etc. These are deliberately loose
# (a diving dig can get quite wide-and-short) since the goal is only to
# reject the obviously-wrong outliers, not to be a real pose classifier.
MIN_PERSON_ASPECT_RATIO = 0.9  # height / width
MAX_PERSON_ASPECT_RATIO = 4.5

# A box touching the frame edge is missing part of the person - a worse
# thumbnail even if it happens to have the largest area.
EDGE_MARGIN_FRACTION = 0.01


def _is_plausible_person_box(box: list[float], frame_size: Optional[tuple[float, float]]) -> bool:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return False

    aspect = height / width
    if not (MIN_PERSON_ASPECT_RATIO <= aspect <= MAX_PERSON_ASPECT_RATIO):
        return False

    if frame_size is not None:
        frame_w, frame_h = frame_size
        margin_x, margin_y = frame_w * EDGE_MARGIN_FRACTION, frame_h * EDGE_MARGIN_FRACTION
        if x1 <= margin_x or y1 <= margin_y or x2 >= frame_w - margin_x or y2 >= frame_h - margin_y:
            return False

    return True


def _frame_size(video_path: Path) -> Optional[tuple[float, float]]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (w, h) if w > 0 and h > 0 else None
    finally:
        cap.release()


def _padded_box(box: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * THUMBNAIL_PADDING_FRACTION
    pad_y = (y2 - y1) * THUMBNAIL_PADDING_FRACTION
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y


def _boxes_intersect(a: tuple[float, float, float, float], b: list[float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _has_conflict(box: list[float], other_boxes: list[list[float]]) -> bool:
    """Whether some other player's raw box falls inside the (padded) region
    that will actually be cropped out for `box`'s thumbnail - i.e. whether
    that other person would visibly appear alongside the subject in the
    thumbnail, making it ambiguous who's being shown without a highlight."""
    padded = _padded_box(box)
    return any(_boxes_intersect(padded, other) for other in other_boxes)


def _ranked_candidates_per_player(
    positions: list[dict], frame_size: Optional[tuple[float, float]] = None
) -> dict[int, list[dict]]:
    """For each stable_id, every appearance that plausibly looks like a
    real, unclipped person (falling back to every appearance at all if none
    do, so a thumbnail is always produced - see _is_plausible_person_box),
    ranked best-first: appearances with no other player's box crowding into
    the thumbnail crop first, largest box within that tier next. Plain
    "largest box wins" alone occasionally picks a spurious detection - a
    fused box, a shadow, a sideline ad board - that happens to be big; the
    plausibility filter is what catches that. The actual thumbnail source is
    then picked from the front of this ranking by _pick_best_source, which
    additionally screens the top candidates for motion blur."""
    plausible: dict[int, list[dict]] = {}
    fallback: dict[int, list[dict]] = {}
    appearances: dict[int, int] = {}

    for entry in positions:
        frame_idx = entry["frame_idx"]
        timestamp_s = entry.get("timestamp_s")
        frame_players = entry["players"]
        for player in frame_players:
            stable_id = player["stable_id"]
            appearances[stable_id] = appearances.get(stable_id, 0) + 1

            box = player["box"]
            other_boxes = [other["box"] for other in frame_players if other is not player]
            record = {
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "box": box,
                "area": _box_area(box),
                "conflict": _has_conflict(box, other_boxes),
            }

            fallback.setdefault(stable_id, []).append(record)
            if _is_plausible_person_box(box, frame_size):
                plausible.setdefault(stable_id, []).append(record)

    result: dict[int, list[dict]] = {}
    for stable_id, records in fallback.items():
        records = plausible.get(stable_id, records)
        for record in records:
            record["appearances"] = appearances[stable_id]
        result[stable_id] = sorted(records, key=lambda r: (r["conflict"], -r["area"]))

    return result


def _best_crop_per_player(positions: list[dict], frame_size: Optional[tuple[float, float]] = None) -> dict[int, dict]:
    """The single best candidate per stable_id (see _ranked_candidates_per_
    player) - for callers that just want one good, non-conflicting crop
    (embed_player, auto_identify_from_gallery) and don't need the blur-aware
    refinement list_players does for the user-facing thumbnail."""
    return {stable_id: records[0] for stable_id, records in _ranked_candidates_per_player(positions, frame_size).items()}


def _sharpness_score(crop_bgr: np.ndarray) -> float:
    """Higher = less motion-blurred. Variance of the Laplacian is a standard
    cheap blur proxy: a sharp image has lots of high-frequency edge content
    (high variance), a blurred one is smoothed out (low variance)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pick_best_source(video_path: Path, candidates: list[dict]) -> Optional[dict]:
    """Refines the front of a stable_id's ranked candidate list (see
    _ranked_candidates_per_player) by motion blur: within the best conflict
    tier present (non-conflicting appearances if there are any, otherwise
    conflicting ones), and among candidates close in size to the largest one
    there (so a small-but-sharp box can't beat a properly-sized one), picks
    whichever actually decodes the sharpest. Falls back to the top-ranked
    candidate outright if there's only one worth comparing, since decoding
    a frame means a real video seek - not worth paying for when there's
    nothing to compare against."""
    if not candidates:
        return None

    best_conflict = candidates[0]["conflict"]
    tier = [c for c in candidates if c["conflict"] == best_conflict][:BLUR_CANDIDATE_COUNT]
    if len(tier) == 1:
        return tier[0]

    max_area = max(c["area"] for c in tier)
    shortlist = [c for c in tier if c["area"] >= 0.5 * max_area]

    best_record, best_score = shortlist[0], -1.0
    cap = cv2.VideoCapture(str(video_path))
    try:
        for record in shortlist:
            crop, _offset = _read_frame_crop(cap, record["frame_idx"], record["box"])
            if crop is None:
                continue
            score = _sharpness_score(crop)
            if score > best_score:
                best_record, best_score = record, score
    finally:
        cap.release()

    return best_record


def _read_frame_crop(cap: cv2.VideoCapture, frame_idx: int, box: list[float]):
    """The padded, raw BGR crop around box at frame_idx from an already-open
    capture, plus its top-left corner in the original frame's own pixel
    coordinates (so a caller can translate the same box into crop-local
    coordinates - see _spotlight_primary) - or (None, None) if the frame/box
    is unusable. Split out from _extract_crop so _pick_best_source can seek
    the same open capture across several candidate frames for one player
    instead of paying to open/close the video file per candidate."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        return None, None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * THUMBNAIL_PADDING_FRACTION
    pad_y = (y2 - y1) * THUMBNAIL_PADDING_FRACTION
    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1)


def _extract_crop(video_path: Path, frame_idx: int, box: list[float]):
    """The padded, raw BGR crop around box at frame_idx, plus its top-left
    corner in the original frame's own pixel coordinates - or (None, None)
    if the frame/box is unusable. Opens its own capture for a single read;
    see _read_frame_crop for the version shared across several reads.
    Shared by _extract_thumbnail/_extract_identification_thumbnail (which
    additionally resize, optionally spotlight the primary subject, and
    JPEG-encode for the UI) and embed_player/auto_identify_from_gallery
    (which want the raw pixels for the appearance encoder, not a re-decoded/
    annotated JPEG)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        return _read_frame_crop(cap, frame_idx, box)
    finally:
        cap.release()


# Colours (BGR) for the primary subject's highlight box - a dark rectangle
# drawn first, then a white one on top at the same coordinates, so the
# visible edge reads as a white rectangle with a darker outline that stays
# visible against any background (a light court floor, a dark jersey, etc.)
# rather than blending into whichever it's drawn over.
PRIMARY_OUTLINE_COLOR = (20, 20, 20)
PRIMARY_FILL_COLOR = (255, 255, 255)
PRIMARY_OUTLINE_THICKNESS = 4
PRIMARY_INNER_THICKNESS = 2

# How much to dim everything outside the primary subject's box - low enough
# that a second person (or the net, the floor) caught in the same frame is
# still visible as context, but unambiguously de-emphasized against the
# full-brightness spotlighted subject, so a human naming this thumbnail
# never mistakes someone else in frame for the person actually being shown.
SPOTLIGHT_DARKEN_FACTOR = 0.25


def _spotlight_primary(crop, box_local: tuple[float, float, float, float]) -> None:
    """Dims everything in `crop` outside box_local (the primary subject's
    own detection box, already translated+scaled into this crop's own
    post-resize pixel coordinates - see _extract_identification_thumbnail)
    and draws a white/dark-outlined rectangle around it (see PRIMARY_*
    above). Only ever called when another player's box actually conflicts
    with this crop (see _has_conflict) - when the subject is alone in
    frame there's nobody to disambiguate from, so the plain crop
    (_extract_thumbnail) is used with no highlight at all. Replaces the
    previous approach of outlining every OTHER detected person
    individually, which could draw several overlapping boxes for what a
    tracking hiccup had (incorrectly) split one real second person into -
    a single spotlight on the actual subject sidesteps that regardless of
    how many other detections land in the same frame. Coordinates are
    clamped to the crop's own bounds first, so the drawn box is always
    fully contained in the thumbnail even if the person's detection box
    was resting right at the original video frame's edge (where
    _extract_crop's own padding had nowhere further to expand into).
    Mutates `crop` in place."""
    crop_h, crop_w = crop.shape[:2]
    x1, y1, x2, y2 = box_local
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(crop_w, int(x2)), min(crop_h, int(y2))

    darkened = (crop.astype(np.float32) * SPOTLIGHT_DARKEN_FACTOR).astype(np.uint8)
    if x2 > x1 and y2 > y1:
        darkened[y1:y2, x1:x2] = crop[y1:y2, x1:x2]
    crop[:] = darkened

    if x2 > x1 and y2 > y1:
        cv2.rectangle(crop, (x1, y1), (x2, y2), PRIMARY_OUTLINE_COLOR, PRIMARY_OUTLINE_THICKNESS)
        cv2.rectangle(crop, (x1, y1), (x2, y2), PRIMARY_FILL_COLOR, PRIMARY_INNER_THICKNESS)


def _decode_resized_crop(
    video_path: Path, frame_idx: int, box: list[float]
) -> tuple[Optional[np.ndarray], Optional[tuple[float, float, float, float]]]:
    """The resized thumbnail crop for box at frame_idx, plus box's own
    coordinates translated into that crop's post-resize pixel space (for
    _spotlight_primary) - or (None, None) if unusable. Shared by
    _extract_thumbnail and _extract_identification_thumbnail so the two
    only differ in whether they call _spotlight_primary on the result."""
    crop, offset = _extract_crop(video_path, frame_idx, box)
    if crop is None:
        return None, None
    # frame[y1:y2, x1:x2] is a view into the decoded frame, not its own
    # array - draw on a copy rather than risk mutating/relying on that.
    crop = crop.copy()

    scale = THUMBNAIL_MAX_DIM / max(crop.shape[0], crop.shape[1])
    if scale < 1.0:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    else:
        scale = 1.0

    ox, oy = offset
    x1, y1, x2, y2 = box
    box_local = ((x1 - ox) * scale, (y1 - oy) * scale, (x2 - ox) * scale, (y2 - oy) * scale)
    return crop, box_local


def _encode_jpeg(crop: np.ndarray) -> Optional[str]:
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


def _extract_thumbnail(video_path: Path, frame_idx: int, box: list[float]) -> Optional[str]:
    """The plain thumbnail - no highlight box, no dimming - used everywhere
    a player's photo shows up (Players/Teams/Stats pages, and the resolved
    Identified/Ignored groups on the identification page itself)."""
    crop, _box_local = _decode_resized_crop(video_path, frame_idx, box)
    if crop is None:
        return None
    return _encode_jpeg(crop)


def _extract_identification_thumbnail(video_path: Path, frame_idx: int, box: list[float]) -> Optional[str]:
    """The spotlighted, white-boxed variant (see _spotlight_primary) - only
    ever generated when the chosen frame has a conflicting second box (see
    _has_conflict), and only ever shown on the player-identification page's
    still-Unidentified tiles, where it's actually needed to tell a human
    which of two people in frame is the one being named."""
    crop, box_local = _decode_resized_crop(video_path, frame_idx, box)
    if crop is None:
        return None
    _spotlight_primary(crop, box_local)
    return _encode_jpeg(crop)


def load_names(output_path: Path) -> dict[str, str]:
    names_file = output_path / config.PLAYER_NAMES_NAME
    if not names_file.exists():
        return {}
    return json.loads(names_file.read_text())


def save_names(output_path: Path, names: dict[str, str]) -> dict[str, str]:
    current = load_names(output_path)

    for stable_id, name in names.items():
        name = name.strip()
        if name:
            current[stable_id] = name
        else:
            current.pop(stable_id, None)

    names_file = output_path / config.PLAYER_NAMES_NAME
    names_file.write_text(json.dumps(current, indent=2))
    return current


def load_ignored(output_path: Path) -> set[int]:
    ignored_file = output_path / config.PLAYER_IGNORED_NAME
    if not ignored_file.exists():
        return set()
    return set(json.loads(ignored_file.read_text()))


def save_ignored(output_path: Path, ignored: set[int]) -> list[int]:
    result = sorted(ignored)
    ignored_file = output_path / config.PLAYER_IGNORED_NAME
    ignored_file.write_text(json.dumps(result))
    return result


# Standard court dimensions (CourtDefinition.court.COURT_LENGTH/COURT_WIDTH)
# - duplicated here rather than cross-imported, matching how ballDetection.py
# keeps its own copy of court.json's schema instead of reaching into the
# Analysis package.
_COURT_LENGTH_M = 18.0
_COURT_WIDTH_M = 9.0

# How far outside the marked court lines a player's foot position can still
# land and count as "on the court" for recalibrate_players. The two
# directions get different allowances rather than one shared margin: behind
# the baselines (the _COURT_LENGTH_M ends - x, below) is where a server
# takes their approach and a deep defender/libero ranges chasing an
# overpass, routinely several metres back, so that margin stays generous.
# Beside the sidelines (the _COURT_WIDTH_M ends - y, below) there's no
# equivalent in-play reason to be far from the lines - what's out there is
# the bench, staff, and spectators, not a player mid-rally - so that margin
# stays tight, just enough for a real wide dig's follow-through. A
# stable_id whose foot position NEVER falls within these margins across the
# whole video is almost certainly not a player at all, not a player who
# just played close to the edge.
PLAYER_BASELINE_MARGIN_M = 5.0
PLAYER_SIDELINE_MARGIN_M = 1.5


def _load_homography(output_path: Path):
    court_file = output_path / config.COURT_FILE_NAME
    if not court_file.exists():
        return None
    try:
        data = json.loads(court_file.read_text())
        return np.array(data["homography"], dtype=np.float64)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _pixel_to_court(x: float, y: float, matrix) -> tuple[float, float]:
    point = np.array([[[x, y]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(point, matrix)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def recalibrate_players(output_path: Path) -> int:
    """
    Re-derives player_positions.json's `court` field (each frame's box-centre
    projected through the homography, same as tracker_offline.py's own phase
    3) from whatever court.json says *now*, and auto-ignores any stable_id
    that was never once inside the (generously padded) court area for the
    whole video - almost always bench/staff/a spectator the detector picked
    up, not a real player. Cheap: pure JSON in/out, no video decode or
    re-detection, since tracker_offline.py already tracks and keeps everyone
    regardless of position (see its load_court_polygon docstring) - only
    which stable_ids count as "on the court" changes when calibration does.

    Never un-ignores anyone a human already ignored manually, and never
    ignores a stable_id that spends even one frame plausibly on the court -
    a real player diving, chasing a wide ball, or just standing near a line
    should never be silently hidden; that's still a human call via the Setup
    tab's Player Identification page.

    Returns how many stable_ids were newly auto-ignored.
    """
    positions_file = output_path / PLAYER_POSITIONS_NAME
    if not positions_file.exists():
        return 0

    matrix = _load_homography(output_path)
    positions = json.loads(positions_file.read_text())

    ever_seen: set[int] = set()
    ever_in_range: set[int] = set()

    for entry in positions:
        for player in entry["players"]:
            stable_id = player["stable_id"]
            ever_seen.add(stable_id)

            x1, y1, x2, y2 = player["box"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            court = _pixel_to_court(cx, cy, matrix) if matrix is not None else None
            player["court"] = list(court) if court is not None else None

            if matrix is None:
                # No calibration at all - can't judge range, so don't
                # auto-ignore anyone; err toward leaving it to a human.
                ever_in_range.add(stable_id)
                continue

            # The range check itself uses a foot point (bottom-centre)
            # rather than the box centre stored above - a standing player's
            # foot is much closer to the calibrated ground plane, so it
            # doesn't systematically read as "off court" the way an
            # elevated box-centre would for someone just standing normally
            # in-bounds (see BallDetection.ballDetection.pixel_to_court's
            # own docstring on this same elevation bias).
            # fx runs along the baselines (0..COURT_LENGTH, net at the
            # midpoint), fy along the sidelines (0..COURT_WIDTH) - see
            # court.py's own corner layout - so the generous/tight margins
            # above apply to the axis they're actually named for.
            fx, fy = _pixel_to_court((x1 + x2) / 2.0, y2, matrix)
            if -PLAYER_BASELINE_MARGIN_M <= fx <= _COURT_LENGTH_M + PLAYER_BASELINE_MARGIN_M and \
                    -PLAYER_SIDELINE_MARGIN_M <= fy <= _COURT_WIDTH_M + PLAYER_SIDELINE_MARGIN_M:
                ever_in_range.add(stable_id)

    positions_file.write_text(json.dumps(positions, indent=2))

    never_in_range = ever_seen - ever_in_range
    if not never_in_range:
        return 0

    ignored = load_ignored(output_path)
    newly_ignored = never_in_range - ignored
    if newly_ignored:
        save_ignored(output_path, ignored | newly_ignored)

    return len(newly_ignored)


def load_player_confirmed(output_path: Path) -> bool:
    """Whether the user has explicitly signed off on this video's player
    identification - same state-management pattern as scoring's
    score_config.json "confirmed" flag (see score.py), just for player
    identification instead."""
    cfg_file = output_path / PLAYER_CONFIG_NAME
    if not cfg_file.exists():
        return False
    try:
        return bool(json.loads(cfg_file.read_text()).get("confirmed", False))
    except json.JSONDecodeError:
        return False


def save_player_confirmed(output_path: Path, confirmed: bool) -> bool:
    cfg_file = output_path / PLAYER_CONFIG_NAME
    cfg_file.write_text(json.dumps({"confirmed": confirmed}))
    return confirmed


def _load_thumbnail_cache(output_path: Path) -> dict:
    cache_file = output_path / THUMBNAIL_CACHE_NAME
    if not cache_file.exists():
        return {}
    try:
        return json.loads(cache_file.read_text())
    except json.JSONDecodeError:
        return {}


def _save_thumbnail_cache(output_path: Path, cache: dict):
    cache_file = output_path / THUMBNAIL_CACHE_NAME
    cache_file.write_text(json.dumps(cache))


def list_players(video_path: Path, output_path: Path, with_thumbnails: bool = True) -> list[dict]:
    positions = _load_player_positions(output_path)
    ranked = _ranked_candidates_per_player(positions, _frame_size(video_path))
    names = load_names(output_path)
    ignored = load_ignored(output_path)

    # Extracting a thumbnail means opening the video and seeking to a
    # specific frame - the single slowest part of this endpoint, and one
    # that's completely wasted work on every repeat call (the Setup tab,
    # PlayerReview, and the Stats page - which does this for every
    # completed job at once) since the chosen frame/box for a given
    # stable_id never changes once tracking has finished. Cache it to disk
    # instead of re-seeking the video every time. Validity is keyed off the
    # top of the (cheap, pure-Python) candidate ranking rather than the
    # actually-chosen frame_idx, since the latter can depend on a blur
    # comparison we'd rather not redo on every request just to find out the
    # cache is still good - the ranking itself only changes if
    # player_positions.json does (re-tracking), which a version bump above
    # already guards against for logic-only changes.
    cache = _load_thumbnail_cache(output_path) if with_thumbnails else {}
    cache_dirty = False

    players = []
    for stable_id in sorted(ranked):
        candidates = ranked[stable_id]
        top_candidate = candidates[0]
        thumbnail = None
        identification_thumbnail = None
        chosen = top_candidate

        if with_thumbnails:
            cache_key = str(stable_id)
            cached = cache.get(cache_key)
            if (
                cached is not None
                and cached.get("v") == THUMBNAIL_CACHE_VERSION
                and cached.get("top_frame_idx") == top_candidate["frame_idx"]
            ):
                thumbnail = cached["thumbnail_base64"]
                identification_thumbnail = cached.get("identification_thumbnail_base64")
                chosen = next(
                    (c for c in candidates if c["frame_idx"] == cached.get("chosen_frame_idx")), top_candidate
                )
            else:
                chosen = _pick_best_source(video_path, candidates) or top_candidate
                thumbnail = _extract_thumbnail(video_path, chosen["frame_idx"], chosen["box"])
                if chosen["conflict"]:
                    identification_thumbnail = _extract_identification_thumbnail(
                        video_path, chosen["frame_idx"], chosen["box"]
                    )
                cache[cache_key] = {
                    "thumbnail_base64": thumbnail,
                    "identification_thumbnail_base64": identification_thumbnail,
                    "top_frame_idx": top_candidate["frame_idx"],
                    "chosen_frame_idx": chosen["frame_idx"],
                    "v": THUMBNAIL_CACHE_VERSION,
                }
                cache_dirty = True

        players.append({
            "stable_id": stable_id,
            "name": names.get(str(stable_id)),
            "appearances": chosen["appearances"],
            "thumbnail_base64": thumbnail,
            "identification_thumbnail_base64": identification_thumbnail,
            "thumbnail_frame_idx": chosen["frame_idx"],
            "thumbnail_timestamp_s": chosen["timestamp_s"],
            "ignored": stable_id in ignored,
        })

    if cache_dirty:
        _save_thumbnail_cache(output_path, cache)

    return players


def embed_player(video_path: Path, output_path: Path, stable_id: int) -> Optional[list]:
    """The appearance embedding for one player's own best crop - used to add
    a newly-named player into the global cross-video gallery (see
    player_gallery.remember, called from the /players/names endpoint right
    after a name is saved) so a *future* video's still-unnamed detections
    can be matched against them."""
    positions = _load_player_positions(output_path)
    best = _best_crop_per_player(positions, _frame_size(video_path))
    record = best.get(stable_id)
    if record is None:
        return None

    crop, _offset = _extract_crop(video_path, record["frame_idx"], record["box"])
    if crop is None:
        return None

    from . import player_gallery

    return player_gallery.embed_crop(crop)


def auto_identify_from_gallery(video_path: Path, output_path: Path) -> int:
    """Runs once, right after player tracking finishes for a video (see
    pipeline._phase_one) - compares every detected player's appearance
    against the global cross-video gallery (player_gallery.py) and writes
    in any confident, unambiguous match directly to player_names.json
    before a human ever opens the Setup tab's Player Identification page.
    Already-named or already-ignored players are left untouched. This is
    deliberately conservative (see player_gallery's distance/margin
    thresholds) - a wrong auto-name would silently corrupt that person's
    stats, so it only ever acts on matches confident enough that a human
    reviewing them would agree. Returns how many players were auto-matched,
    for the pipeline log."""
    from . import player_gallery

    positions = _load_player_positions(output_path)
    if not positions:
        return 0

    best = _best_crop_per_player(positions, _frame_size(video_path))
    names = load_names(output_path)
    ignored = load_ignored(output_path)

    matched: dict[str, str] = {}
    for stable_id, record in best.items():
        if stable_id in ignored or names.get(str(stable_id)):
            continue
        crop, _offset = _extract_crop(video_path, record["frame_idx"], record["box"])
        if crop is None:
            continue
        embedding = player_gallery.embed_crop(crop)
        if embedding is None:
            continue
        result = player_gallery.match(embedding)
        if result is None:
            continue
        matched[str(stable_id)] = result[0]

    if matched:
        save_names(output_path, matched)

    return len(matched)


def load_candidate_matches(output_path: Path) -> list[dict]:
    """Pairs of still-separate stable_ids the tracker's appearance model
    thought might be the same person re-appearing after being lost, but
    wasn't confident enough to auto-merge - see tracker_offline._find_
    candidate_matches(). Filtered down to pairs that are still actually
    useful to show a human: neither side ignored, and not already resolved
    by the human giving both the same name (an unresolved pair can still
    have exactly one side named - that's a hint the other should get the
    same name) or explicitly rejected (both named, but differently)."""
    matches_file = output_path / CANDIDATE_MATCHES_NAME
    if not matches_file.exists():
        return []

    names = load_names(output_path)
    ignored = load_ignored(output_path)
    raw = json.loads(matches_file.read_text())

    result = []
    for entry in raw:
        a, b = entry["a"], entry["b"]
        if a in ignored or b in ignored:
            continue

        name_a, name_b = names.get(str(a)), names.get(str(b))
        if name_a and name_b and name_a != name_b:
            continue  # human already looked at both and decided they differ

        result.append(entry)

    return result


def build_canonical_mapping(names: dict[str, str]) -> dict[int, int]:
    """Two or more stable_ids assigned the same name are treated as one
    person - this is how a human corrects the identity bank occasionally
    splitting one real player into two stable_ids (e.g. after a long
    occlusion). Maps every non-canonical member of such a group to the
    smallest stable_id in that group; ids with a unique or blank name are
    left out entirely (they map to themselves)."""
    groups: dict[str, list[int]] = {}
    for stable_id_str, name in names.items():
        name = name.strip()
        if not name:
            continue
        groups.setdefault(name, []).append(int(stable_id_str))

    mapping: dict[int, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = min(members)
        for stable_id in members:
            if stable_id != canonical:
                mapping[stable_id] = canonical

    return mapping


def write_grouped_actions(output_path: Path) -> Path:
    """Writes a copy of actions.json with grouped stable_ids remapped to
    their group's canonical id and ignored players' actions dropped
    entirely, for consolidateStats to read instead of the raw per-detection
    actions log - actions.json itself is left untouched so re-grouping or
    un-ignoring later is always working from the original data."""
    actions_file = output_path / ACTIONS_NAME
    grouped_file = output_path / GROUPED_ACTIONS_NAME

    actions = json.loads(actions_file.read_text()) if actions_file.exists() else []
    mapping = build_canonical_mapping(load_names(output_path))
    ignored = load_ignored(output_path)

    result = []
    for action in actions:
        stable_id = action.get("player_stable_id")
        if stable_id is None:
            result.append(action)
            continue

        canonical = mapping.get(stable_id, stable_id)
        if stable_id in ignored or canonical in ignored:
            continue

        if canonical != stable_id:
            action = {**action, "player_stable_id": canonical}
        result.append(action)

    grouped_file.write_text(json.dumps(result, indent=2))
    return grouped_file


def merge_names_into_stats(output_path: Path):
    """Stamps whatever names were assigned during player review onto
    player_stats.json, so downstream consumers (dashboard, frontend results
    view) can show a name instead of a bare stable_id without a separate
    fetch of player_names.json."""
    stats_file = output_path / STATS_NAME
    if not stats_file.exists():
        return

    stats = json.loads(stats_file.read_text())
    names = load_names(output_path)

    for player_id, record in stats.get("players", {}).items():
        record["name"] = names.get(player_id)

    stats_file.write_text(json.dumps(stats, indent=2))
