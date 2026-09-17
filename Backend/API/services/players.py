import base64
import functools
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .. import config
from ..resultcache import ttl_cache

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
THUMBNAIL_CACHE_VERSION = 5

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


@functools.lru_cache(maxsize=16)
def _load_player_positions_cached(path_str: str, mtime_ns: int) -> list[dict]:
    """mtime_ns (not just path_str) is part of the cache key so a
    reprocessed/re-tracked job (which rewrites this file) invalidates
    itself automatically instead of serving stale positions forever."""
    return json.loads(Path(path_str).read_text())


def load_player_positions(output_path: Path) -> list[dict]:
    """player_positions.json is one entry per frame for the whole video -
    100-250+ MB is normal for a real match (see docs/assets/README.md's
    perf notes) - parsing it is genuinely expensive, and this is called
    once per completed job by both PlayersPage (getPlayers) and every
    team/player radar computation (teams.assign_teams), so without caching
    a single page view can re-parse hundreds of MB of JSON many times over.
    The cache is keyed on mtime so it stays correct across reprocessing."""
    positions_file = output_path / PLAYER_POSITIONS_NAME
    if not positions_file.exists():
        return []
    return _load_player_positions_cached(str(positions_file), positions_file.stat().st_mtime_ns)


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


# How coarsely a candidate appearance's isolation is bucketed before box size
# is allowed to break the tie, in multiples of the subject's own box width
# (see _isolation_score). Without a step, isolation is a continuous number
# that no two appearances ever tie on, so it would decide every ranking on
# its own and a frame where the player is a handful of pixels tall could beat
# a big clear one for being a hair further from the nearest neighbour. A
# quarter of a body width is about the point past which two crops are equally
# uncluttered to look at, so within a step the larger crop wins as before.
ISOLATION_BUCKET_WIDTHS = 0.25


def _isolation_score(box: list[float], other_boxes: list[list[float]]) -> float:
    """How much room this appearance has to itself: the distance from the
    subject's box centre to the nearest other player's box centre, in
    multiples of the subject's own box width, bucketed by
    ISOLATION_BUCKET_WIDTHS. Higher is cleaner.

    Graded rather than the yes/no _has_conflict above, because on real
    volleyball footage almost every appearance conflicts with something -
    measured on a real match, the nearest other detection sits a median of
    0.2 body widths away and only about 2% of appearances are properly
    isolated. Ranking on the boolean therefore puts nearly every appearance
    of a player into one undifferentiated "conflicted" tier and then picks
    purely on size, which is how a player with a perfectly clean moment
    somewhere in his five seconds on screen still ends up represented by a
    crop with someone standing across him. The same reasoning drives
    PlayerDetection.tracker's own gallery selection (see its
    GALLERY_ISOLATION_WHEN_ALONE).

    Scaled by the subject's own width so it means the same thing for a
    player near the camera and one at the far baseline.
    """
    centre_x = (box[0] + box[2]) / 2.0
    width = max(1.0, box[2] - box[0])

    gaps = [
        abs((other[0] + other[2]) / 2.0 - centre_x) / width
        for other in other_boxes
    ]
    if not gaps:
        return float("inf")

    return round(min(gaps) / ISOLATION_BUCKET_WIDTHS)


def _ranked_candidates_per_player(
    positions: list[dict], frame_size: Optional[tuple[float, float]] = None
) -> dict[int, list[dict]]:
    """For each stable_id, every appearance that plausibly looks like a
    real, unclipped person (falling back to every appearance at all if none
    do, so a thumbnail is always produced - see _is_plausible_person_box),
    ranked best-first: the appearances where the player has most room to
    themselves first (see _isolation_score), largest box within each
    isolation bucket next. Plain "largest box wins" alone occasionally picks
    a spurious detection - a fused box, a shadow, a sideline ad board - that
    happens to be big; the plausibility filter is what catches that. The
    actual thumbnail source is then picked from the front of this ranking by
    _pick_best_source, which additionally screens the top candidates for
    motion blur.

    Isolation leads the ranking because this crop is what a human is asked
    to identify someone from, and - for the callers that feed it to the
    appearance encoder rather than the screen (_best_crop_per_player, and so
    embed_player and auto_identify_from_gallery) - what a match is decided
    on. A player who was alone in frame for one second of their five should
    be represented by that second, not by whichever moment happened to have
    the biggest box. Where a player genuinely never had a clean moment the
    ranking still returns their least-bad one; nothing is discarded for
    being crowded."""
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
                "isolation": _isolation_score(box, other_boxes),
                # Still a plain yes/no, because what it drives is a yes/no:
                # whether this thumbnail needs the white highlight box drawn
                # on it to say which person is the subject.
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
        result[stable_id] = sorted(records, key=lambda r: (-r["isolation"], -r["area"]))

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
# Thin: the box is there to say which person is the subject, and a heavy
# stroke on a ~220px-wide thumbnail eats into the very crop it is pointing
# at - on a distant player it can cover most of them.
PRIMARY_OUTLINE_THICKNESS = 2
PRIMARY_INNER_THICKNESS = 1

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


# Standard court dimensions (CourtDetection.court.COURT_LENGTH/COURT_WIDTH)
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

# A stable_id whose in-range frames fall below this fraction of its own
# total screen time is auto-ignored. Replaces an earlier "ever in range,
# even once" rule, which meant a bench player, coach, or spectator who
# happened to lean onto the sideline or step behind the baseline just once
# across a whole match-length video was never auto-ignored at all - on real
# footage that let dozens of off-court people accumulate as permanent
# "players" alongside the real ~12-14 on the roster. A real player's
# occasional excursion near a line stays comfortably above this bar (their
# screen time is overwhelmingly on-court); this only catches someone whose
# presence on screen is overwhelmingly off-court.
MIN_IN_RANGE_FRACTION = 0.15


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
    projected through the homography) and its `court_ground` field (the same
    frame's foot point, i.e. where the player actually stands), same as
    trackplayers_offline's own phase 3, from whatever court.json says *now*,
    and auto-ignores any
    stable_id whose in-range frames fall below MIN_IN_RANGE_FRACTION of its
    own total screen time - almost always bench/staff/a spectator the
    detector picked up, not a real player. Cheap: pure JSON in/out, no video
    decode or re-detection, since trackplayers_offline already tracks and
    keeps everyone regardless of position (see its load_court_polygon
    docstring) - only which stable_ids count as "on the court" changes when
    calibration does.

    Never un-ignores anyone a human already ignored manually. A stable_id
    that's overwhelmingly on-court - a real player diving, chasing a wide
    ball, or just standing near a line for a handful of frames - stays
    comfortably above MIN_IN_RANGE_FRACTION and is never touched; a human
    can still always override either direction via the Setup tab's Player
    Identification page.

    Returns how many stable_ids were newly auto-ignored.
    """
    positions_file = output_path / PLAYER_POSITIONS_NAME
    if not positions_file.exists():
        return 0

    matrix = _load_homography(output_path)
    positions = json.loads(positions_file.read_text())

    seen_count: dict[int, int] = {}
    in_range_count: dict[int, int] = {}

    for entry in positions:
        for player in entry["players"]:
            stable_id = player["stable_id"]
            seen_count[stable_id] = seen_count.get(stable_id, 0) + 1

            x1, y1, x2, y2 = player["box"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            court = _pixel_to_court(cx, cy, matrix) if matrix is not None else None
            player["court"] = list(court) if court is not None else None

            if matrix is None:
                # No calibration at all - can't judge range, so don't
                # auto-ignore anyone; err toward leaving it to a human.
                player["court_ground"] = None
                in_range_count[stable_id] = in_range_count.get(stable_id, 0) + 1
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
            # Kept as court_ground rather than discarded after the range
            # check: it is the player's real standing position, and anything
            # comparing a player against fixed court geometry (which side of
            # the net, distance to the net) needs it rather than the
            # elevation-biased box-centre `court` above - see
            # PlayerDetection.tracker._replay_and_export's own note.
            player["court_ground"] = [fx, fy]

            if -PLAYER_BASELINE_MARGIN_M <= fx <= _COURT_LENGTH_M + PLAYER_BASELINE_MARGIN_M and \
                    -PLAYER_SIDELINE_MARGIN_M <= fy <= _COURT_WIDTH_M + PLAYER_SIDELINE_MARGIN_M:
                in_range_count[stable_id] = in_range_count.get(stable_id, 0) + 1

    positions_file.write_text(json.dumps(positions, indent=2))

    mostly_out_of_range = {
        stable_id for stable_id, total in seen_count.items()
        if in_range_count.get(stable_id, 0) / total < MIN_IN_RANGE_FRACTION
    }
    if not mostly_out_of_range:
        return 0

    ignored = load_ignored(output_path)
    newly_ignored = mostly_out_of_range - ignored
    if newly_ignored:
        save_ignored(output_path, ignored | newly_ignored)

    return len(newly_ignored)


IDENTITY_GRAPH_NAME = "player_identity_graph.json"


def identification_is_provisional(output_path: Path) -> bool:
    """Whether this job's player identities are one-per-tracklet
    placeholders from a pass that ran before the court was calibrated,
    rather than a real matching - see PlayerDetection.tracker._consolidate.

    Everything that resolves identity is court-derived (which side of the
    net, rotation zone, physical continuity in metres, and the court gate
    that tells a player from a spectator), so tracking that runs first -
    which is the normal order, calibration being a Setup-tab step - can only
    hand back its raw tracklets and wait. Naming those is refused until the
    court exists, because re-consolidation renumbers every id and a name
    pinned to a placeholder would silently come back pointing at somebody
    else.
    """
    graph_file = output_path / IDENTITY_GRAPH_NAME
    if not graph_file.exists():
        return False
    try:
        return bool(json.loads(graph_file.read_text()).get("provisional", False))
    except (json.JSONDecodeError, OSError):
        return False


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


@ttl_cache()
def list_players(video_path: Path, output_path: Path, with_thumbnails: bool = True) -> list[dict]:
    """Every detection in the job, with a thumbnail for the ones a reviewer
    still has to make a decision about. Ignored detections come back with
    thumbnail_base64 None - see the loop below for why."""
    positions = load_player_positions(output_path)
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

        # An ignored detection never gets a photo. It is excluded from stats
        # and overlays already, and its tile exists only so a human can put
        # it back - which the id, the timestamp and the whole-frame preview
        # behind the tile's magnifier all answer without decoding anything
        # up front. Generating one costs a video seek and rides back in the
        # payload as base64, and auto-ignore routinely produces hundreds of
        # these on a job whose court gate could not run at tracking time
        # (see recalibrate_players), which is exactly the case where the
        # page was slowest and the photos least worth having.
        if with_thumbnails and stable_id not in ignored:
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


# Longest side of the whole-frame preview served behind a thumbnail's
# magnifier (see extract_player_frame). Full 1080p would be several hundred
# KB per open for no extra readability in a dialog.
FRAME_PREVIEW_MAX_DIM = 1280

# The highlight box on a whole frame needs a heavier stroke than the one
# drawn on a 220px thumbnail (PRIMARY_*_THICKNESS): the same 1-2px at full
# frame width is a hairline, and the whole point of this view is to find the
# subject in a crowd at a glance.
FRAME_PREVIEW_OUTLINE_THICKNESS = 6
FRAME_PREVIEW_INNER_THICKNESS = 3


def extract_player_frame(video_path: Path, output_path: Path, stable_id: int) -> Optional[bytes]:
    """The whole source frame a player's thumbnail was cut from, with the
    same white box drawn around them - what the magnifier on an unidentified
    tile opens.

    The thumbnail alone answers "what does this person look like" but throws
    away everything a reviewer actually uses to place someone: who they were
    standing next to, where on the court they were, what was happening. This
    hands that back without changing the thumbnail itself.

    Deliberately reads the frame the CACHED thumbnail was taken from where
    there is one, rather than re-deriving it: _pick_best_source breaks ties
    on a blur score it would have to re-measure, and a preview of a
    different moment than the tile it opened from would be worse than no
    preview. Falls back to the head of the ranking when nothing is cached
    yet, which is the same frame list_players itself would start from.

    Returns encoded JPEG bytes, or None if the player or frame is unusable.
    """
    ranked = _ranked_candidates_per_player(
        load_player_positions(output_path), _frame_size(video_path)
    )
    candidates = ranked.get(stable_id)
    if not candidates:
        return None

    cached = _load_thumbnail_cache(output_path).get(str(stable_id)) or {}
    chosen = next(
        (c for c in candidates if c["frame_idx"] == cached.get("chosen_frame_idx")),
        candidates[0],
    )

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, chosen["frame_idx"])
        success, frame = cap.read()
    finally:
        cap.release()

    if not success or frame is None:
        return None

    x1, y1, x2, y2 = (int(v) for v in chosen["box"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), PRIMARY_OUTLINE_COLOR, FRAME_PREVIEW_OUTLINE_THICKNESS)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PRIMARY_FILL_COLOR, FRAME_PREVIEW_INNER_THICKNESS)

    height, width = frame.shape[:2]
    scale = min(1.0, FRAME_PREVIEW_MAX_DIM / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buffer.tobytes() if ok else None


def embed_player(video_path: Path, output_path: Path, stable_id: int) -> Optional[list]:
    """The appearance embedding for one player's own best crop - used to add
    a newly-named player into the global cross-video gallery (see
    player_gallery.remember, called from the /players/names endpoint right
    after a name is saved) so a *future* video's still-unnamed detections
    can be matched against them."""
    positions = load_player_positions(output_path)
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

    positions = load_player_positions(output_path)
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
    wasn't confident enough to auto-merge - see
    PlayerDetection.tracker._find_candidate_matches(). Filtered down to
    pairs that are still actually
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


def _frames_by_stable_id(output_path: Path, wanted: set[int]) -> dict[int, set[int]]:
    """Which frames each of `wanted` appears in - the raw material for every
    "can these two be the same person?" check here, since one person cannot
    be in two places at once. Restricted to the ids actually being asked
    about, because a full match's position log is hundreds of MB and the
    callers only ever care about a handful of ids."""
    frames: dict[int, set[int]] = {}
    for entry in load_player_positions(output_path):
        frame_idx = entry["frame_idx"]
        for player in entry.get("players", []):
            stable_id = player["stable_id"]
            if stable_id in wanted:
                frames.setdefault(stable_id, set()).add(frame_idx)
    return frames


def build_candidate_groups(output_path: Path) -> list[dict]:
    """The candidate pairs (see load_candidate_matches) joined up into
    groups of detections that are probably all one person, for the review UI
    to offer as a single "name all of these at once" unit instead of making
    a human rediscover the same grouping by eye across dozens of tiles.

    A group is only ever grown into a CLIQUE: two groups join only if every
    cross-pair between them is itself a candidate match, not merely the one
    pair that happened to link them. Plain transitive chaining - A-B plus
    B-C gives {A, B, C} even though A and C were never matched - was tried
    first and measured against a real job's completed, hand-done
    identification. It is the single biggest source of wrong groups: one bad
    pair anywhere fuses two people's fragments for good, and on that job
    only 5 of 15 groups held one person, the worst mixing three. Requiring
    the clique took it to 13 of 24 groups pure and left no group mixing more
    than two people.
    
    Tightening the per-pair evidence instead does not work, which is why
    this is a linkage rule rather than another threshold. Cutting 17% of the
    wrong pairs (via an independent stature signal, and via a tighter
    appearance bar) changed group purity on that job by exactly nothing:
    enough wrong pairs survive to chain the same clusters together anyway.
    Only refusing to chain at all breaks it.

    The cost is real and deliberate: a clique splits some people across two
    groups rather than risking one group holding two people. That is the
    right way round, because the two mistakes are not equally expensive to
    the reviewer - naming two groups the same name reunites them for free
    (see build_canonical_mapping), while spotting one stranger among twenty
    tiles and deselecting it is work, and missing them corrupts stats.

    The one hard rule is the one that isn't a matter of taste: a group can
    never contain two detections that share a frame, because that would be
    proposing a player is in two places at once - the same thing
    find_simultaneous_name_conflicts refuses to let a human save by hand.
    Groups are grown strongest-link-first so that when that rule blocks a
    join, it's the weakest evidence that loses.

    Each group carries the weakest link holding it together (its lowest
    pairwise confidence) rather than the strongest, so the number shown next
    to it is what the group is actually worth, and a suggested_name when
    some member has already been named - that member is the answer to "who
    are these people" already.
    """
    matches = load_candidate_matches(output_path)
    if not matches:
        return []

    members = {stable_id for match in matches for stable_id in (match["a"], match["b"])}
    frames = _frames_by_stable_id(output_path, members)
    names = load_names(output_path)
    named = {stable_id: names[str(stable_id)].strip() for stable_id in members
             if names.get(str(stable_id), "").strip()}

    # stable_id -> the group it currently belongs to, and that group's own
    # member list/confidence, joined strongest link first.
    group_of = {stable_id: stable_id for stable_id in members}
    groups = {stable_id: [stable_id] for stable_id in members}
    confidence = {stable_id: 1.0 for stable_id in members}

    matched = {(min(m["a"], m["b"]), max(m["a"], m["b"])): m["confidence"] for m in matches}

    for match in sorted(matches, key=lambda m: -m["confidence"]):
        a, b = group_of[match["a"]], group_of[match["b"]]
        if a == b:
            confidence[a] = min(confidence[a], match["confidence"])
            continue

        if any(frames.get(x, set()) & frames.get(y, set()) for x in groups[a] for y in groups[b]):
            continue

        # The clique rule: everyone in one group has to be a candidate match
        # for everyone in the other, or this is chaining rather than
        # grouping.
        cross = [matched.get((min(x, y), max(x, y))) for x in groups[a] for y in groups[b]]
        if any(c is None for c in cross):
            continue

        # A human has already said these are different people. Individual
        # pairs named differently are dropped by load_candidate_matches, but
        # transitivity routes around that - A named Kai, C named Marisa, and
        # an unnamed B matched to both would otherwise chain all three into
        # one group and then suggest a single name for it.
        if {named[x] for x in groups[a] if x in named} != {named[y] for y in groups[b] if y in named}                 and ({named[x] for x in groups[a] if x in named}
                     and {named[y] for y in groups[b] if y in named}):
            continue

        groups[a] += groups[b]
        confidence[a] = min([confidence[a], confidence[b]] + cross)
        for stable_id in groups[b]:
            group_of[stable_id] = a
        del groups[b], confidence[b]

    result = []
    for root, stable_ids in groups.items():
        if len(stable_ids) < 2:
            continue
        suggested = next((named[stable_id] for stable_id in sorted(stable_ids)
                          if stable_id in named), None)
        result.append({
            "stable_ids": sorted(stable_ids),
            "confidence": round(confidence[root], 3),
            "suggested_name": suggested,
        })

    # Biggest first: the group that saves the most clicks is the one worth
    # looking at first.
    result.sort(key=lambda g: (-len(g["stable_ids"]), -g["confidence"]))
    return result


def find_simultaneous_name_conflicts(
    output_path: Path, names: dict[str, str], ignored: Optional[set[int]] = None
) -> list[dict]:
    """Same name on two identities that are on screen at the same moment.

    Naming two stable_ids the same is the normal way to tell the system they
    are one person (see build_canonical_mapping) - but only if they never
    coexist, because one player cannot be in two places at once. When they
    do coexist the naming is describing something impossible: either the
    tracker split two different people badly enough that both looked like
    the same person to a reviewer, or the reviewer simply mislabelled one.
    Either way, merging them would fuse two people's stats together while
    the rendered video shows the name twice on court simultaneously.

    Ignored stable_ids are skipped - a duplicate that has been ignored is
    already excluded from stats and overlays, so it isn't on court in any
    sense that matters.

    Returns one entry per offending pair, with enough detail for the caller
    to point a human at the exact moment to look at.
    """
    ignored = ignored or set()

    groups: dict[str, list[int]] = {}
    for stable_id_str, name in names.items():
        clean = name.strip()
        stable_id = int(stable_id_str)
        if not clean or stable_id in ignored:
            continue
        groups.setdefault(clean, []).append(stable_id)

    groups = {name: ids for name, ids in groups.items() if len(ids) > 1}
    if not groups:
        return []

    watched = {stable_id for ids in groups.values() for stable_id in ids}
    frames: dict[int, set[int]] = {}
    timestamps: dict[int, float] = {}
    for entry in load_player_positions(output_path):
        frame_idx = entry["frame_idx"]
        for player in entry.get("players", []):
            stable_id = player["stable_id"]
            if stable_id in watched:
                frames.setdefault(stable_id, set()).add(frame_idx)
                timestamps[frame_idx] = entry.get("timestamp_s", 0.0)

    conflicts = []
    for name, ids in sorted(groups.items()):
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                shared = frames.get(ids[i], set()) & frames.get(ids[j], set())
                if not shared:
                    continue
                first = min(shared)
                conflicts.append({
                    "name": name,
                    "stable_ids": [ids[i], ids[j]],
                    "frames": len(shared),
                    "first_frame": first,
                    "first_timestamp_s": round(timestamps.get(first, 0.0), 1),
                })

    return conflicts


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
