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
# though its frame_idx hasn't changed (e.g. adding the "outline other people
# in frame" overlay below) - a cache entry stamped with an older version is
# treated as a miss and regenerated once, rather than serving a stale
# thumbnail indefinitely.
THUMBNAIL_CACHE_VERSION = 2

# A detector's box is often a touch tight around the actual person - pad it
# outward proportionally (same fraction for everyone, so crops stay
# consistent) rather than cropping exactly to the box, which regularly
# clips the top of someone's head or their feet.
THUMBNAIL_PADDING_FRACTION = 0.18


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


def _best_crop_per_player(positions: list[dict], frame_size: Optional[tuple[float, float]] = None) -> dict[int, dict]:
    """For each stable_id, pick the frame/box to use as its thumbnail: the
    largest box among the ones that plausibly look like a real, unclipped
    person, falling back to the largest box overall if none do (so a
    thumbnail is always produced). Plain "largest box wins" alone
    occasionally picks a spurious detection - a fused box, a shadow, a
    sideline ad board - that happens to be big, producing a thumbnail with
    no player in it; the plausibility filter above is what catches that."""
    best: dict[int, dict] = {}
    best_plausible: dict[int, dict] = {}
    appearances: dict[int, int] = {}

    for entry in positions:
        frame_idx = entry["frame_idx"]
        timestamp_s = entry.get("timestamp_s")
        for player in entry["players"]:
            stable_id = player["stable_id"]
            appearances[stable_id] = appearances.get(stable_id, 0) + 1

            box = player["box"]
            area = _box_area(box)
            record = {"frame_idx": frame_idx, "timestamp_s": timestamp_s, "box": box, "area": area}

            current = best.get(stable_id)
            if current is None or area > current["area"]:
                best[stable_id] = record

            if _is_plausible_person_box(box, frame_size):
                current_plausible = best_plausible.get(stable_id)
                if current_plausible is None or area > current_plausible["area"]:
                    best_plausible[stable_id] = record

    for stable_id in best:
        chosen = best_plausible.get(stable_id, best[stable_id])
        chosen["appearances"] = appearances[stable_id]
        best[stable_id] = chosen

    return best


def _extract_crop(video_path: Path, frame_idx: int, box: list[float]):
    """The padded, raw BGR crop around box at frame_idx, plus its top-left
    corner in the original frame's own pixel coordinates (so a caller can
    translate some other detection's box into crop-local coordinates - see
    _outline_other_people) - or (None, None) if the frame/box is unusable.
    Shared by _extract_thumbnail (which additionally resizes, outlines any
    other person caught in the same frame, and JPEG-encodes it for the UI)
    and embed_player/auto_identify_from_gallery (which want the raw pixels
    for the appearance encoder, not a re-decoded/annotated JPEG)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
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
    finally:
        cap.release()


# Colours (BGR) for outlining a second detected person inside a thumbnail
# crop - a dark rectangle drawn first, then a white one on top at the same
# coordinates, so the visible edge reads as a white rectangle with a darker
# outline that stays visible against any background (a light court floor, a
# dark jersey, etc.) rather than blending into whichever it's drawn over.
OTHER_PERSON_OUTLINE_COLOR = (20, 20, 20)
OTHER_PERSON_FILL_COLOR = (255, 255, 255)
OTHER_PERSON_OUTLINE_THICKNESS = 4
OTHER_PERSON_INNER_THICKNESS = 2

# Below this many pixels of a second person's box actually landing inside
# the crop (post-resize), outlining them isn't worth it - a sliver of
# someone at the very edge of the padded crop is more visual noise than a
# useful "someone else is here too" signal.
OTHER_PERSON_MIN_VISIBLE_PX = 4


def _outline_other_people(
    crop, frame_players: list[dict], primary_stable_id: int, offset: tuple[int, int], scale: float,
) -> None:
    """Draws a white/dark-outlined rectangle (see OTHER_PERSON_* above)
    around every OTHER detected person whose box overlaps this thumbnail
    crop, so a human naming/reviewing primary_stable_id can tell at a glance
    that a second person caught in the same frame isn't the one being
    identified. Mutates `crop` in place."""
    ox, oy = offset
    crop_h, crop_w = crop.shape[:2]

    for player in frame_players:
        if player["stable_id"] == primary_stable_id:
            continue

        x1, y1, x2, y2 = player["box"]
        lx1, ly1 = (x1 - ox) * scale, (y1 - oy) * scale
        lx2, ly2 = (x2 - ox) * scale, (y2 - oy) * scale

        cx1, cy1 = max(0.0, lx1), max(0.0, ly1)
        cx2, cy2 = min(float(crop_w), lx2), min(float(crop_h), ly2)
        if cx2 - cx1 < OTHER_PERSON_MIN_VISIBLE_PX or cy2 - cy1 < OTHER_PERSON_MIN_VISIBLE_PX:
            continue

        p1, p2 = (int(cx1), int(cy1)), (int(cx2), int(cy2))
        cv2.rectangle(crop, p1, p2, OTHER_PERSON_OUTLINE_COLOR, OTHER_PERSON_OUTLINE_THICKNESS)
        cv2.rectangle(crop, p1, p2, OTHER_PERSON_FILL_COLOR, OTHER_PERSON_INNER_THICKNESS)


def _extract_thumbnail(
    video_path: Path, frame_idx: int, box: list[float], stable_id: int, frame_players: list[dict],
) -> Optional[str]:
    crop, offset = _extract_crop(video_path, frame_idx, box)
    if crop is None:
        return None
    # frame[y1:y2, x1:x2] is a view into the decoded frame, not its own
    # array - draw on a copy rather than risk mutating/relying on that.
    crop = crop.copy()

    scale = THUMBNAIL_MAX_DIM / max(crop.shape[0], crop.shape[1])
    if scale < 1.0:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    else:
        scale = 1.0

    _outline_other_people(crop, frame_players, stable_id, offset, scale)

    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None

    return base64.b64encode(buffer).decode("ascii")


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
# land and count as "on the court" for recalibrate_players - generous, since
# a real dig/dive routinely lands a step or two past the sideline or
# baseline. A stable_id whose foot position NEVER falls within this margin
# across the whole video is almost certainly not a player at all (bench,
# staff, a spectator), not a player who just played close to the edge.
PLAYER_COURT_MARGIN_M = 3.0


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
            # in-bounds (see CourtDefinition.BallDetection.ballDetection.
            # pixel_to_court's own docstring on this same elevation bias).
            fx, fy = _pixel_to_court((x1 + x2) / 2.0, y2, matrix)
            if -PLAYER_COURT_MARGIN_M <= fx <= _COURT_LENGTH_M + PLAYER_COURT_MARGIN_M and \
                    -PLAYER_COURT_MARGIN_M <= fy <= _COURT_WIDTH_M + PLAYER_COURT_MARGIN_M:
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
    best = _best_crop_per_player(positions, _frame_size(video_path))
    names = load_names(output_path)
    ignored = load_ignored(output_path)
    # Every player detected in a given frame, for _outline_other_people -
    # lets a thumbnail mark up any second person who happened to be caught
    # in the same frame as the one actually being shown.
    players_by_frame = {entry["frame_idx"]: entry["players"] for entry in positions}

    # Extracting a thumbnail means opening the video and seeking to a
    # specific frame - the single slowest part of this endpoint, and one
    # that's completely wasted work on every repeat call (the Setup tab,
    # PlayerReview, and the Stats page - which does this for every
    # completed job at once) since the chosen frame/box for a given
    # stable_id never changes once tracking has finished. Cache it to disk
    # instead of re-seeking the video every time.
    cache = _load_thumbnail_cache(output_path) if with_thumbnails else {}
    cache_dirty = False

    players = []
    for stable_id in sorted(best):
        record = best[stable_id]
        thumbnail = None
        if with_thumbnails:
            cache_key = str(stable_id)
            cached = cache.get(cache_key)
            if (
                cached is not None
                and cached.get("frame_idx") == record["frame_idx"]
                and cached.get("v") == THUMBNAIL_CACHE_VERSION
            ):
                thumbnail = cached["thumbnail_base64"]
            else:
                frame_players = players_by_frame.get(record["frame_idx"], [])
                thumbnail = _extract_thumbnail(
                    video_path, record["frame_idx"], record["box"], stable_id, frame_players
                )
                cache[cache_key] = {
                    "thumbnail_base64": thumbnail,
                    "frame_idx": record["frame_idx"],
                    "v": THUMBNAIL_CACHE_VERSION,
                }
                cache_dirty = True

        players.append({
            "stable_id": stable_id,
            "name": names.get(str(stable_id)),
            "appearances": record["appearances"],
            "thumbnail_base64": thumbnail,
            "thumbnail_frame_idx": record["frame_idx"],
            "thumbnail_timestamp_s": record["timestamp_s"],
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
