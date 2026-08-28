import base64
import json
from pathlib import Path
from typing import Optional

import cv2

from . import config

PLAYER_POSITIONS_NAME = "player_positions.json"
STATS_NAME = "player_stats.json"
ACTIONS_NAME = "actions.json"
GROUPED_ACTIONS_NAME = "actions_grouped.json"
CANDIDATE_MATCHES_NAME = "player_candidate_matches.json"

THUMBNAIL_MAX_DIM = 220
JPEG_QUALITY = 85

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


def _extract_thumbnail(video_path: Path, frame_idx: int, box: list[float]) -> Optional[str]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = (x2 - x1) * THUMBNAIL_PADDING_FRACTION
        pad_y = (y2 - y1) * THUMBNAIL_PADDING_FRACTION
        x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        x2, y2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        scale = THUMBNAIL_MAX_DIM / max(crop.shape[0], crop.shape[1])
        if scale < 1.0:
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))

        ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None

        return base64.b64encode(buffer).decode("ascii")
    finally:
        cap.release()


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


def list_players(video_path: Path, output_path: Path, with_thumbnails: bool = True) -> list[dict]:
    positions = _load_player_positions(output_path)
    best = _best_crop_per_player(positions, _frame_size(video_path))
    names = load_names(output_path)
    ignored = load_ignored(output_path)

    players = []
    for stable_id in sorted(best):
        record = best[stable_id]
        thumbnail = None
        if with_thumbnails:
            thumbnail = _extract_thumbnail(video_path, record["frame_idx"], record["box"])

        players.append({
            "stable_id": stable_id,
            "name": names.get(str(stable_id)),
            "appearances": record["appearances"],
            "thumbnail_base64": thumbnail,
            "thumbnail_frame_idx": record["frame_idx"],
            "thumbnail_timestamp_s": record["timestamp_s"],
            "ignored": stable_id in ignored,
        })

    return players


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
