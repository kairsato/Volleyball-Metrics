"""
Sub-frame contact-point correction for actions.json's hits.

ActionDetection.actionDetection's own hit detection (_find_hits) is
timing-accurate only to whichever single raw detected frame first tripped a
velocity-change threshold, and its in/out speeds come from a noisy two-point
delta. This module refines each of those hits, where the data supports it,
using the ball's own flight physics plus the attributed player's tracked
position:

  1. Fit the incoming and outgoing ball flight (every real detection between
     this hit and its neighbouring hits/rally boundary, on each side) as a
     simple projectile arc - constant horizontal velocity, and a
     gravity-constrained parabola vertically wherever height data exists.
  2. Solve for the sub-frame time at which those two independently-fitted
     arcs actually come closest together - a much better estimate of the
     true contact instant than "whichever frame was flagged first".
  3. Validate that solved point against the attributed player's own tracked
     position (re-derived from their box's bottom-centre, not the box
     centre player_positions.json itself stores - see
     _player_bottom_centre_court). A physics fit with no plausible player
     anywhere near it is more likely a fitting artefact than a real contact.

Whenever any of the above can't be done confidently - too few real
detections on either side, the two arcs never actually come close together,
or no player is plausibly there - the hit is returned completely unchanged,
i.e. the existing (less accurate) method is the fallback of last resort.

Deliberately self-contained (small local copies of court-calibration loading
and the gravity-parabola fit, rather than importing BallDetection.
ballDetection or ActionDetection.actionDetection) - the former eagerly
imports torch/ultralytics at module load for no benefit here, and the latter
would create a circular import since it imports refine_hits from this
module. Same reasoning as actionDetection.py's own locally-duplicated _iou.
"""
import json
from pathlib import Path

import cv2
import numpy as np

COURT_FILE_NAME = "court.json"
GRAVITY_M_S2 = 9.81

# Minimum real (non-interpolated) ball detections required on EACH side of a
# hit before an arc fit is even attempted - fewer than this and a 2-parameter
# linear (or 2-parameter gravity) fit has no meaningful residual signal, it
# just passes exactly through whatever noise is there.
MIN_ARC_POINTS = 4

# How far past the last/first real detection on either side the contact-time
# search is allowed to look - the true contact is often just outside the
# cleanly-detected run (that's often exactly why detection got noisy/lost
# right around it - the ball nearing a hand/body).
SEARCH_MARGIN_S = 0.05

# Sub-frame resolution for the contact-time grid search - fine enough to be
# well below a frame at any realistic video fps, not a claim of true
# millisecond measurement accuracy.
GRID_STEP_S = 1.0 / 240.0

# How close the independently-fitted incoming/outgoing arcs must actually
# come to each other at the best candidate contact time to be trusted as one
# real, continuous flight rather than two segments that just happen to both
# exist near the same hit. Metres in court space; a fraction of the frame
# diagonal in pixel-only space (no calibration).
MAX_GAP_RESIDUAL_M = 1.0
MAX_GAP_RESIDUAL_PIXEL_FRACTION = 0.05

# How far a candidate refined contact point may be from any tracked player
# and still be considered plausible - mirrors
# actionDetection.MAX_PLAYER_ATTRIBUTION_DISTANCE_M/_FRACTION (duplicated,
# not imported, see module docstring).
MAX_PLAYER_DISTANCE_M = 2.5
MAX_PLAYER_DISTANCE_FRACTION = 0.15
PLAYER_SEARCH_FRAME_RADIUS = 5

# Mirrors BallDetection.ballDetection.MAX_PLAUSIBLE_REPORTED_SPEED_MS - a
# refined court-space speed above this is trusted less than the arc fit that
# produced it (most often a flat ground-plane homography extrapolating badly
# for a job with no solved camera pose, the same known issue
# pixel_to_court's own docstring describes - see BallDetection.ballDetection).
# Rejected outright rather than clamped, same "last resort" reasoning as
# every other gate in this module: an unbelievable answer isn't a corrected
# one, so the original hit is kept instead.
MAX_PLAUSIBLE_SPEED_MS = 45.0

# How far back/forward (seconds) a flight segment is ever allowed to reach
# for real ball points, even when no rally boundary or neighbouring hit
# constrains it sooner (e.g. game_status.json missing, or hits sparse) - a
# volleyball flight between two touches is never anywhere close to this
# long, so it's purely a safety cap against pulling in an unrelated flight.
MAX_SEGMENT_LOOKBACK_S = 3.0
MAX_SEGMENT_LOOKAHEAD_S = 3.0


def _load_homography(output_path):
    """Small local copy of BallDetection.ballDetection.load_homography."""
    court_file = Path(output_path) / COURT_FILE_NAME
    if not court_file.exists():
        return None
    try:
        with open(court_file) as f:
            data = json.load(f)
        return np.array(data["homography"], dtype=np.float64)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None


def _pixel_to_court(x, y, matrix):
    """Same flat ground-plane projection as
    BallDetection.ballDetection.pixel_to_court."""
    point = np.array([[[x, y]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(point, matrix)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def _player_bottom_centre_court(player, matrix):
    """player["court"] (as stored in player_positions.json) is projected
    from the tracked box's CENTRE, which - per this codebase's own
    documented caveat - lands well off-court for a standing player;
    re-derive it from the box's BOTTOM-centre instead (the same fix the API
    layer applies at request time - see results_router.get_player_trajectory
    's court_of()), since an accurate player position is exactly what this
    module's contact-point validation depends on."""
    box = player.get("box")
    if matrix is None or box is None:
        return None
    x1, y1, x2, y2 = box
    return _pixel_to_court((x1 + x2) / 2.0, y2, matrix)


def _corrected_frame_players(frame_players, matrix):
    return [{**p, "court": _player_bottom_centre_court(p, matrix)} for p in frame_players]


def _players_near_frame(players_by_frame, frame_idx):
    for offset in range(PLAYER_SEARCH_FRAME_RADIUS + 1):
        for idx in {frame_idx - offset, frame_idx + offset}:
            if idx in players_by_frame:
                return players_by_frame[idx]
    return None


def _nearest_player_ok(frame_players, point_pixel, point_court, diagonal):
    """Whether ANY tracked player is close enough to `point` to plausibly
    have made contact there - same distance logic/thresholds as
    actionDetection._nearest_player, used here purely as a validity gate
    (not to pick an id - the existing attribution call downstream
    re-derives the actual id once a refined point is accepted)."""
    for p in frame_players:
        if point_court is not None and p.get("court") is not None:
            distance = float(np.linalg.norm(np.array(p["court"], dtype=float) - np.array(point_court, dtype=float)))
            if distance <= MAX_PLAYER_DISTANCE_M:
                return True
        elif point_pixel is not None and p.get("pixel") is not None:
            distance = float(np.linalg.norm(np.array(p["pixel"], dtype=float) - np.array(point_pixel, dtype=float)))
            if distance / diagonal <= MAX_PLAYER_DISTANCE_FRACTION:
                return True
    return False


def _fit_linear(t, v):
    """Least-squares v(t) = v0 + rate*t."""
    design = np.vstack([np.ones_like(t), t]).T
    (v0, rate), *_ = np.linalg.lstsq(design, v, rcond=None)
    return float(v0), float(rate)


def _fit_gravity(t, z):
    """Least-squares z(t) = z0 + vz*t - 0.5*g*t^2, gravity fixed rather than
    a free third coefficient - same constrained fit as
    BallDetection.ballDetection._fit_gravity_parabola (small local copy, see
    module docstring)."""
    design = np.vstack([np.ones_like(t), t]).T
    (z0, vz), *_ = np.linalg.lstsq(design, z + 0.5 * GRAVITY_M_S2 * t ** 2, rcond=None)
    return float(z0), float(vz)


class _Arc:
    """A fitted projectile arc: position(t)/velocity(t), t in seconds
    relative to the hit's own raw detected frame. z_params is None (a flat
    2D arc) whenever there weren't enough height-tagged points on this side
    to fit a vertical parabola too."""

    def __init__(self, x0, vx, y0, vy, z_params):
        self.x0, self.vx, self.y0, self.vy = x0, vx, y0, vy
        self.z_params = z_params  # (z0, vz) or None

    def position(self, t):
        pos = [self.x0 + self.vx * t, self.y0 + self.vy * t]
        if self.z_params is not None:
            z0, vz = self.z_params
            pos.append(z0 + vz * t - 0.5 * GRAVITY_M_S2 * t ** 2)
        return np.array(pos)

    def velocity(self, t):
        vel = [self.vx, self.vy]
        if self.z_params is not None:
            _z0, vz = self.z_params
            vel.append(vz - GRAVITY_M_S2 * t)
        return np.array(vel)


def _fit_arc(points, hit_frame_idx, fps, space):
    """points: real ball_speed.json entries. space: "court" (metres, only
    possible when every point has one) or "pixel" (always possible - pixel
    is guaranteed present on every real point). Returns an _Arc, or None if
    there aren't enough usable points for this space."""
    if len(points) < MIN_ARC_POINTS:
        return None
    if space == "court" and any(p["court"] is None for p in points):
        return None

    t = np.array([(p["frame_idx"] - hit_frame_idx) / fps for p in points])
    xy = np.array([p[space] for p in points], dtype=float)

    x0, vx = _fit_linear(t, xy[:, 0])
    y0, vy = _fit_linear(t, xy[:, 1])

    z_params = None
    if space == "court":
        height_pts = [(tt, p["height_m"]) for tt, p in zip(t, points) if p.get("height_m") is not None]
        if len(height_pts) >= MIN_ARC_POINTS:
            tz = np.array([hp[0] for hp in height_pts])
            z = np.array([hp[1] for hp in height_pts])
            z_params = _fit_gravity(tz, z)

    return _Arc(x0, vx, y0, vy, z_params)


def _refine_one(hit, prev_bound, next_bound, real, players_by_frame, fps, matrix, diagonal):
    incoming = [p for p in real if prev_bound <= p["frame_idx"] <= hit["frame_idx"]]
    outgoing = [p for p in real if hit["frame_idx"] <= p["frame_idx"] <= next_bound]

    court_in = _fit_arc(incoming, hit["frame_idx"], fps, "court")
    court_out = _fit_arc(outgoing, hit["frame_idx"], fps, "court")
    pixel_in = _fit_arc(incoming, hit["frame_idx"], fps, "pixel")
    pixel_out = _fit_arc(outgoing, hit["frame_idx"], fps, "pixel")

    # Prefer solving in real court metres whenever both sides have it; fall
    # back to solving in pixel space (still useful for refining WHEN a
    # contact happened, just not a real-units speed) when they don't.
    if court_in is not None and court_out is not None:
        solve_in, solve_out = court_in, court_out
        space, max_gap = "court", MAX_GAP_RESIDUAL_M
    elif pixel_in is not None and pixel_out is not None:
        solve_in, solve_out = pixel_in, pixel_out
        space, max_gap = "pixel", MAX_GAP_RESIDUAL_PIXEL_FRACTION * diagonal
    else:
        return None

    t_last_in = (incoming[-1]["frame_idx"] - hit["frame_idx"]) / fps
    t_first_out = (outgoing[0]["frame_idx"] - hit["frame_idx"]) / fps
    t_lo = min(t_last_in, t_first_out) - SEARCH_MARGIN_S
    t_hi = max(t_last_in, t_first_out) + SEARCH_MARGIN_S
    if t_hi - t_lo < GRID_STEP_S:
        t_hi = t_lo + GRID_STEP_S
    grid = np.arange(t_lo, t_hi, GRID_STEP_S)

    gaps = [float(np.linalg.norm(solve_in.position(tt)[:2] - solve_out.position(tt)[:2])) for tt in grid]
    best_idx = int(np.argmin(gaps))
    best_t, best_gap = float(grid[best_idx]), gaps[best_idx]
    if best_gap > max_gap:
        return None

    contact_pixel = hit["pixel"]
    if pixel_in is not None and pixel_out is not None:
        avg = (pixel_in.position(best_t) + pixel_out.position(best_t)) / 2.0
        contact_pixel = (float(avg[0]), float(avg[1]))

    contact_court = None
    contact_height = None
    if space == "court":
        avg_xy = (court_in.position(best_t)[:2] + court_out.position(best_t)[:2]) / 2.0
        contact_court = (float(avg_xy[0]), float(avg_xy[1]))
        if court_in.z_params is not None and court_out.z_params is not None:
            z_in = court_in.position(best_t)[2]
            z_out = court_out.position(best_t)[2]
            contact_height = float((z_in + z_out) / 2.0)

    frame_players = _players_near_frame(players_by_frame, hit["frame_idx"]) or []
    corrected_players = _corrected_frame_players(frame_players, matrix)
    if not _nearest_player_ok(corrected_players, contact_pixel, contact_court, diagonal):
        return None

    speed_in = float(np.linalg.norm(solve_in.velocity(best_t)))
    speed_out = float(np.linalg.norm(solve_out.velocity(best_t)))
    real_units = space == "court"

    if real_units and (speed_in > MAX_PLAUSIBLE_SPEED_MS or speed_out > MAX_PLAUSIBLE_SPEED_MS):
        return None

    return {
        "pixel": contact_pixel,
        "court": contact_court,
        "height_m": contact_height,
        "timestamp_s": hit["frame_idx"] / fps + best_t,
        "speed_in_ms_or_pxs": speed_in,
        "speed_out_ms_or_pxs": speed_out,
        "speed_in_real_units": real_units,
        "speed_out_real_units": real_units,
    }


def _rally_bounds_for(rallies, frame_idx):
    for rally in rallies:
        if rally["start_frame"] <= frame_idx <= rally["end_frame"]:
            return rally["rally_index"], rally["start_frame"], rally["end_frame"]
    return None, None, None


def refine_hits(hits, ball_entries, rallies, players_by_frame, fps, output_path, diagonal):
    """Returns a new list, same shape/order/length as `hits`, with
    frame_idx left untouched (still needed for video-frame/rally/player-
    frame lookups elsewhere) but pixel/court/height_m/speed_in/speed_out
    updated in place - plus a new timestamp_s and contact_refined: bool -
    for whichever hits could be confidently refined. Every hit this module
    can't improve is returned byte-identical to the input, aside from
    contact_refined: False."""
    if not hits:
        return hits

    real = [e for e in ball_entries if not e["interpolated"] and e["pixel"] is not None]
    matrix = _load_homography(output_path)

    lookback_frames = MAX_SEGMENT_LOOKBACK_S * fps
    lookahead_frames = MAX_SEGMENT_LOOKAHEAD_S * fps

    refined = []
    for i, hit in enumerate(hits):
        rally_index, rally_start, rally_end = _rally_bounds_for(rallies, hit["frame_idx"])

        if i > 0 and _rally_bounds_for(rallies, hits[i - 1]["frame_idx"])[0] == rally_index:
            prev_bound = hits[i - 1]["frame_idx"]
        else:
            prev_bound = rally_start if rally_start is not None else hit["frame_idx"] - lookback_frames
        prev_bound = max(prev_bound, hit["frame_idx"] - lookback_frames)

        if i + 1 < len(hits) and _rally_bounds_for(rallies, hits[i + 1]["frame_idx"])[0] == rally_index:
            next_bound = hits[i + 1]["frame_idx"]
        else:
            next_bound = rally_end if rally_end is not None else hit["frame_idx"] + lookahead_frames
        next_bound = min(next_bound, hit["frame_idx"] + lookahead_frames)

        result = _refine_one(hit, prev_bound, next_bound, real, players_by_frame, fps, matrix, diagonal)

        new_hit = dict(hit)
        new_hit["contact_refined"] = result is not None
        if result is not None:
            new_hit.update(result)
        refined.append(new_hit)

    return refined
