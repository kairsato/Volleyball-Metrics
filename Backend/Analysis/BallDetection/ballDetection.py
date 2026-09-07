import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from CourtDefinition.court import COURT_LENGTH, COURT_WIDTH, camera_to_world, camera_pose_candidates

MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "MachineLearning" / "models" / "ballDetection_yolo26x_best.pt"
)
SECONDARY_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "MachineLearning" / "models" / "ballDetection_yolo11x_best.pt"
)
OUTPUT_VIDEO_NAME = "ball.mp4"
SPEED_LOG_NAME = "ball_speed.json"
TRAJECTORY_LOG_NAME = "ball_trajectory.json"
RAW_CANDIDATES_LOG_NAME = "ball_candidates.json"

BALL_CLASS_ID = 0
COLLECTION_CONF_THRESHOLD = 0.05
COURT_FILE_NAME = "court.json"
BALL_DIAMETER_M = 0.21
BALL_HEIGHT_MAX_PLAUSIBLE_M = 12.0
GRAVITY_M_S2 = 9.81
MIN_FLIGHT_SEGMENT_FRAMES = 8
REFINE_MAX_REPROJECTION_ERROR_RATIO = 2.0
MODEL_IMG_SIZE = 960
MIN_BALL_DIAGONAL_PX = 6.0
MAX_PLAUSIBLE_REPORTED_SPEED_MS = 45.0
SPEED_WINDOW_FRAMES = 3

STILL_CELL_PX = 6.0
STILL_WINDOW_FRAMES = 15
STILL_MIN_HITS = 10
AGREEMENT_RADIUS_FRAC = 0.03
MIN_SELECT_CONFIDENCE = 0.60
MIN_SIZE_SAMPLES = 30
SIZE_RANGE_STD_FACTOR = 1.5
MAX_INTERPOLATE_GAP_FRAMES = 20
INTERPOLATE_TANGENT_DAMPING = 0.6

TRAIL_LENGTH = 15
MINIMAP_WIDTH = 320
MINIMAP_HEIGHT = 160
MINIMAP_MARGIN_PX = 16
BALL_ROI_PADDING_FRACTION = 0.08


def load_homography(output_path):
    """
    Load the pixel -> real-world-court-metres transform CourtDefinition.court
    saved for this video. Returns None (speed then falls back to pixels/sec)
    if no calibration has been saved yet.
    """
    court_file = Path(output_path) / COURT_FILE_NAME

    if not court_file.exists():
        return None

    try:
        with open(court_file) as f:
            data = json.load(f)

        matrix = np.array(data["homography"], dtype=np.float64)

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(f"Could not read court calibration from {court_file} ({e}); "
              f"ball speed will be reported in pixels/sec instead of real units.")
        return None

    return matrix


def load_court_corners(output_path):
    """Load the four court corners CourtDefinition.court saved for this video,
    in frame pixel coordinates. Returns None if no calibration has been saved."""

    court_file = Path(output_path) / COURT_FILE_NAME

    if not court_file.exists():
        return None

    try:
        with open(court_file) as f:
            data = json.load(f)

        points = [(p["x"], p["y"]) for p in data["image_points"]]

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(f"Could not read court boundary from {court_file} ({e}); "
              f"ball detection will search the full frame instead.")
        return None

    if len(points) != 4:
        return None

    return np.array(points, dtype=np.float64)


def load_camera_pose(output_path):
    """
    Load the camera pose (intrinsics + extrinsics) CourtDefinition.court's
    estimate_camera_pose solved and calibration.py cached into court.json's
    "camera_pose" key - see estimate_ball_height below for what it's used
    for. Returns None if no calibration has been saved, the net-top points
    were never actually marked (see calibration.py's net_top_calibrated
    flag - an untouched preset guess never gets a camera_pose entry at all),
    or pose-solving itself failed at save time.
    """
    court_file = Path(output_path) / COURT_FILE_NAME

    if not court_file.exists():
        return None

    try:
        with open(court_file) as f:
            data = json.load(f)

        pose = data.get("camera_pose")
        if pose is None:
            return None

        focal_px = float(pose["focal_px"])
        cx, cy = pose["principal_point"]
        K = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        rvec = np.array(pose["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(pose["tvec"], dtype=np.float64).reshape(3, 1)

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(f"Could not read camera pose from {court_file} ({e}); "
              f"ball height will not be estimated.")
        return None

    return K, rvec, tvec


def load_calibration_points(output_path):
    """
    Load the 6 raw image points, net height, and frame dimensions
    CalibrationPanel saved into court.json - what camera_pose_candidates'
    focal-length search needs to re-run, as opposed to load_camera_pose's
    single already-solved answer. Used by refine_camera_pose_from_flight to
    re-rank that same search space against the ball's own flight physics.

    Frame width/height aren't stored directly; they're recovered from the
    cached camera_pose's principal_point, which estimate_camera_pose always
    sets to the image centre (see court.py's own comment on that
    simplification).

    Returns a tuple in exactly camera_pose_candidates' argument order, or
    None if no calibration/camera pose has been saved for this job.
    """
    court_file = Path(output_path) / COURT_FILE_NAME
    if not court_file.exists():
        return None

    try:
        with open(court_file) as f:
            data = json.load(f)

        points = data["points"]
        net_top_points = data["net_top_points"]
        net_height_m = data["net"]["height_m"]
        cx, cy = data["camera_pose"]["principal_point"]

        return (
            (points["middle_left"]["x"], points["middle_left"]["y"]),
            (points["middle_right"]["x"], points["middle_right"]["y"]),
            (points["far_left"]["x"], points["far_left"]["y"]),
            (points["far_right"]["x"], points["far_right"]["y"]),
            (net_top_points["net_top_left"]["x"], net_top_points["net_top_left"]["y"]),
            (net_top_points["net_top_right"]["x"], net_top_points["net_top_right"]["y"]),
            net_height_m,
            cx * 2.0,
            cy * 2.0,
        )

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(f"Could not read calibration points from {court_file} ({e}); "
              f"camera pose will not be refined against the ball's flight.")
        return None


def estimate_ball_world_position(box, camera_pose):
    """
    Estimates the ball's full 3D position in world (court) coordinates -
    metres, origin/axes per CourtDefinition.court's convention, Z = height
    above the ground plane - from a single detection box, using the camera
    pose CourtDefinition.court.estimate_camera_pose solved (see
    load_camera_pose above) plus the ball's known real-world diameter
    (BALL_DIAMETER_M):

      1. Apparent diameter in pixels - deliberately the SMALLER of the box's
         width/height, not an average: motion blur elongates a fast-moving
         ball along its direction of travel, so the smaller axis is the more
         robust true-diameter proxy of the two.
      2. Depth (distance along the camera's optical axis) from similar
         triangles: real diameter / apparent diameter = depth / focal length.
      3. Back-project the box centre through the inverse intrinsic matrix,
         scaled by that depth, to get the ball's position in camera-space
         3D coordinates.
      4. Rotate/translate that into world coordinates via the camera's
         extrinsics (court.camera_to_world) - the world frame's Z axis is
         exactly "height above the court plane" by construction (see
         estimate_camera_pose's world point layout), so no further
         conversion is needed.

    Returns None if the box is degenerate (zero-size) or the implied height
    is implausible (see BALL_HEIGHT_MAX_PLAUSIBLE_M) - a wrong reading here
    would otherwise silently poison anything built on top of it (serve/set
    trajectory-height scoring, flight-segment fitting), same reasoning as
    speed's own MAX_PLAUSIBLE_REPORTED_SPEED_MS clamp.
    """
    K, rvec, tvec = camera_pose

    x1, y1, x2, y2 = box
    apparent_diameter_px = min(x2 - x1, y2 - y1)
    if apparent_diameter_px <= 0:
        return None

    focal_px = K[0, 0]
    depth = focal_px * BALL_DIAMETER_M / apparent_diameter_px

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ray = np.linalg.inv(K) @ np.array([cx, cy, 1.0])
    point_camera = depth * ray

    point_world = camera_to_world(point_camera, rvec, tvec)

    if point_world[2] < -0.5 or point_world[2] > BALL_HEIGHT_MAX_PLAUSIBLE_M:
        return None

    return point_world


def estimate_ball_height(box, camera_pose):
    """Height (metres) above the court's ground plane - the Z component of
    estimate_ball_world_position, for callers that only need height. See
    that function for the full 3D position and how it's derived."""
    point_world = estimate_ball_world_position(box, camera_pose)
    return None if point_world is None else float(point_world[2])


def extract_flight_segments(trajectory):
    """Contiguous runs of consecutive "real" (actually-detected, never
    interpolated) trajectory frames, long enough (MIN_FLIGHT_SEGMENT_FRAMES)
    for a stable quadratic fit - see refine_camera_pose_from_flight.
    Interpolated frames are deliberately excluded: their box is a guessed
    position with no real apparent size behind it (see build_speed_log's own
    height-estimation comment), so it can't feed a physics fit without just
    circularly confirming whatever assumption produced the guess.

    Returns a list of (start_frame_idx, [box, box, ...]) - start_frame_idx
    plus each box's position in that list gives its timestamp
    (start_frame_idx + offset) / fps.
    """
    segments = []
    run_start = None
    run_boxes = []

    def flush():
        if len(run_boxes) >= MIN_FLIGHT_SEGMENT_FRAMES:
            segments.append((run_start, list(run_boxes)))

    for idx, entry in enumerate(trajectory):
        if entry["status"] == "real":
            if run_start is None:
                run_start = idx
            run_boxes.append(entry["box"])
        else:
            flush()
            run_start = None
            run_boxes = []
    flush()

    return segments


def _fit_gravity_parabola(t, z):
    """Least-squares (z0, vz) for z(t) = z0 + vz*t - 0.5*GRAVITY_M_S2*t^2 -
    i.e. the closest-possible real projectile arc through the given (t, z)
    points, given gravity is fixed and known rather than a free third
    coefficient (see refine_camera_pose_from_flight's docstring for why
    that constraint matters: letting acceleration float would just let
    per-frame depth noise back into the fit).

    Returns (z0, vz, predicted_z), predicted_z being the fitted curve
    evaluated at every t.
    """
    design = np.vstack([np.ones_like(t), t]).T
    (z0, vz), *_ = np.linalg.lstsq(design, z + 0.5 * GRAVITY_M_S2 * t ** 2, rcond=None)
    predicted_z = z0 + vz * t - 0.5 * GRAVITY_M_S2 * t ** 2
    return float(z0), float(vz), predicted_z


def _flight_physics_mean_sq_error(pose, segments, fps):
    """How well a candidate camera pose's implied ball heights, across every
    flight segment, fit a real projectile arc (see _fit_gravity_parabola).
    Lower is better. Each segment gets its own fit, since each flight starts
    with its own height/velocity, but every segment's squared residuals are
    pooled into one mean - see refine_camera_pose_from_flight.

    Returns None if fewer than 3 usable height readings are available across
    all segments combined (3 points is the minimum that meaningfully
    constrains a 2-parameter fit rather than just interpolating it exactly).
    """
    total_sq_error = 0.0
    n_points = 0

    for _start_idx, boxes in segments:
        t, z = [], []
        for offset, box in enumerate(boxes):
            height = estimate_ball_height(box, pose)
            if height is not None:
                t.append(offset / fps)
                z.append(height)
        if len(t) < 3:
            continue

        t = np.array(t)
        z = np.array(z)
        _z0, _vz, predicted_z = _fit_gravity_parabola(t, z)

        total_sq_error += float(np.sum((z - predicted_z) ** 2))
        n_points += len(t)

    if n_points == 0:
        return None

    return total_sq_error / n_points


def refine_camera_pose_from_flight(trajectory, camera_pose, output_path, fps):
    """
    Re-picks the camera's focal length (and the extrinsics solvePnP jointly
    resolves alongside it) using the ball's own flight physics instead of
    only reprojection error against the 6 marked calibration points.

    estimate_camera_pose's own docstring already flags the problem this
    solves: a single-view focal-length search has a well-known ambiguity
    where several (focal length, camera distance) pairs reproject the same
    static calibration points almost equally well, so reprojection error
    alone can't always tell them apart. The ball breaks that tie - it's the
    one thing in frame whose real-world motion is known exactly (constant
    downward acceleration between touches) regardless of calibration, so
    only the genuinely-correct focal length will make its camera-pose-
    derived height trace out a real parabola across a whole flight, instead
    of a warped one (see _flight_physics_mean_sq_error). Candidates whose
    reprojection error is far worse than the best one found are excluded
    first (REFINE_MAX_REPROJECTION_ERROR_RATIO) - physics can pick among
    near-tied candidates but should never override one that plainly doesn't
    fit the marked points at all.

    Returns camera_pose unchanged if there's nothing to refine against: no
    camera pose, no calibration points on file, or too few/short clean
    (unoccluded, non-interpolated) flight segments in this trajectory - see
    MIN_FLIGHT_SEGMENT_FRAMES. Callers can always just use whatever this
    returns.
    """
    if camera_pose is None:
        return None

    calibration_points = load_calibration_points(output_path)
    if calibration_points is None:
        return camera_pose

    segments = extract_flight_segments(trajectory)
    if not segments:
        return camera_pose

    candidates = camera_pose_candidates(*calibration_points)
    if not candidates:
        return camera_pose

    best_reprojection_error = min(c[3] for c in candidates)
    plausible = [c for c in candidates if c[3] <= REFINE_MAX_REPROJECTION_ERROR_RATIO * best_reprojection_error]

    best_pose, best_error = None, None
    for K, rvec, tvec, _reprojection_error in plausible:
        error = _flight_physics_mean_sq_error((K, rvec, tvec), segments, fps)
        if error is not None and (best_error is None or error < best_error):
            best_pose, best_error = (K, rvec, tvec), error

    if best_pose is None:
        return camera_pose

    return best_pose


def build_ball_roi(corners, frame_width, frame_height):
    """Axis-aligned region where ball detections are considered valid - see
    BALL_ROI_PADDING_FRACTION above. The top always reaches the top of the
    frame, since the ball's flight can carry it very high above the court."""

    diagonal = float(np.hypot(frame_width, frame_height))
    pad = BALL_ROI_PADDING_FRACTION * diagonal

    min_x = float(corners[:, 0].min()) - pad
    max_x = float(corners[:, 0].max()) + pad
    min_y = 0.0
    max_y = float(corners[:, 1].max()) + pad

    return min_x, min_y, max_x, max_y


def box_in_roi(box, roi):
    if roi is None:
        return True

    min_x, min_y, max_x, max_y = roi
    cx, cy = centre_of(box)

    return min_x <= cx <= max_x and min_y <= cy <= max_y


def pixel_to_court(x, y, matrix):
    """
    Project a pixel coordinate onto the calibrated court plane, in metres.
    Only accurate for points actually on that plane - an airborne ball is
    above it, so this systematically under-reports speed the higher off the
    court and further from the camera's sightline the ball is (a serve or
    spike's peak speed will read low). Good enough for a first pass; a more
    accurate reading would need to also estimate the ball's height, e.g. from
    its known real diameter (BALL_DIAMETER_M) versus its apparent size.
    """
    point = np.array([[[x, y]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(point, matrix)

    return float(transformed[0][0][0]), float(transformed[0][0][1])


def centre_of(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_diagonal(box):
    return float(np.hypot(max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])))


def unpack_ball_boxes(result):
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()


def modelDetection(video_path, models, roi, device):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_raw = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            candidates = []
            for model_name, model in models:
                result = model.predict(source=frame, classes=[BALL_CLASS_ID], conf=COLLECTION_CONF_THRESHOLD,
                                        imgsz=MODEL_IMG_SIZE, device=device, verbose=False)[0]
                boxes, confs = unpack_ball_boxes(result)
                for box, conf in zip(boxes, confs):
                    if box_diagonal(box) < MIN_BALL_DIAGONAL_PX:
                        continue
                    if not box_in_roi(box, roi):
                        continue
                    candidates.append({"box": [float(v) for v in box], "conf": float(conf), "model": model_name})

            frames_raw.append({"frame_idx": frame_idx, "candidates": candidates})
            frame_idx += 1
    finally:
        cap.release()

    return frames_raw, fps, frame_w, frame_h


def _cell_of(box):
    x1, y1, x2, y2 = box
    return round((x1 + x2) / 2.0 / STILL_CELL_PX), round((y1 + y2) / 2.0 / STILL_CELL_PX)


def _find_still_cells(frames_raw):
    cell_frames = {}
    for f in frames_raw:
        for cell in {_cell_of(c["box"]) for c in f["candidates"]}:
            cell_frames.setdefault(cell, []).append(f["frame_idx"])

    still = set()
    for cell, idxs in cell_frames.items():
        left = 0
        for right in range(len(idxs)):
            while idxs[right] - idxs[left] >= STILL_WINDOW_FRAMES:
                left += 1
            if right - left + 1 >= STILL_MIN_HITS:
                still.add(cell)
                break
    return still


def _combined_confidence(cands, centres, i, agreement_radius):
    agreeing = max(
        (cands[j]["conf"] for j in range(len(cands))
         if cands[j]["model"] != cands[i]["model"] and np.linalg.norm(centres[i] - centres[j]) <= agreement_radius),
        default=0.0,
    )
    return 1.0 - (1.0 - cands[i]["conf"]) * (1.0 - agreeing)


def _select_ball(filtered, confs_per_frame, size_range):
    chosen = []
    for cands, confs in zip(filtered, confs_per_frame):
        if not cands:
            chosen.append({"status": "out_of_frame", "box": None, "conf": None, "model": None})
            continue

        if size_range is None:
            best_i = max(range(len(cands)), key=lambda i: confs[i])
        else:
            low_w, high_w, low_h, high_h = size_range

            def deviation(i):
                x1, y1, x2, y2 = cands[i]["box"]
                return max(0.0, low_w - (x2 - x1), (x2 - x1) - high_w) + max(0.0, low_h - (y2 - y1), (y2 - y1) - high_h)

            best_i = min(range(len(cands)), key=lambda i: (deviation(i), -confs[i]))

        if confs[best_i] < MIN_SELECT_CONFIDENCE:
            chosen.append({"status": "out_of_frame", "box": None, "conf": None, "model": None})
            continue

        best = cands[best_i]
        chosen.append({"status": "real", "box": best["box"], "conf": confs[best_i], "model": best["model"]})
    return chosen


def _estimate_size_range(chosen):
    widths = [c["box"][2] - c["box"][0] for c in chosen if c["status"] == "real"]
    heights = [c["box"][3] - c["box"][1] for c in chosen if c["status"] == "real"]
    if len(widths) < MIN_SIZE_SAMPLES:
        return None

    widths, heights = np.array(widths), np.array(heights)
    w_mean, w_std = widths.mean(), widths.std()
    h_mean, h_std = heights.mean(), heights.std()
    low_w, high_w = max(0.0, w_mean - SIZE_RANGE_STD_FACTOR * w_std), w_mean + SIZE_RANGE_STD_FACTOR * w_std
    low_h, high_h = max(0.0, h_mean - SIZE_RANGE_STD_FACTOR * h_std), h_mean + SIZE_RANGE_STD_FACTOR * h_std
    print(f"Ball size profile: {w_mean:.1f}x{h_mean:.1f}px avg over {len(widths)} frames, "
          f"plausible range [{low_w:.1f}-{high_w:.1f}] x [{low_h:.1f}-{high_h:.1f}]px")
    return low_w, high_w, low_h, high_h


def _interpolate_gaps(chosen):
    n = len(chosen)
    zero_v = np.array([0.0, 0.0])
    i = 0
    while i < n:
        if chosen[i]["status"] != "out_of_frame":
            i += 1
            continue
        j = i
        while j < n and chosen[j]["status"] == "out_of_frame":
            j += 1
        gap_len = j - i
        left_ok = i > 0 and chosen[i - 1]["status"] == "real"
        right_ok = j < n and chosen[j]["status"] == "real"
        if gap_len <= MAX_INTERPOLATE_GAP_FRAMES and left_ok and right_ok:
            p0 = centre_of(chosen[i - 1]["box"])
            p1 = centre_of(chosen[j]["box"])
            box0, box1 = chosen[i - 1]["box"], chosen[j]["box"]

            v_in = zero_v
            if i - 2 >= 0 and chosen[i - 2]["box"] is not None:
                v_in = p0 - centre_of(chosen[i - 2]["box"])
            v_out = zero_v
            if j + 1 < n and chosen[j + 1]["box"] is not None:
                v_out = centre_of(chosen[j + 1]["box"]) - p1

            span = gap_len + 1
            m0 = v_in * span * INTERPOLATE_TANGENT_DAMPING
            m1 = v_out * span * INTERPOLATE_TANGENT_DAMPING

            for t, idx in enumerate(range(i, j), start=1):
                frac = t / span
                f2, f3 = frac * frac, frac * frac * frac
                h00 = 2 * f3 - 3 * f2 + 1
                h10 = f3 - 2 * f2 + frac
                h01 = -2 * f3 + 3 * f2
                h11 = f3 - f2
                pos = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

                w0, h0 = box0[2] - box0[0], box0[3] - box0[1]
                w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
                w = w0 + (w1 - w0) * frac
                h = h0 + (h1 - h0) * frac
                chosen[idx] = {
                    "status": "interpolated",
                    "box": [pos[0] - w / 2, pos[1] - h / 2, pos[0] + w / 2, pos[1] + h / 2],
                    "conf": None,
                    "model": None,
                }
        i = j
    return chosen


def postProcessing(frames_raw, frame_w, frame_h):
    diagonal = math.hypot(frame_w, frame_h)
    agreement_radius = AGREEMENT_RADIUS_FRAC * diagonal

    still_cells = _find_still_cells(frames_raw)
    filtered = [[c for c in f["candidates"] if _cell_of(c["box"]) not in still_cells] for f in frames_raw]
    centres_per_frame = [[centre_of(c["box"]) for c in cands] for cands in filtered]
    confs_per_frame = [
        [_combined_confidence(cands, centres, i, agreement_radius) for i in range(len(cands))]
        for cands, centres in zip(filtered, centres_per_frame)
    ]

    chosen = _select_ball(filtered, confs_per_frame, None)
    size_range = _estimate_size_range(chosen)
    if size_range is not None:
        chosen = _select_ball(filtered, confs_per_frame, size_range)

    return _interpolate_gaps(chosen)


# =====================================================================
# Speed/height/court derivation from a resolved trajectory
# =====================================================================

def build_speed_log(trajectory, fps, matrix, camera_pose):
    """Turns postProcessing's per-frame status/box list into
    SPEED_LOG_NAME's schema: one entry per frame, with real-world
    position/speed/height wherever calibration allows it."""

    log = []
    history = []  # recent (frame_idx, pixel, court) for real/interpolated frames only

    for entry in trajectory:
        frame_idx = len(log)
        status = entry["status"]

        if status == "out_of_frame":
            log.append({
                "frame_idx": frame_idx, "confidence": None, "interpolated": False,
                "pixel": None, "court": None, "height_m": None,
                "real_units": False, "speed_m_per_s": None, "speed_px_per_s": None,
            })
            continue

        pixel = centre_of(entry["box"])
        interpolated = status == "interpolated"
        confidence = entry["conf"] if not interpolated else None
        # Height is only ever derived from an actually-measured box, not an
        # interpolated one - there's no apparent size to measure a depth
        # from once it's just a guessed position.
        height = estimate_ball_height(entry["box"], camera_pose) if camera_pose is not None and not interpolated else None
        court = pixel_to_court(pixel[0], pixel[1], matrix) if matrix is not None else None

        point = {"frame_idx": frame_idx, "pixel": pixel, "court": court, "interpolated": interpolated}
        history.append(point)
        if len(history) > TRAIL_LENGTH:
            history.pop(0)

        speed_m_per_s = None
        speed_px_per_s = None
        real_units = False
        if len(history) >= 2:
            window_start = history[max(0, len(history) - 1 - SPEED_WINDOW_FRAMES)]
            dt = (point["frame_idx"] - window_start["frame_idx"]) / fps
            if dt > 0:
                # A real-world speed is only ever computed between two
                # actually-measured endpoints, never an interpolated one -
                # that position is a guess, not a detection, and the
                # ground-plane projection is already noise-sensitive enough
                # without compounding it through a guessed point.
                endpoints_measured = not point["interpolated"] and not window_start["interpolated"]
                court_speed = None
                if endpoints_measured and point["court"] is not None and window_start["court"] is not None:
                    court_speed = float(np.linalg.norm(np.array(point["court"]) - np.array(window_start["court"]))) / dt
                if court_speed is not None and court_speed <= MAX_PLAUSIBLE_REPORTED_SPEED_MS:
                    real_units = True
                    speed_m_per_s = court_speed
                else:
                    speed_px_per_s = float(np.linalg.norm(point["pixel"] - window_start["pixel"])) / dt

        log.append({
            "frame_idx": frame_idx,
            "confidence": confidence,
            "interpolated": interpolated,
            "pixel": pixel.tolist(),
            "court": list(court) if court is not None else None,
            "height_m": height,
            "real_units": real_units,
            "speed_m_per_s": speed_m_per_s,
            "speed_px_per_s": speed_px_per_s,
        })

    return log


# =====================================================================
# Annotation (debug/preview rendering only - see detectBall's save_video/
# show_preview args)
# =====================================================================

def draw_roi(frame, roi):
    """Outline the region ball detections are restricted to, so it's easy to
    visually confirm the court + padding + max-height area is where you'd
    expect (e.g. that it's actually excluding a sideline ball basket)."""

    if roi is None:
        return

    min_x, min_y, max_x, max_y = roi
    height, width = frame.shape[:2]

    p1 = (int(np.clip(min_x, 0, width - 1)), int(np.clip(min_y, 0, height - 1)))
    p2 = (int(np.clip(max_x, 0, width - 1)), int(np.clip(max_y, 0, height - 1)))

    cv2.rectangle(frame, p1, p2, (255, 0, 255), 2)
    cv2.putText(frame, "ball search area", (p1[0] + 6, p1[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)


def draw_court_minimap(frame, court_trail):
    """
    Small top-down diagram of the court in the corner of the frame, with the
    ball's recent real-world positions plotted on it - where the ball
    actually is on the court, independent of this camera's particular angle.
    Skipped entirely if there's no real-world position to plot yet (no
    homography, or nothing has landed on the court plane this run).
    """
    if not court_trail:
        return

    height, width = frame.shape[:2]
    x1 = width - MINIMAP_WIDTH - MINIMAP_MARGIN_PX
    y1 = MINIMAP_MARGIN_PX
    x2 = x1 + MINIMAP_WIDTH
    y2 = y1 + MINIMAP_HEIGHT

    if x1 < 0 or y2 > height:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def to_minimap(point):
        cx, cy = point
        px = x1 + int(np.clip(cx / COURT_LENGTH, 0.0, 1.0) * MINIMAP_WIDTH)
        py = y1 + int(np.clip(cy / COURT_WIDTH, 0.0, 1.0) * MINIMAP_HEIGHT)
        return px, py

    # Net line down the middle of the court's length, plus the attack lines
    # 3m either side of it - standard volleyball court markings, drawn purely
    # from the known court dimensions rather than anything detected.
    net_x = x1 + MINIMAP_WIDTH // 2
    cv2.line(frame, (net_x, y1), (net_x, y2), (200, 200, 200), 1)

    for offset_m in (-3.0, 3.0):
        attack_x = x1 + int(((COURT_LENGTH / 2 + offset_m) / COURT_LENGTH) * MINIMAP_WIDTH)
        cv2.line(frame, (attack_x, y1), (attack_x, y2), (110, 110, 110), 1)

    for i in range(1, len(court_trail)):
        fade = i / len(court_trail)
        colour = (0, int(140 + 100 * fade), int(255 * fade))
        cv2.line(frame, to_minimap(court_trail[i - 1]), to_minimap(court_trail[i]), colour, 2)

    cv2.circle(frame, to_minimap(court_trail[-1]), 4, (0, 255, 255), -1)

    cv2.putText(frame, "court map", (x1 + 6, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


STATUS_COLOR = {
    "real": (0, 255, 0),          # green - a real, chosen detection
    "interpolated": (255, 200, 0),  # cyan-blue - filled through a short gap
}

# BGR (cv2 convention), deliberately far from STATUS_COLOR's green/cyan-blue
# above so a raw candidate is never mistaken for the DP's own chosen box.
CANDIDATE_MODEL_COLOR = {
    "primary": (255, 0, 255),    # magenta - yolo26x
    "secondary": (0, 165, 255),  # orange - yolo11x
}

# Horizontal offset (px) for each model's confidence label - the single
# most useful frame to look at is exactly the one where both models agree
# closely (cross-model agreement is postProcessing's own strongest signal),
# which means their boxes usually sit
# almost exactly on top of each other. Without this, both labels land on
# the same few pixels and the second one drawn just overwrites the first,
# hiding whichever model happened to be drawn earlier - defeating the
# entire point of this debug view for the one case it matters most.
CANDIDATE_LABEL_DX = {"primary": 0, "secondary": 34}


def annotate_ball_candidates(frame, candidates):
    """
    Debug aid for postProcessing's own selection: draws EVERY raw
    modelDetection candidate this frame - every box either model produced,
    before postProcessing ever filters or picks a winner - each in its own
    model-coded colour (CANDIDATE_MODEL_COLOR) labelled with its confidence
    as a percentage. Drawn thin/underneath annotate_ball's own chosen-
    trajectory box (see _render_annotated_video's draw order) so a wrong
    pick is visible against everything that was actually available to pick
    from that frame, not just what got picked.
    """
    for c in candidates:
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        colour = CANDIDATE_MODEL_COLOR.get(c["model"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)

        label = f"{c['conf'] * 100:.0f}%"
        label_x = x1 + CANDIDATE_LABEL_DX.get(c["model"], 0)
        label_y = y1 - 4 if y1 - 4 > 10 else y2 + 14  # flip below the box near the top edge, so it stays on-screen
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)


def draw_candidate_legend(frame):
    """Small always-on key (top-left) for annotate_ball_candidates/
    annotate_ball's colour coding - without it, "which model is magenta"
    is only answered by reading this module's source."""
    rows = [
        ("primary (raw)", CANDIDATE_MODEL_COLOR["primary"]),
        ("secondary (raw)", CANDIDATE_MODEL_COLOR["secondary"]),
        ("chosen: real", STATUS_COLOR["real"]),
        ("chosen: interpolated", STATUS_COLOR["interpolated"]),
    ]
    x, y = 10, 20
    for label, colour in rows:
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        y += 18


def annotate_ball(frame, trail, status, box):
    """Draws the resolved trajectory's recent trail plus this frame's box,
    colour-coded by status (see STATUS_COLOR) - deliberately just the box,
    no speed label: with annotate_ball_candidates now also drawing every
    raw candidate's own percentage label right next to this same box (see
    _render_annotated_video's draw order), a further speed label here was
    just more text competing for the same small area. A plain box is enough
    to tell where the chosen trajectory landed against everything else
    drawn nearby; the actual speed is already in ball_speed.json for
    whenever it's the thing being checked."""

    for i in range(1, len(trail)):
        fade = i / len(trail)
        colour = (0, int(140 + 100 * fade), int(255 * fade))
        cv2.line(frame, tuple(int(v) for v in trail[i - 1]), tuple(int(v) for v in trail[i]), colour, 2)

    if status == "out_of_frame" or box is None:
        return

    x1, y1, x2, y2 = (int(v) for v in box)
    colour = STATUS_COLOR.get(status, (0, 200, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)


def _render_annotated_video(video_path, output_path, trajectory, speed_log, frames_raw, fps, frame_w, frame_h, show_preview):
    """Second pass over the source video, drawing the already-resolved
    trajectory PLUS every raw Phase 1 candidate from both models
    (annotate_ball_candidates) - only run when save_video/show_preview is
    requested (see detectBall), since the live pipeline never needs a
    rendered ball.mp4 of its own (PostProcessed/renderVideo.py renders the
    real, end-user-facing video from every stage's logs together, without
    any of this raw-candidate debug clutter)."""

    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(Path(output_path) / OUTPUT_VIDEO_NAME),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))

    # Every frame's annotations come entirely from already-resolved data
    # (trajectory/speed_log/frames_raw are fully computed by the time this
    # runs - no model inference happens in this pass, see this function's
    # own docstring), so there's nothing to gain from watching it live as it
    # writes - unlike a real-time preview, replaying the exact same frames
    # afterward loses nothing. Writing straight through first, then handing
    # off to _interactive_preview on the finished file, is what actually
    # lets that second pass be scrubbable (seek forward/back, pause) instead
    # of a one-shot playthrough with no way back to a frame you just missed.
    trail = []
    court_trail = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            entry = trajectory[frame_idx] if frame_idx < len(trajectory) else {"status": "out_of_frame", "box": None}
            reading = speed_log[frame_idx] if frame_idx < len(speed_log) else None

            if entry["status"] != "out_of_frame":
                trail.append(centre_of(entry["box"]))
                if len(trail) > TRAIL_LENGTH:
                    trail.pop(0)
                if reading is not None and reading["court"] is not None:
                    court_trail.append(reading["court"])
                    if len(court_trail) > TRAIL_LENGTH:
                        court_trail.pop(0)

            if frame_idx < len(frames_raw):
                annotate_ball_candidates(frame, frames_raw[frame_idx]["candidates"])
            annotate_ball(frame, trail, entry["status"], entry["box"])
            draw_court_minimap(frame, court_trail)
            draw_candidate_legend(frame)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    rendered_path = Path(output_path) / OUTPUT_VIDEO_NAME
    print(f"Ball detection video saved: {rendered_path}")

    if show_preview:
        _interactive_preview(rendered_path)


# Windows-specific extended key codes cv2.waitKeyEx returns for the arrow
# keys (plain cv2.waitKey only ever returns the low byte, which mangles
# these - see _interactive_preview). 'a'/'d' work as a portable fallback
# either way, so this only ever adds convenience, never a hard requirement.
_ARROW_LEFT, _ARROW_RIGHT = 2424832, 2555904


def _interactive_preview(video_path):
    """
    Scrub the just-rendered debug video with a trackbar plus keyboard
    controls, rather than only being able to watch it play through once at
    whatever speed decoding keeps up with - re-looking at a specific frame
    range used to mean re-running detection from scratch. Controls:

      Space           pause/resume
      A / Left arrow  step back one frame (pauses)
      D / Right arrow step forward one frame (pauses)
      Trackbar drag   jump to that frame (pauses)
      Esc             close

    Reads back the file _render_annotated_video just wrote rather than the
    live frames it was drawing - every frame's already fully annotated
    there, so this is free to seek arbitrarily with plain
    cv2.CAP_PROP_POS_FRAMES instead of needing to re-run any of the drawing
    above in a random-access-friendly way.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    window_name = "Ball Detection (Space: pause | A/D or arrows: step | Esc: quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Callback intentionally does nothing - the main loop below polls the
    # trackbar's position itself each frame (see "user dragged it" below)
    # rather than reacting from here, so a manual drag and this same loop's
    # own cv2.setTrackbarPos calls (to keep the bar in sync while playing)
    # are handled by one single code path instead of two.
    cv2.createTrackbar("Frame", window_name, 0, total_frames - 1, lambda _pos: None)

    frame_idx = 0
    paused = False
    try:
        while True:
            dragged_to = cv2.getTrackbarPos("Frame", window_name)
            if dragged_to != frame_idx:
                frame_idx = dragged_to
                paused = True

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(window_name, frame)

            key = cv2.waitKeyEx(0 if paused else 30)
            if key == 27:  # Esc
                break
            elif key == 32:  # Space
                paused = not paused
            elif key in (ord("a"), ord("A"), _ARROW_LEFT):
                paused = True
                frame_idx = max(0, frame_idx - 1)
            elif key in (ord("d"), ord("D"), _ARROW_RIGHT):
                paused = True
                frame_idx = min(total_frames - 1, frame_idx + 1)
            elif not paused:
                frame_idx += 1
                if frame_idx >= total_frames:
                    frame_idx = total_frames - 1
                    paused = True  # stop at the end instead of closing outright

            cv2.setTrackbarPos("Frame", window_name, frame_idx)
    finally:
        cap.release()
        cv2.destroyAllWindows()


# =====================================================================
# Entry points
# =====================================================================

def reselect_ball(output_path):
    """
    Re-runs postProcessing over RAW_CANDIDATES_LOG_NAME (modelDetection's
    cached per-frame candidates from both models) against whatever
    court.json says *now*, without re-running detection - rewrites
    SPEED_LOG_NAME and TRAJECTORY_LOG_NAME to match.

    One thing a from-scratch reprocess would catch that this can't:
    modelDetection already applied whatever court ROI was active *at
    collection time* - a differently-calibrated court that would have let
    through a candidate the old ROI excluded can't recover it here. A job
    whose calibration turns out to need that needs a real reprocess instead
    (see jobs_router.redo_job).

    Raises ValueError if this job's ball detection ran before the current
    two-model RAW_CANDIDATES_LOG_NAME format existed - there's nothing
    compatible to re-pick from.
    """
    candidates_file = Path(output_path) / RAW_CANDIDATES_LOG_NAME
    if not candidates_file.exists():
        raise ValueError(
            f"No raw ball candidates saved for this job ({candidates_file} doesn't exist) - "
            f"a full reprocess is required to recalibrate its ball tracking."
        )

    with open(candidates_file) as f:
        data = json.load(f)

    if "frames" not in data or "fps" not in data or "frame_w" not in data:
        raise ValueError(
            f"{candidates_file} is in an old, incompatible format (from before the two-phase "
            f"ball detector) - a full reprocess is required to recalibrate its ball tracking."
        )

    fps = data["fps"]
    frame_w = data["frame_w"]
    frame_h = data["frame_h"]
    frames_raw = data["frames"]

    matrix = load_homography(output_path)
    camera_pose = load_camera_pose(output_path)

    trajectory = postProcessing(frames_raw, frame_w, frame_h)
    camera_pose = refine_camera_pose_from_flight(trajectory, camera_pose, output_path, fps)
    speed_log = build_speed_log(trajectory, fps, matrix, camera_pose)

    with open(Path(output_path) / SPEED_LOG_NAME, "w") as f:
        json.dump(speed_log, f)
    with open(Path(output_path) / TRAJECTORY_LOG_NAME, "w") as f:
        json.dump({"fps": fps, "frame_w": frame_w, "frame_h": frame_h, "frames": trajectory}, f)


def detectBall(video_path, output_path, show_preview=False, save_video=False):
    """
    show_preview/save_video (both default False) render an annotated
    ball.mp4 from the already-resolved trajectory (a second, cheap pass over
    the video - no model inference) - not needed day to day, since the
    JSON logs (ball_speed.json) are the real output once
    PostProcessed/renderVideo.py renders a combined annotated video from
    every stage's logs together.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Both checkpoints are named best.pt (each is just wherever
    # model_benchmark.py's training run left it) - the parent directory
    # (yolo26x/yolo11x) is what actually distinguishes them.
    print(f"Loading models: {Path(MODEL_PATH).parent.parent.name}, {Path(SECONDARY_MODEL_PATH).parent.parent.name}...")
    models = [("primary", YOLO(MODEL_PATH)), ("secondary", YOLO(SECONDARY_MODEL_PATH))]
    device = 0 if torch.cuda.is_available() else "cpu"

    matrix = load_homography(output_path)
    if matrix is not None:
        print(f"Loaded court calibration from {Path(output_path) / COURT_FILE_NAME}; "
              f"ball speed will be reported in real-world units (km/h).")
    else:
        print("No court calibration found; ball speed will be reported in pixels/sec instead.")

    camera_pose = load_camera_pose(output_path)
    if camera_pose is not None:
        print("Loaded camera pose; ball height will be estimated.")
    else:
        print("No camera pose available (net-top calibration points not marked); "
              "ball height will not be estimated.")

    corners = load_court_corners(output_path)
    roi = None
    if corners is not None:
        cap = cv2.VideoCapture(str(video_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        roi = build_ball_roi(corners, frame_w, frame_h)
        print(f"Restricting ball detection to the court area (+{BALL_ROI_PADDING_FRACTION:.0%} padding, "
              f"full frame height above it).")
    else:
        print("No court boundary found; ball detection will search the full frame.")

    t_start = time.perf_counter()
    print("Detecting ball candidates across the whole video...")
    frames_raw, fps, frame_w, frame_h = modelDetection(video_path, models, roi, device)
    print(f"Model detection done in {time.perf_counter() - t_start:.1f}s "
          f"({sum(len(f['candidates']) for f in frames_raw)} candidates over {len(frames_raw)} frames).")

    t_post = time.perf_counter()
    trajectory = postProcessing(frames_raw, frame_w, frame_h)
    real = sum(1 for e in trajectory if e["status"] == "real")
    interpolated = sum(1 for e in trajectory if e["status"] == "interpolated")
    out_of_frame = sum(1 for e in trajectory if e["status"] == "out_of_frame")
    print(f"Post-processing done in {time.perf_counter() - t_post:.1f}s. "
          f"real: {real} | interpolated: {interpolated} | out_of_frame: {out_of_frame}")

    camera_pose = refine_camera_pose_from_flight(trajectory, camera_pose, output_path, fps)
    speed_log = build_speed_log(trajectory, fps, matrix, camera_pose)

    with open(Path(output_path) / SPEED_LOG_NAME, "w") as f:
        json.dump(speed_log, f)
    with open(Path(output_path) / TRAJECTORY_LOG_NAME, "w") as f:
        json.dump({"fps": fps, "frame_w": frame_w, "frame_h": frame_h, "frames": trajectory}, f)
    with open(Path(output_path) / RAW_CANDIDATES_LOG_NAME, "w") as f:
        json.dump({"fps": fps, "frame_w": frame_w, "frame_h": frame_h, "frames": frames_raw}, f)

    print(f"Ball speed log saved: {Path(output_path) / SPEED_LOG_NAME}")
    print(f"Raw ball candidates saved: {Path(output_path) / RAW_CANDIDATES_LOG_NAME}")
    print(f"Total time: {time.perf_counter() - t_start:.1f}s")

    if save_video or show_preview:
        _render_annotated_video(video_path, output_path, trajectory, speed_log, frames_raw, fps, frame_w, frame_h, show_preview)
