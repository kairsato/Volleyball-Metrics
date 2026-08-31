import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# The stage modules live under Backend/Analysis/, a sibling of this API/ package.
BACKEND_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from CourtDefinition.court import (  # noqa: E402
    CALIBRATION_FRAME,
    COURT_LENGTH,
    COURT_WIDTH,
    create_half_court_homography,
    estimate_camera_pose,
    predict_court_geometry,
)

from . import config

JPEG_QUALITY = 90

# Standard men's/mixed vs. women's net heights - the only two choices the
# frontend's dropdown offers, so this is deliberately a closed set rather
# than an arbitrary float. Consumed by estimate_camera_pose below (the net
# top's known height is what makes camera-pose/ball-height estimation
# possible at all) as well as saved as reference metadata.
NET_HEIGHT_OPTIONS = {
    "mens": 2.43,
    "womens": 2.24,
}
DEFAULT_NET_HEIGHT_M = NET_HEIGHT_OPTIONS["mens"]

# The 4 points the web calibration UI actually asks for, in the fixed order
# create_half_court_homography expects - see court.py's module diagram.
POINT_NAMES = ("middle_left", "middle_right", "far_left", "far_right")

# How far (pixels) net_top_left/net_top_right have to move from their preset
# guess before they're trusted as a real calibration rather than an
# untouched default - same idea (and same tolerance) as the desktop tool's
# own NET_TOUCHED_TOLERANCE_PX. An untouched preset never gets a camera pose
# solved from it - see save() below.
NET_TOP_TOUCHED_TOLERANCE_PX = 3

# How far above middle_left/middle_right (pixels) the net-top preset starts,
# before the user drags it onto the actual net/antenna top - matches
# CourtDefinition.court's own NET_PRESET_HEIGHT_PX so the desktop and web
# tools' presets look the same.
NET_TOP_PRESET_OFFSET_PX = 120


def frame_size(video_path: Path) -> tuple[int, int]:
    """Cheap (width, height) read - container metadata only, no frame
    decode - unlike read_calibration_frame below, which actually reads and
    JPEG-encodes a frame. Used wherever only the dimensions matter (building
    a preset, solving a camera pose)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def read_calibration_frame(video_path: Path, timestamp_s: Optional[float] = None) -> tuple[bytes, int, int]:
    """Grabs a single frame JPEG-encoded for a web UI to draw on - frame 0 by
    default (the desktop calibration tool's fixed frame), or a specific
    timestamp when given, so callers like the score OCR region picker can
    let the user scrub to a moment where whatever they're marking is
    actually visible."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Could not open video")

        if timestamp_s is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, CALIBRATION_FRAME)
        success, frame = cap.read()
        if not success:
            raise ValueError("Could not read the calibration frame")

        height, width = frame.shape[:2]
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError("Could not encode the calibration frame")

        return bytes(buffer), width, height
    finally:
        cap.release()


# How many frames to sample across the video for build_clean_court_frame,
# and how far from the very start/end to stay (a few seconds of black frame,
# intro title card, or a warm-up huddle right at the edges is common, and
# skews the sample if it's a big fraction of it).
BACKGROUND_SAMPLE_COUNT = 20
BACKGROUND_SAMPLE_MARGIN_S = 2.0


def build_clean_court_frame(video_path: Path, output_path: Path) -> tuple[bytes, int, int]:
    """Best-effort "clean plate" of the court with people erased, for the
    court calibration picker specifically - a raw single frame is very
    likely to have players standing on exactly the spots (net line, court
    corners) someone needs an unobstructed view of to calibrate accurately.

    Works by sampling BACKGROUND_SAMPLE_COUNT frames spread evenly across
    the video and taking the per-pixel median across them: a person is only
    in any given pixel for a fraction of those samples (assuming they move
    around over the course of the video rather than standing in one exact
    spot from start to finish), so the median usually recovers the static
    court underneath them. Not perfect - a long huddle/timeout near the
    camera can still show through - but a solid starting point, and no
    worse than an arbitrary single frame that happens to have someone stood
    right on a line.

    Cached to output_path (see config.COURT_BACKGROUND_FILE_NAME) since
    it's expensive relative to a single frame read, and the source video
    never changes after upload, so there's nothing to invalidate it with.
    """
    cached = output_path / config.COURT_BACKGROUND_FILE_NAME
    if cached.exists():
        jpeg_bytes = cached.read_bytes()
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            height, width = image.shape[:2]
            return jpeg_bytes, width, height
        # Fall through and recompute if the cached file is somehow corrupt.

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            raise ValueError("Could not read video length")

        duration_s = frame_count / fps
        margin_s = min(BACKGROUND_SAMPLE_MARGIN_S, duration_s / 4)
        start_s = margin_s
        end_s = max(start_s, duration_s - margin_s)

        samples = []
        for i in range(BACKGROUND_SAMPLE_COUNT):
            t = start_s if BACKGROUND_SAMPLE_COUNT == 1 else start_s + (end_s - start_s) * i / (BACKGROUND_SAMPLE_COUNT - 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            success, frame = cap.read()
            if success:
                samples.append(frame)

        if not samples:
            raise ValueError("Could not read any frames from the video")

        background = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
        height, width = background.shape[:2]

        ok, buffer = cv2.imencode(".jpg", background, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError("Could not encode the background frame")

        jpeg_bytes = bytes(buffer)
        output_path.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(jpeg_bytes)
        return jpeg_bytes, width, height
    finally:
        cap.release()


def video_duration_s(video_path: Path) -> Optional[float]:
    """The video's own length in seconds, from its container metadata - no
    frame is actually decoded, so this is cheap enough to call once per
    upload (and, as a backfill, once per already-uploaded job that predates
    Job.duration_s existing)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            return None
        return frame_count / fps
    finally:
        cap.release()


def default_points(frame_width: int, frame_height: int) -> dict:
    """Rough starting guess for the 4 ground points plus the 2 net-top
    points, shaped like a typical sideline view: the far baseline (smaller,
    further from the camera) sits higher up and narrower than the net line
    (closer, wider) - purely a starting position to drag from, not a real
    estimate of this particular shot's perspective. The 4 ground points are
    chosen (via a grid search against predict_court_geometry) so the
    predicted near baseline and both attack lines land within the visible
    frame for this starting shape - a preset with a much steeper far/close
    ratio pushes the predicted near baseline far outside the frame before
    the user has even touched a point. The 2 net-top points start directly
    above middle_left/middle_right (NET_TOP_PRESET_OFFSET_PX higher) -
    dragging them onto the actual net/antenna top is what net_top_calibrated
    (see save() below) checks for."""
    far_top = int(frame_height * 0.35)
    far_left_x = int(frame_width * 0.29)
    far_right_x = int(frame_width * 0.71)

    mid_top = int(frame_height * 0.50)
    mid_left_x = int(frame_width * 0.26)
    mid_right_x = int(frame_width * 0.74)

    net_top_y = max(0, mid_top - NET_TOP_PRESET_OFFSET_PX)

    return {
        "middle_left": {"x": mid_left_x, "y": mid_top},
        "middle_right": {"x": mid_right_x, "y": mid_top},
        "far_left": {"x": far_left_x, "y": far_top},
        "far_right": {"x": far_right_x, "y": far_top},
        "net_top_left": {"x": mid_left_x, "y": net_top_y},
        "net_top_right": {"x": mid_right_x, "y": net_top_y},
        "net_height_m": DEFAULT_NET_HEIGHT_M,
    }


def existing_points(output_path: Path, frame_width: int, frame_height: int) -> Optional[dict]:
    court_file = output_path / config.COURT_FILE_NAME
    if not court_file.exists():
        return None

    data = json.loads(court_file.read_text())
    points = data.get("points")
    if points and all(name in points for name in POINT_NAMES):
        net_top = data.get("net_top_points")
        if net_top and "net_top_left" in net_top and "net_top_right" in net_top:
            net_top_left, net_top_right = net_top["net_top_left"], net_top["net_top_right"]
        else:
            # Predates net-top points existing at all - preset guess, same
            # as a never-calibrated job, rather than crashing on a missing
            # field. net_top_calibrated below already defaults to False for
            # this case, so nothing downstream mistakes it for a real mark.
            preset = default_points(frame_width, frame_height)
            net_top_left, net_top_right = preset["net_top_left"], preset["net_top_right"]

        camera_pose = data.get("camera_pose")
        return {
            **{name: points[name] for name in POINT_NAMES},
            "net_top_left": net_top_left,
            "net_top_right": net_top_right,
            "net_height_m": data.get("net", {}).get("height_m", DEFAULT_NET_HEIGHT_M),
            "predicted": data.get("predicted"),
            # True for anything saved before this field existed - an
            # already-calibrated job shouldn't suddenly show up unlocked.
            "confirmed": data.get("confirmed", True),
            "net_top_calibrated": data.get("net_top_calibrated", False),
            "camera_pose_available": camera_pose is not None,
            "camera_pose_reprojection_error_px": (
                camera_pose.get("reprojection_error_px") if camera_pose else None
            ),
        }

    # Backward compatibility: a court.json saved before this 4-point
    # redesign only has the old 6-point set (4 full corners + 2 net-top
    # points - see CourtDefinition.court's desktop tool, still the format
    # `image_points` is written in). Approximate the new 4 points from
    # those corners so an already-calibrated older job still reads as
    # calibrated (with a reasonable, if approximate, summary) instead of
    # falsely claiming it was never calibrated at all. This never re-saves
    # anything - the job's already-tracked data used the old homography,
    # which stays untouched here; only re-calibrating through the web UI
    # converts it to the new format. The desktop tool's own net-top points
    # aren't reused here even though they exist in this legacy format -
    # they were marked for antenna-height display only, never solved into a
    # camera pose, so treating them as "net_top_calibrated" would overstate
    # confidence in a height estimate that was never actually computed.
    legacy_corners = data.get("image_points")
    if legacy_corners and len(legacy_corners) == 4:
        top_left, top_right, bottom_right, bottom_left = legacy_corners
        preset = default_points(frame_width, frame_height)
        return {
            "far_left": top_left,
            "far_right": top_right,
            "middle_left": bottom_left,
            "middle_right": bottom_right,
            "net_top_left": preset["net_top_left"],
            "net_top_right": preset["net_top_right"],
            "net_height_m": data.get("net", {}).get("height_m", DEFAULT_NET_HEIGHT_M),
            "predicted": None,
            # Same as the main branch above - defaults True for anything
            # predating this field, but still respects an explicit redo
            # (set_confirmed writes this key regardless of which format the
            # rest of the file is in).
            "confirmed": data.get("confirmed", True),
            "net_top_calibrated": False,
            "camera_pose_available": False,
            "camera_pose_reprojection_error_px": None,
        }

    return None


def set_confirmed(output_path: Path, confirmed: bool) -> None:
    """Flips court.json's `confirmed` flag without touching the actual
    points - this is what "Redo Court Identification" persists server-side
    (setUnlocked in CalibrationPanel.tsx was purely local React state and
    reset itself on every reload, which read as "redo doesn't stick").
    Saving new points (see save() below) always marks it confirmed again,
    same one-action save+lock as the initial "Set Court Identification"."""
    court_file = output_path / config.COURT_FILE_NAME
    if not court_file.exists():
        raise ValueError("No calibration saved yet")
    data = json.loads(court_file.read_text())
    data["confirmed"] = confirmed
    court_file.write_text(json.dumps(data, indent=2))


def save(
    output_path: Path,
    middle_left: tuple[float, float],
    middle_right: tuple[float, float],
    far_left: tuple[float, float],
    far_right: tuple[float, float],
    net_top_left: tuple[float, float],
    net_top_right: tuple[float, float],
    net_height_m: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    if net_height_m not in NET_HEIGHT_OPTIONS.values():
        raise ValueError(f"net_height_m must be one of {sorted(NET_HEIGHT_OPTIONS.values())}")

    matrix = create_half_court_homography(middle_left, middle_right, far_left, far_right)
    predicted = predict_court_geometry(matrix)

    # Only trust net_top_left/net_top_right - and therefore only attempt a
    # camera pose - once they've actually been dragged onto the net/antenna;
    # solving a "pose" from an untouched preset guess would silently produce
    # a confident-looking but meaningless height estimate.
    preset = default_points(frame_width, frame_height)
    net_top_calibrated = bool(
        np.hypot(net_top_left[0] - preset["net_top_left"]["x"], net_top_left[1] - preset["net_top_left"]["y"])
        > NET_TOP_TOUCHED_TOLERANCE_PX
        or np.hypot(
            net_top_right[0] - preset["net_top_right"]["x"], net_top_right[1] - preset["net_top_right"]["y"]
        )
        > NET_TOP_TOUCHED_TOLERANCE_PX
    )

    camera_pose_data = None
    if net_top_calibrated:
        pose = estimate_camera_pose(
            middle_left, middle_right, far_left, far_right, net_top_left, net_top_right,
            net_height_m, frame_width, frame_height,
        )
        if pose is not None:
            K, rvec, tvec, error_px = pose
            camera_pose_data = {
                "focal_px": float(K[0, 0]),
                "principal_point": [float(K[0, 2]), float(K[1, 2])],
                "rvec": np.asarray(rvec).flatten().tolist(),
                "tvec": np.asarray(tvec).flatten().tolist(),
                "reprojection_error_px": error_px,
            }

    output_path.mkdir(parents=True, exist_ok=True)
    data = {
        "court": {"length_m": COURT_LENGTH, "width_m": COURT_WIDTH},
        "points": {
            "middle_left": {"x": middle_left[0], "y": middle_left[1]},
            "middle_right": {"x": middle_right[0], "y": middle_right[1]},
            "far_left": {"x": far_left[0], "y": far_left[1]},
            "far_right": {"x": far_right[0], "y": far_right[1]},
        },
        "net_top_points": {
            "net_top_left": {"x": net_top_left[0], "y": net_top_left[1]},
            "net_top_right": {"x": net_top_right[0], "y": net_top_right[1]},
        },
        "net_top_calibrated": net_top_calibrated,
        # None whenever net_top_calibrated is False, or estimate_camera_pose
        # itself couldn't find a solution - ballDetection.load_camera_pose
        # treats either the same way (no height estimation for this job).
        "camera_pose": camera_pose_data,
        # Purely for the frontend to draw as a preview - the near baseline
        # and both attack lines, derived from the 4 points above (see
        # court.predict_court_geometry), never something the user edits.
        "predicted": {name: {"x": x, "y": y} for name, (x, y) in predicted.items()},
        # A full, sequential 4-corner quad (far baseline + predicted near
        # baseline) - kept in this exact shape for ball_detection.py's
        # court-area restriction (load_court_corners), which just wants a
        # simple polygon to test points against and doesn't care how those
        # corners were actually obtained.
        "image_points": [
            {"x": far_left[0], "y": far_left[1]},
            {"x": far_right[0], "y": far_right[1]},
            {"x": predicted["near_right"][0], "y": predicted["near_right"][1]},
            {"x": predicted["near_left"][0], "y": predicted["near_left"][1]},
        ],
        "net": {"height_m": net_height_m},
        "homography": matrix.tolist(),
        # Saving is itself the confirm action - "Set Court Identification"
        # is one combined save+lock step, same as the initial save. Redoing
        # (see set_confirmed above) is the only way this becomes False.
        "confirmed": True,
    }
    (output_path / config.COURT_FILE_NAME).write_text(json.dumps(data, indent=2))
    return data
