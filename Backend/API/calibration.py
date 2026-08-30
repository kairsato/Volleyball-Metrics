import json
import sys
from pathlib import Path
from typing import Optional

import cv2

# The stage modules live under Backend/Analysis/, a sibling of this API/ package.
BACKEND_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from CourtDefinition.court import (  # noqa: E402
    CALIBRATION_FRAME,
    PRESET_INSET_X,
    PRESET_INSET_Y,
    create_homography,
    net_point_presets,
    save_calibration,
)

from . import config

JPEG_QUALITY = 90


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
    """Mirrors the desktop tool's reset_points(): corners inset from the
    frame edges, net points guessed from the corner midpoints."""
    left = int(frame_width * PRESET_INSET_X)
    right = int(frame_width * (1.0 - PRESET_INSET_X))
    top = int(frame_height * PRESET_INSET_Y)
    bottom = int(frame_height * (1.0 - PRESET_INSET_Y))

    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]

    # A net point guessed above the frame is allowed - the web canvas lets
    # markers sit slightly past the frame edge - but only slightly, matching
    # the OVERFLOW_FRACTION allowance the frontend applies when dragging.
    overflow_fraction = 0.08
    min_y = -frame_height * overflow_fraction
    net_points = [(x, max(min_y, y)) for x, y in net_point_presets(corners)]

    return {
        "corners": [{"x": x, "y": y} for x, y in corners],
        "net_points": [{"x": x, "y": y} for x, y in net_points],
    }


def existing_points(output_path: Path) -> Optional[dict]:
    court_file = output_path / config.COURT_FILE_NAME
    if not court_file.exists():
        return None

    data = json.loads(court_file.read_text())
    return {
        "corners": data.get("image_points", []),
        "net_points": data.get("net_points", []),
    }


def save(output_path: Path, corners: list[tuple[float, float]], net_points: list[tuple[float, float]]) -> dict:
    if len(corners) != 4:
        raise ValueError("Exactly 4 court corners are required")
    if len(net_points) != 2:
        raise ValueError("Exactly 2 net points are required")

    matrix = create_homography(corners)
    output_path.mkdir(parents=True, exist_ok=True)
    save_calibration(matrix, str(output_path), corners, net_points)

    return json.loads((output_path / config.COURT_FILE_NAME).read_text())
