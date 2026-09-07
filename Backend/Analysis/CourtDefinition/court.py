import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

# Standard indoor volleyball court
COURT_WIDTH = 9.0
COURT_LENGTH = 18.0

# Which frame to use for calibration.
# 0 = first frame
CALIBRATION_FRAME = 0

# Side panel holding the instructions and buttons
PANEL_WIDTH = 380
PANEL_BG = (38, 38, 38)
MIN_CANVAS_HEIGHT = 740

# Empty space drawn around the video so corners can be dragged
# outside the frame without leaving the window
MARGIN = 140

# Preset corner positions as a fraction of the frame size
PRESET_INSET_X = 0.15
PRESET_INSET_Y = 0.20

# How close (in frame pixels) a click must be to grab a corner
GRAB_RADIUS = 18

CORNER_LABELS = [
    "1.  TOP LEFT",
    "2.  TOP RIGHT",
    "3.  BOTTOM RIGHT",
    "4.  BOTTOM LEFT"
]

# The net's top edge crosses each sideline at a known height - marking those
# two points lets ball detection later work out how high above the court an
# airborne ball actually is, instead of assuming it's on the ground. "FAR"/
# "NEAR" match the TOP/BOTTOM corner sideline each one sits on.
NET_LABELS = [
    "5.  NET TOP (FAR)",
    "6.  NET TOP (NEAR)"
]

ALL_LABELS = CORNER_LABELS + NET_LABELS

# Standard men's/mixed net height plus the antenna's mandatory extension
# above it - the antenna itself (a thin rod, usually brightly coloured) is
# normally much easier to click precisely than the net tape itself. Adjust
# NET_HEIGHT_M if this isn't a standard-height net (e.g. 2.24m for women's).
NET_HEIGHT_M = 2.43
ANTENNA_EXTENSION_M = 0.80
ANTENNA_HEIGHT_M = NET_HEIGHT_M + ANTENNA_EXTENSION_M

# Starting vertical offset (pixels) for the net points' preset position above
# each sideline's midpoint - just a rough guess to drag from, not a real
# estimate of the net's height in this particular shot.
NET_PRESET_HEIGHT_PX = 120

# How far (pixels) a net point has to move from its auto-generated preset
# before it's trusted as an actual calibration rather than an untouched
# guess - see save_calibration's "net_calibrated" flag.
NET_TOUCHED_TOLERANCE_PX = 3


# ============================================================
# GLOBAL STATE
# ============================================================

# Corner positions in frame coordinates; may fall outside the frame
points = []
frame = None

dragging_index = None
hovered_point = None

# name -> (x1, y1, x2, y2) in canvas coordinates
buttons = {}
hovered_button = None
pending_action = None

status_lines = []


def set_status(*lines):
    global status_lines
    status_lines = list(lines)


def net_point_presets(corners):
    """Rough starting guess for the two net-top points, based on wherever
    the four given court corners are - meant to be dragged onto the
    actual visible antenna top, not used as-is."""

    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

    far_mid = ((tl[0] + tr[0]) // 2, (tl[1] + tr[1]) // 2)
    near_mid = ((bl[0] + br[0]) // 2, (bl[1] + br[1]) // 2)

    return [
        (far_mid[0], far_mid[1] - NET_PRESET_HEIGHT_PX),
        (near_mid[0], near_mid[1] - NET_PRESET_HEIGHT_PX)
    ]


def reset_points():

    frame_height, frame_width = frame.shape[:2]

    left = int(frame_width * PRESET_INSET_X)
    right = int(frame_width * (1.0 - PRESET_INSET_X))
    top = int(frame_height * PRESET_INSET_Y)
    bottom = int(frame_height * (1.0 - PRESET_INSET_Y))

    points.clear()

    points.extend([
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom)
    ])

    points.extend(net_point_presets(points[:4]))


def load_saved_points(output_path):

    court_file = Path(output_path) / "court.json"

    if not court_file.exists():
        return False

    try:
        with open(court_file) as f:
            data = json.load(f)

        saved = [(int(p["x"]), int(p["y"])) for p in data["image_points"]]

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False

    if len(saved) != 4:
        return False

    points.clear()
    points.extend(clamp_to_canvas(x, y) for x, y in saved)

    # Net points are a newer addition - older court.json files won't have
    # them, and even a current one might have skipped them (see
    # save_calibration's "net_calibrated" flag). Either way, fall back to a
    # fresh preset guess rather than leaving the net uncalibrated.
    try:
        saved_net = [(int(p["x"]), int(p["y"])) for p in data["net_points"]]
        if len(saved_net) != 2:
            raise ValueError("expected exactly 2 net points")
        points.extend(clamp_to_canvas(x, y) for x, y in saved_net)
    except (KeyError, TypeError, ValueError):
        points.extend(net_point_presets(points[:4]))

    return True


def find_point(x, y):

    for i, (px, py) in enumerate(points):

        if abs(px - x) <= GRAB_RADIUS and abs(py - y) <= GRAB_RADIUS:
            return i

    return None


def clamp_to_canvas(x, y):

    frame_height, frame_width = frame.shape[:2]

    canvas_height = max(frame_height + MARGIN * 2, MIN_CANVAS_HEIGHT)

    # Limits expressed in frame coordinates
    top_limit = -MARGIN
    bottom_limit = frame_height + (canvas_height - frame_height - MARGIN)

    x = max(-MARGIN, min(frame_width + MARGIN - 1, x))
    y = max(top_limit, min(bottom_limit - 1, y))

    return x, y


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global hovered_button, pending_action
    global dragging_index, hovered_point

    frame_height, frame_width = frame.shape[:2]

    panel_x = frame_width + MARGIN * 2

    # Canvas -> frame coordinates
    fx = x - MARGIN
    fy = y - MARGIN

    if event == cv2.EVENT_MOUSEMOVE:

        if dragging_index is not None:
            points[dragging_index] = clamp_to_canvas(fx, fy)
            return

        hovered_button = None

        for name, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                hovered_button = name
                break

        hovered_point = None if hovered_button else find_point(fx, fy)

        return

    if event == cv2.EVENT_LBUTTONUP:

        if dragging_index is not None:
            set_status("Drag the corners to fit the court.")

        dragging_index = None

        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Click inside the panel = button press
    if x >= panel_x:

        for name, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                pending_action = name
                return

        return

    index = find_point(fx, fy)

    if index is None:
        return

    dragging_index = index
    hovered_point = index

    set_status(f"Moving {ALL_LABELS[index].strip()}")


# ============================================================
# RENDERING
# ============================================================

def draw_selection(canvas):

    # Points are stored in frame coordinates; shift them onto the canvas
    shifted = [(x + MARGIN, y + MARGIN) for x, y in points]

    # The four court corners form a closed quadrilateral; the two net-top
    # points are a separate pair, joined by their own line representing the
    # net's top edge - not part of the court quad.
    quad = shifted[:4]
    for i in range(len(quad)):
        cv2.line(canvas, quad[i], quad[(i + 1) % len(quad)],
                 (0, 255, 255), 2)

    if len(shifted) >= 6:
        cv2.line(canvas, shifted[4], shifted[5], (255, 200, 0), 2)

    for i, (x, y) in enumerate(shifted):

        active = (i == dragging_index) or (i == hovered_point)
        is_net_point = i >= 4

        radius = 11 if active else 8
        colour = (0, 165, 255) if active else ((255, 200, 0) if is_net_point else (0, 255, 0))

        cv2.circle(canvas, (x, y), radius, colour, -1)
        cv2.circle(canvas, (x, y), radius, (20, 20, 20), 1)

        cv2.putText(
            canvas,
            str(i + 1),
            (x + 14, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2
        )


def draw_button(canvas, rect, label, colour):

    x1, y1, x2, y2 = rect

    if label == hovered_button:
        colour = tuple(min(255, c + 45) for c in colour)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (235, 235, 235), 1)

    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

    cv2.putText(
        canvas,
        label,
        (x1 + (x2 - x1 - size[0]) // 2, y1 + (y2 - y1 + size[1]) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


def draw_panel(canvas, panel_x, canvas_height):

    buttons.clear()

    left = panel_x + 24
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, "COURT CALIBRATION", (left, 50),
                font, 0.8, (255, 255, 255), 2)

    cv2.line(canvas, (left, 68), (panel_x + PANEL_WIDTH - 24, 68),
             (90, 90, 90), 1)

    cv2.putText(canvas, "Drag markers 1-4 onto the", (left, 105),
                font, 0.55, (185, 185, 185), 1)

    cv2.putText(canvas, "court corners, 5-6 onto the", (left, 129),
                font, 0.55, (185, 185, 185), 1)

    cv2.putText(canvas, "net/antenna top (or leave as a", (left, 153),
                font, 0.55, (185, 185, 185), 1)

    cv2.putText(canvas, "rough guess if not visible):", (left, 177),
                font, 0.55, (185, 185, 185), 1)

    y = 217

    for i, label in enumerate(ALL_LABELS):

        active = (i == dragging_index) or (i == hovered_point)

        default_colour = (200, 165, 90) if i >= 4 else (200, 200, 200)
        colour = (0, 165, 255) if active else default_colour

        cv2.putText(canvas, f"{i + 1}.  {label.split('.')[1].strip()}",
                    (left, y), font, 0.62, colour, 2 if active else 1)

        y += 34

    y += 14

    cv2.putText(canvas, "Keys:  R reset   ENTER confirm", (left, y),
                font, 0.5, (150, 150, 150), 1)

    cv2.putText(canvas, "       ESC cancel", (left, y + 24),
                font, 0.5, (150, 150, 150), 1)

    button_width = PANEL_WIDTH - 48
    button_height = 54
    spacing = 16

    stack_height = button_height * 3 + spacing * 2
    button_top = canvas_height - 28 - stack_height

    specs = [
        ("CONFIRM", (40, 130, 40)),
        ("RESET", (130, 105, 30)),
        ("CANCEL", (40, 40, 140))
    ]

    for label, colour in specs:

        rect = (left, button_top,
                left + button_width, button_top + button_height)

        buttons[label] = rect

        draw_button(canvas, rect, label, colour)

        button_top += button_height + spacing

    # Status text sits above the button stack
    status_y = canvas_height - 28 - stack_height - 24 - 24 * len(status_lines)

    for line in status_lines[:5]:

        cv2.putText(canvas, line, (left, status_y),
                    font, 0.52, (0, 235, 255), 1)

        status_y += 24


def render():

    frame_height, frame_width = frame.shape[:2]

    canvas_height = max(frame_height + MARGIN * 2, MIN_CANVAS_HEIGHT)
    canvas_width = frame_width + MARGIN * 2 + PANEL_WIDTH

    canvas = np.full(
        (canvas_height, canvas_width, 3),
        PANEL_BG,
        dtype=np.uint8
    )

    canvas[MARGIN:MARGIN + frame_height, MARGIN:MARGIN + frame_width] = frame

    # Outline showing where the video ends
    cv2.rectangle(
        canvas,
        (MARGIN - 1, MARGIN - 1),
        (MARGIN + frame_width, MARGIN + frame_height),
        (110, 110, 110),
        1
    )

    draw_selection(canvas)

    draw_panel(canvas, frame_width + MARGIN * 2, canvas_height)

    return canvas


# ============================================================
# CREATE HOMOGRAPHY
# ============================================================

# ============================================================
# HALF-COURT CALIBRATION (4 points: net line + far baseline)
# ============================================================
#
# The web app only asks for 4 points - both ends of the net line and both
# ends of the far baseline - rather than the desktop tool's full 6-point set
# above (4 corners + 2 net-top). Those 4 points alone are enough to solve a
# full pixel<->court homography (cv2.getPerspectiveTransform needs exactly
# 4 correspondences), and once solved, every other line on the court - the
# near baseline and both attack lines - has a *known* real-world position
# (see the layout below), so where it lands in the image is a matter of
# projecting through the *inverse* of that same homography, not something
# the user has to mark by hand. Verified against a synthetic ground-truth
# perspective transform to sub-pixel accuracy.
#
#   x=0 (near)   x=6 (near attack)   x=9 (net/middle)   x=12 (far attack)   x=18 (far)
#        |------------------|--------------|------------------|------------------|
#      y=0 (left sideline)                                                  y=0
#      y=9 (right sideline)                                                 y=9

# Standard distance from the net to each attack (3m) line.
ATTACK_LINE_OFFSET_M = 3.0

HALF_COURT_DESTINATION = np.array([
    [COURT_LENGTH / 2, 0.0],           # middle_left  (net, left sideline)
    [COURT_LENGTH / 2, COURT_WIDTH],   # middle_right (net, right sideline)
    [COURT_LENGTH, 0.0],               # far_left
    [COURT_LENGTH, COURT_WIDTH],       # far_right
], dtype=np.float32)


def create_half_court_homography(middle_left, middle_right, far_left, far_right):
    """Solves the pixel -> court-metres homography from just the net line
    and far baseline - see the module diagram above. Argument order matters:
    it has to line up with HALF_COURT_DESTINATION."""
    source_points = np.array(
        [middle_left, middle_right, far_left, far_right], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(source_points, HALF_COURT_DESTINATION)


# ============================================================
# CAMERA POSE (adds net_top_left/net_top_right for ball-height estimation)
# ============================================================
#
# create_half_court_homography above only ever solves a flat pixel<->ground
# -plane mapping - it has no notion of "up" at all, so it can never tell an
# airborne ball's height above the court, only where it is horizontally.
# Recovering height needs the camera's actual 3D pose (where it sits and how
# it's oriented, not just a 2D warp), which needs at least one 3D reference
# point that ISN'T on the ground plane - the net's top edge, at the known
# height NET_HEIGHT_OPTIONS/net_height_m, is exactly that: together with the
# 4 ground corners it gives 6 known-3D-position <-> marked-pixel
# correspondences, spanning two different heights rather than lying flat on
# one plane.
#
# There's no real (multi-image) camera calibration here, so the focal
# length is unknown - only the principal point is assumed (image centre, a
# standard simplification absent an actual calibration target). With 6
# correspondences instead of the bare minimum 4, a plain 1D search over
# candidate focal lengths - solving cv2.solvePnP at each and keeping whichever
# minimizes reprojection error - is well-constrained enough to produce a
# usable (if approximate) pose. This is meaningfully weaker than a real
# multi-image calibration (there is a well-known focal-length/distance
# ambiguity in single-view pose recovery), so treat the result as an
# estimate: see CAMERA_POSE_MAX_REPROJECTION_ERROR_PX below for the
# confidence signal callers should surface, and BALL_HEIGHT_MAX_PLAUSIBLE_M
# in ballDetection.py for the sanity clamp on anything derived from it.

# Candidate focal lengths tried, as a fraction of the frame's own width -
# corresponds to roughly a 15-70 degree horizontal field of view for a
# typical court-side camera/phone, comfortably covering both a tight
# telephoto shot from the stands and a wide-angle courtside mount.
_FOCAL_LENGTH_SEARCH_FRACTIONS = np.linspace(0.5, 3.0, 60)


def camera_pose_candidates(
    middle_left, middle_right, far_left, far_right,
    net_top_left, net_top_right, net_height_m,
    frame_width, frame_height,
):
    """
    Every (K, rvec, tvec, reprojection_error_px) a cv2.solvePnP solve finds
    across _FOCAL_LENGTH_SEARCH_FRACTIONS' candidate focal lengths - the full
    search space estimate_camera_pose below picks its single best-
    reprojection answer from.

    Exposed separately (rather than folded directly into
    estimate_camera_pose) so a caller with an independent signal for which
    candidate is actually right - see BallDetection.ballDetection.
    refine_camera_pose_from_flight, which uses the ball's own physically-
    known parabolic flight instead of reprojection error alone - can re-rank
    this same search space instead of re-deriving it.
    """
    world_points = np.array([
        [COURT_LENGTH / 2, 0.0, 0.0],             # middle_left
        [COURT_LENGTH / 2, COURT_WIDTH, 0.0],     # middle_right
        [COURT_LENGTH, 0.0, 0.0],                 # far_left
        [COURT_LENGTH, COURT_WIDTH, 0.0],         # far_right
        [COURT_LENGTH / 2, 0.0, net_height_m],    # net_top_left
        [COURT_LENGTH / 2, COURT_WIDTH, net_height_m],  # net_top_right
    ], dtype=np.float64)

    image_points = np.array([
        middle_left, middle_right, far_left, far_right, net_top_left, net_top_right,
    ], dtype=np.float64)

    principal_point = (frame_width / 2.0, frame_height / 2.0)

    candidates = []
    for focal_fraction in _FOCAL_LENGTH_SEARCH_FRACTIONS:
        focal_px = float(focal_fraction * frame_width)
        K = np.array([
            [focal_px, 0.0, principal_point[0]],
            [0.0, focal_px, principal_point[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            world_points, image_points, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            continue

        reprojected, _ = cv2.projectPoints(world_points, rvec, tvec, K, None)
        error_px = float(np.linalg.norm(reprojected.reshape(-1, 2) - image_points, axis=1).mean())

        candidates.append((K, rvec, tvec, error_px))

    return candidates


def estimate_camera_pose(
    middle_left, middle_right, far_left, far_right,
    net_top_left, net_top_right, net_height_m,
    frame_width, frame_height,
):
    """
    Recovers the camera's 3D pose (intrinsics + extrinsics) from the 6
    marked calibration points, by searching over candidate focal lengths and
    keeping whichever gives the lowest reprojection error against a
    cv2.solvePnP solve - see the module comment above for why this is
    needed and its limits.

    Returns (K, rvec, tvec, reprojection_error_px), or None if no candidate
    focal length produced a valid solvePnP solution at all (e.g. degenerate/
    near-identical marked points).
    """
    candidates = camera_pose_candidates(
        middle_left, middle_right, far_left, far_right,
        net_top_left, net_top_right, net_height_m,
        frame_width, frame_height,
    )
    if not candidates:
        return None

    return min(candidates, key=lambda candidate: candidate[3])


def world_to_camera(point_world, rvec, tvec):
    """P_cam = R @ P_world + tvec - the extrinsic transform solvePnP solves
    for, world metres -> camera-space metres."""
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation @ np.asarray(point_world, dtype=np.float64) + tvec.reshape(3)


def camera_to_world(point_camera, rvec, tvec):
    """Inverse of world_to_camera: P_world = R^T @ (P_cam - tvec)."""
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation.T @ (np.asarray(point_camera, dtype=np.float64) - tvec.reshape(3))


def predict_court_geometry(matrix):
    """Everything else on the court, derived purely from the homography
    above and the court's known real dimensions - never marked by the user.
    Returns pixel coordinates (as (x, y) tuples) for the near baseline's two
    corners and both attack lines' sideline-to-sideline endpoints."""
    inverse = np.linalg.inv(matrix)

    def to_pixel(court_x, court_y):
        point = np.array([[[court_x, court_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, inverse)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    near_x = 0.0
    far_attack_x = COURT_LENGTH / 2 + ATTACK_LINE_OFFSET_M
    near_attack_x = COURT_LENGTH / 2 - ATTACK_LINE_OFFSET_M

    return {
        "near_left": to_pixel(near_x, 0.0),
        "near_right": to_pixel(near_x, COURT_WIDTH),
        "attack_far_left": to_pixel(far_attack_x, 0.0),
        "attack_far_right": to_pixel(far_attack_x, COURT_WIDTH),
        "attack_near_left": to_pixel(near_attack_x, 0.0),
        "attack_near_right": to_pixel(near_attack_x, COURT_WIDTH),
    }


def create_homography(image_points):

    # IMPORTANT:
    # Keep these in the same order as the points you click:
    #
    # 1 = top-left
    # 2 = top-right
    # 3 = bottom-right
    # 4 = bottom-left

    destination_points = np.array([
        [0.0, 0.0],
        [COURT_LENGTH, 0.0],
        [COURT_LENGTH, COURT_WIDTH],
        [0.0, COURT_WIDTH]
    ], dtype=np.float32)

    source_points = np.array(
        image_points,
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    return matrix


# ============================================================
# SAVE CALIBRATION
# ============================================================

def save_calibration(matrix, output_path, image_points, net_points):

    # An untouched preset guess isn't a real calibration - flag it as such
    # so ball detection knows whether to trust these points for height
    # estimation or ignore them.
    net_calibrated = any(
        abs(nx - px) > NET_TOUCHED_TOLERANCE_PX or abs(ny - py) > NET_TOUCHED_TOLERANCE_PX
        for (nx, ny), (px, py) in zip(net_points, net_point_presets(image_points))
    )

    data = {
        "court": {
            "length_m": COURT_LENGTH,
            "width_m": COURT_WIDTH
        },

        "image_points": [
            {
                "x": int(x),
                "y": int(y)
            }
            for x, y in image_points
        ],

        "net_points": [
            {
                "x": int(x),
                "y": int(y)
            }
            for x, y in net_points
        ],

        "net_calibrated": net_calibrated,

        "net": {
            "height_m": NET_HEIGHT_M,
            "antenna_extension_m": ANTENNA_EXTENSION_M,
            "antenna_height_m": ANTENNA_HEIGHT_M
        },

        "homography": matrix.tolist()
    }

    with open(Path(output_path) / "court.json", "w") as f:
        json.dump(data, f, indent=4)



# ============================================================
# PIXEL → COURT COORDINATES
# ============================================================

def pixel_to_court(x, y, matrix):

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        matrix
    )

    court_x = float(transformed[0][0][0])
    court_y = float(transformed[0][0][1])

    return court_x, court_y


# ============================================================
# MAIN
# ============================================================

def courtDefine(video_path,output_path):

    global frame
    global pending_action

    video_path = Path(video_path)

    if not video_path.exists():
        print(f"ERROR: Video not found:")
        print(video_path)
        return

    print("Opening video...")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    # Move to calibration frame
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        CALIBRATION_FRAME
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        print("ERROR: Could not read video frame.")
        return

    reset_points()

    if load_saved_points(output_path):
        set_status("Loaded saved corners.", "Drag them to adjust.")
    else:
        set_status("Drag the corners to fit the court.")

    window_name = "Court Calibration"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    canvas = render()

    window_height = min(900, canvas.shape[0])
    window_width = int(canvas.shape[1] * window_height / canvas.shape[0])

    cv2.resizeWindow(
        window_name,
        window_width,
        window_height
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    while True:

        cv2.imshow(
            window_name,
            render()
        )

        key = cv2.waitKey(20) & 0xFF

        action = pending_action
        pending_action = None

        if key == 27:
            action = "CANCEL"
        elif key == ord("r"):
            action = "RESET"
        elif key in (13, 10):
            action = "CONFIRM"

        if action is None:
            continue

        if action == "CANCEL":
            break

        if action == "RESET":

            reset_points()

            set_status("Corners reset to default positions.")

            continue

        matrix = create_homography(points[:4])

        save_calibration(matrix, output_path, points[:4], points[4:6])

        set_status(
            "Calibration complete.",
            f"Saved: {output_path}"
        )

        cv2.imshow(window_name, render())
        cv2.waitKey(900)

        break

    cv2.destroyAllWindows()


