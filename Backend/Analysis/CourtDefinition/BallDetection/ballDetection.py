import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from CourtDefinition.court import COURT_LENGTH, COURT_WIDTH

# Ball detection and speed estimation.
#
# The detector (ball_yolo11m_finetuned.pt) is a YOLO11m fine-tuned specifically
# on volleyball footage - single class ("volleyball"), ~15k training images
# from the open VolleyVision/Volleyball_v2 dataset (Roboflow, CC BY 4.0,
# credit: shukur-sabzaliev1). Validation after fine-tuning: precision 0.929,
# recall 0.758, mAP50 0.856, mAP50-95 0.525. This replaced an earlier version
# that used YOLO's pretrained COCO "sports ball" class as a generic stand-in -
# a real volleyball-specific detector is a large step up, but it was trained
# on other people's footage/venues, not this project's own, so it won't be
# perfectly calibrated for every camera/lighting setup. On top of whatever the
# detector itself gets right, this pipeline works to squeeze out usable recall
# and precision despite the ball being tiny (~15-25px on a 1920x1080 source)
# and often motion-blurred:
#
#   1. Acquisition: while there's no active track, the whole frame is split
#      into overlapping tiles (see ACQUISITION_TILE_GRID) and each tile is
#      run through YOLO near its native resolution. This finds the ball at a
#      much higher effective resolution than resizing the entire frame down
#      to fit one inference pass ever could - and it only runs while there's
#      no lock, not on every frame.
#   2. Tracking: once the ball is found, a constant-velocity Kalman filter
#      (BallKalmanFilter) predicts where it should be next frame, and
#      detection narrows to a small, tightly-scoped crop around that
#      prediction (ZOOM_IMG_SIZE). That crop sees the ball at or above its
#      native pixel size - effectively zooming in - which recovers a lot of
#      the frames a full-frame pass would lose to blur or scale, and it's
#      cheap enough to run every frame since it's one small-image inference,
#      not five full-frame ones (the old five-tracker ensemble bought almost
#      nothing: every tracker sat on top of the same underlying detector, so
#      they mostly missed and false-positived together - the 5x compute was
#      spent on redundant agreement, not on actually seeing the ball better).
#   3. The filter's own uncertainty (grows while coasting through a miss,
#      shrinks while locked on) drives both how far a detection is allowed to
#      be from the prediction and still count (GATE_SIGMA_MULTIPLIER) and how
#      large the next zoom crop needs to be - so a brief occlusion widens the
#      search automatically, and a clean run keeps it tight and fast.
#   4. If the ball still isn't found after MAX_COAST_SECONDS, the track is
#      dropped and the pipeline falls back to the full tiled scan to
#      reacquire it from scratch.
#
# If recall/precision still aren't good enough on this project's own venues
# after all of the above, the next-highest-ceiling fix is fine-tuning further
# on this project's own footage specifically (the current model has never
# seen it), rather than more tracking-layer tuning.

MODEL_PATH = str(Path(__file__).parent / "ball_yolo11m_finetuned.pt")
OUTPUT_VIDEO_NAME = "ball.mp4"
SPEED_LOG_NAME = "ball_speed.json"

BALL_CLASS_ID = 0  # single-class model: 0 = "volleyball"

# Confidence for the full-frame tiled acquisition scan. 0.05 let through far
# too much noise in practice - weak misclassifications of ceiling fixtures,
# lighting, exit signs and wall-mirror reflections were winning out over the
# real ball. 0.25 is a safer default; if recall is still too low, lower this
# back down gradually.
CONFIDENCE_THRESHOLD = 0.25

# Fallback detector, tried only on frames the fine-tuned model itself comes
# up empty on (both in the zoom crop while tracking, and the tiled scan while
# acquiring). This is the original generic pretrained checkpoint this project
# used before fine-tuning - a real volleyball detector and a generic "any
# sports ball" classifier tend to miss different frames (different training
# data, different failure modes on blur/scale), so trying a second, genuinely
# different detector on an otherwise-empty frame can recover some misses a
# single model - however good - never will. Deliberately only ever invoked on
# a miss, not every frame, since it's the same cost as a normal detection
# pass and most frames don't need it.
FALLBACK_MODEL_PATH = str(Path(__file__).parent.parent.parent / "yolo26x.pt")
FALLBACK_BALL_CLASS_ID = 32  # COCO class index for "sports ball"

# The fallback detector is a generic "any sports ball" classifier - it
# happily calls plenty of round, ball-sized things a sports ball that don't
# actually look anything like this video's ball (a player's head, a dark
# jersey, a wall fixture). A colour histogram is a near-free way (no second
# model, just a crop and a histogram compare) to catch most of those: the
# reference is built from the fine-tuned model's own confident detections as
# the video plays, so it self-calibrates to this specific ball/lighting/venue
# rather than needing a hand-picked reference colour up front.
HIST_BINS_H = 30
HIST_BINS_S = 32
# Hue+Saturation only (no Value/brightness) so the comparison is reasonably
# tolerant of the exposure changes a ball picks up moving through light and
# shadow, which would otherwise swamp the actual colour signal.
HIST_REFERENCE_MIN_SAMPLES = 20  # confident detections needed before the veto kicks in
HIST_EMA_ALPHA = 0.05  # how fast the reference adapts to new confident detections
# Bhattacharyya distance (0 = identical, 1 = totally different) above which a
# fallback candidate is rejected as not colour-matching the reference, even
# though the fallback model called it a "sports ball".
HIST_DISTANCE_THRESHOLD = 0.65

# Name of the calibration file CourtDefinition.court saves into output_path
COURT_FILE_NAME = "court.json"

# Official volleyball diameter in metres - kept for reference; not required by
# the homography-based speed calculation below, but useful if this is ever
# extended to estimate height/depth from the ball's apparent size.
BALL_DIAMETER_M = 0.21

# --- Acquisition (full-frame search, used while there's no active track) ---

# The frame is split into a cols x rows grid of overlapping tiles, each run
# through YOLO near its own native resolution rather than the whole frame
# being squashed down to one inference size. Tuned for ~1920x1080 source
# footage; a much higher- or lower-resolution source may want a different
# grid (more tiles for 4K, fewer/none for something already small).
ACQUISITION_TILE_GRID = (2, 2)  # (cols, rows)
# Overlap as a fraction of a tile's own size, generous enough that a ball
# straddling a tile boundary still falls entirely within at least one tile.
ACQUISITION_TILE_OVERLAP_FRACTION = 0.2
ACQUISITION_IMG_SIZE = 960
# How closely two tiles' boxes must overlap to be treated as the same ball
# detected twice in the overlap margin, rather than two separate objects.
TILE_NMS_IOU_THRESH = 0.3

# --- Tracking (local zoom search, used once a track is locked on) ---

ZOOM_IMG_SIZE = 640
# The zoom crop is a small, tightly-scoped region, so a stray
# misclassification is far less likely to wander in than during a full-frame
# scan - safe to accept weaker detections here than the acquisition pass would.
ZOOM_CONFIDENCE_THRESHOLD = 0.15
# Margin added on top of the gate radius (see below) when sizing the zoom
# crop, so the crop always comfortably contains the region a detection would
# actually be accepted from.
ZOOM_MARGIN_PX = 60
ZOOM_MIN_HALF_SIZE_PX = 120
ZOOM_MAX_HALF_SIZE_PX = 420

# --- Kalman filter tuning (position/velocity in pixel space) ---

# How much the ball's velocity is assumed to change per second, as a fraction
# of the frame diagonal per second^2. Deliberately generous: a spike or a
# block deflection can reverse the ball's velocity in a couple of frames, and
# a filter tuned for smooth, gently-curving motion would lag badly or reject
# the very moments (hits) that matter most.
ACCEL_NOISE_FRACTION = 1.2
# Assumed pixel jitter in a single YOLO box's centre, as a fraction of the
# frame diagonal - i.e. how much to trust one detection vs. the filter's own
# running estimate.
MEASUREMENT_NOISE_FRACTION = 0.006

# A detection must land within this many standard deviations of the filter's
# predicted position to be accepted as the ball rather than something else.
# GATE_MIN/MAX_FRACTION bound the resulting radius so it's never so tight
# that normal detection jitter fails the gate, nor so loose that after a long
# coast it snaps onto an unrelated object anywhere nearby.
GATE_SIGMA_MULTIPLIER = 4.0
GATE_MIN_FRACTION = 0.02
GATE_MAX_FRACTION = 0.30

# When there's no active track to gate against (first-ever lock, or just
# reacquiring), multiple simultaneous "sports ball" candidates can be real at
# once - e.g. an adjacent court's game, or one reflected in a venue mirror,
# both show a real ball a plain classifier can't tell apart from this court's
# own. Apparent box size is used as a tiebreaker in that case (a ball closer
# to this camera subtends more pixels than the same object seen further away
# in the background), restricted to candidates within this fraction of the
# top confidence so a low-confidence but large misclassification can't win on
# size alone.
ACQUISITION_SIZE_TIEBREAK_CONF_FRACTION = 0.7

# A real ball is ~15-25px across on this project's ~1920x1080 source at its
# closest to the camera, and necessarily smaller (never larger) further away
# - anything detected noticeably below that is far more likely a stray
# single/few-pixel artifact than the actual ball, however the detector
# labelled it. Deliberately conservative (well under the near-camera size) so
# this only cuts obvious noise, not a genuinely distant ball.
MIN_BALL_DIAGONAL_PX = 6.0

# How often, even while confidently locked on, to run a full-frame tiled
# scan instead of the narrow zoom crop, purely to sanity-check the lock
# against the rest of the frame (see BallTracker.revalidate). Without this,
# a lock that drifted onto a plausible-but-wrong moving object (a player's
# head, tracked as if it were the ball) can never self-correct, since the
# zoom crop it searches every other frame is centred on that same wrong
# object and never sees anything else. Frequent enough to catch and correct
# a bad lock within a couple of seconds; infrequent enough that the extra
# full-frame cost stays a small fraction of total runtime.
REVALIDATION_INTERVAL_SECONDS = 2.0

# How many consecutive frames the filter is allowed to coast on prediction
# alone (no matching detection) before the track is considered lost and
# detection falls back to a full-frame scan. Generous enough to bridge
# typical motion blur or a brief hand/body occlusion; short enough that the
# trail doesn't drift over empty air for too long once actually lost.
MAX_COAST_SECONDS = 0.5

# The fastest volleyball spikes ever recorded are around 35 m/s (~126 km/h).
# A homography-projected speed above this is never real ball motion - it's
# the flat ground-plane projection breaking down for a detection well above
# the court (see pixel_to_court) - so it's never reported as real-world
# speed, even though the underlying pixel position is still fine to track.
MAX_PLAUSIBLE_REPORTED_SPEED_MS = 45.0

# Speed is measured between the current position and one this many frames
# back, not frame-to-frame, since single-frame position noise would
# otherwise dominate a 1/dt term at typical video frame rates.
SPEED_WINDOW_FRAMES = 3

# How many recent pixel positions to draw as a fading trail
TRAIL_LENGTH = 15

# Top-down court minimap drawn in the corner of the output video, showing
# where the ball actually is on the court rather than just its position in
# this particular camera's perspective. Sized to the real court's own 2:1
# (length:width) aspect ratio so the diagram isn't distorted. Only drawn when
# a homography is available - without one there's no real-world position to
# place it at.
MINIMAP_WIDTH = 320
MINIMAP_HEIGHT = 160
MINIMAP_MARGIN_PX = 16

# Anything classified as "sports ball" that keeps reappearing in almost the
# same spot for this many continuous seconds is learned as a static, non-ball
# object (a wall-mounted screen, a fixture, a poster) and excluded from then
# on. The court ROI only excludes candidates by WHERE they are (e.g. the
# sideline basket); this complements it by excluding candidates by BEHAVIOUR
# (never actually moving).
STATIC_ZONE_LEARN_SECONDS = 4.0
STATIC_ZONE_RADIUS_FRACTION = 0.03       # of frame diagonal - how close counts as "the same spot"
STATIC_ZONE_EXCLUSION_RADIUS_FRACTION = 0.05  # slightly larger, to cover the rest of the object

# Where ball detections are considered valid: the court's own bounding box,
# padded outward for balls that land just outside the lines, and extended all
# the way to the top of the frame - the court polygon alone only covers the
# playing surface, but the ball spends most of its time well above it (a jump
# serve can carry it very high in frame).
BALL_ROI_PADDING_FRACTION = 0.08       # of frame diagonal - sideways/below the court lines


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


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = float(boxAArea + boxBArea - interArea)

    if union <= 0:
        return 0.0

    return interArea / union


def unpack_ball_boxes(result):
    """Every ball-class box a detection pass produced, as (box_xyxy,
    confidence) arrays."""

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))

    return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()


def crop_frame(frame, box):
    """Clip `box` (xyxy, float, may extend past the frame) to the frame and
    return (crop, (offset_x, offset_y)), or (None, ...) if it's empty."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = int(np.clip(np.floor(x1), 0, width))
    y1 = int(np.clip(np.floor(y1), 0, height))
    x2 = int(np.clip(np.ceil(x2), 0, width))
    y2 = int(np.clip(np.ceil(y2), 0, height))

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1)

    return frame[y1:y2, x1:x2].copy(), (x1, y1)


def generate_tiles(frame_width, frame_height, grid, overlap_fraction):
    """Overlapping tile boxes covering the full frame - see
    ACQUISITION_TILE_GRID/ACQUISITION_TILE_OVERLAP_FRACTION."""

    cols, rows = grid
    tile_w = frame_width / cols
    tile_h = frame_height / rows
    pad_x = tile_w * overlap_fraction
    pad_y = tile_h * overlap_fraction

    tiles = []
    for row in range(rows):
        for col in range(cols):
            x1 = max(0.0, col * tile_w - pad_x)
            y1 = max(0.0, row * tile_h - pad_y)
            x2 = min(float(frame_width), (col + 1) * tile_w + pad_x)
            y2 = min(float(frame_height), (row + 1) * tile_h + pad_y)
            tiles.append((x1, y1, x2, y2))

    return tiles


def non_max_suppress(candidates, iou_thresh):
    """Collapse duplicate detections of the same ball caught by two
    overlapping tiles down to the single highest-confidence box."""

    ordered = sorted(candidates, key=lambda c: c[1], reverse=True)
    kept = []

    for box, conf in ordered:
        if all(calculate_iou(box, kept_box) <= iou_thresh for kept_box, _ in kept):
            kept.append((box, conf))

    return kept


def detect_in_crop(model, frame, crop_box, imgsz, conf, device, class_id=BALL_CLASS_ID):
    """Run YOLO on a sub-region of the frame, returning detections mapped
    back into full-frame pixel coordinates."""

    crop, (ox, oy) = crop_frame(frame, crop_box)
    if crop is None:
        return []

    result = model.predict(source=crop, classes=[class_id], conf=conf,
                            imgsz=imgsz, device=device, verbose=False)[0]
    boxes, confs = unpack_ball_boxes(result)
    offset = np.array([ox, oy, ox, oy], dtype=float)

    return [(box + offset, float(c)) for box, c in zip(boxes, confs)]


def detect_tiled(model, frame, tiles, imgsz, conf, device, class_id=BALL_CLASS_ID):
    """Run YOLO across every acquisition tile in one batched call, returning
    deduplicated detections in full-frame pixel coordinates."""

    crops = []
    offsets = []

    for tile in tiles:
        crop, offset = crop_frame(frame, tile)
        if crop is None:
            continue
        crops.append(crop)
        offsets.append(offset)

    if not crops:
        return []

    results = model.predict(source=crops, classes=[class_id], conf=conf,
                             imgsz=imgsz, device=device, verbose=False)

    candidates = []
    for result, (ox, oy) in zip(results, offsets):
        boxes, confs = unpack_ball_boxes(result)
        offset = np.array([ox, oy, ox, oy], dtype=float)
        for box, c in zip(boxes, confs):
            candidates.append((box + offset, float(c)))

    return non_max_suppress(candidates, TILE_NMS_IOU_THRESH)


class BallColorProfile:
    """
    Running colour signature of the ball in this specific video, built only
    from the fine-tuned model's own confident detections. Used to veto
    fallback-detector candidates that don't colour-match it - see
    HIST_DISTANCE_THRESHOLD.
    """

    def __init__(self):
        self.histogram = None
        self.samples = 0

    @property
    def is_confident(self):
        return self.samples >= HIST_REFERENCE_MIN_SAMPLES

    def _histogram_for(self, frame, box):
        crop, _ = crop_frame(frame, box)
        if crop is None or crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS_H, HIST_BINS_S], [0, 180, 0, 256])
        total = hist.sum()

        if total <= 0:
            return None

        return (hist / total).astype(np.float32)

    def update(self, frame, box):
        hist = self._histogram_for(frame, box)
        if hist is None:
            return

        if self.histogram is None:
            self.histogram = hist
        else:
            self.histogram = (1 - HIST_EMA_ALPHA) * self.histogram + HIST_EMA_ALPHA * hist

        self.samples += 1

    def distance(self, frame, box):
        """Bhattacharyya distance to the reference (0 = identical, 1 =
        totally different), or None if there isn't yet a confident enough
        reference to judge against."""
        if not self.is_confident:
            return None

        hist = self._histogram_for(frame, box)
        if hist is None:
            return None

        return cv2.compareHist(self.histogram, hist, cv2.HISTCMP_BHATTACHARYYA)


class StaticZoneLearner:
    """
    Learns to ignore "ball" detections that never actually move - a wall-
    mounted screen, a fixture, a poster, anything that keeps getting
    (mis)classified as a sports ball in the same spot. Any raw detection is
    tracked as a candidate cluster by position; if the same spot keeps
    getting detected for STATIC_ZONE_LEARN_SECONDS straight, it's promoted to
    a permanently excluded zone.
    """

    def __init__(self, fps, frame_width, frame_height):
        diagonal = float(np.hypot(frame_width, frame_height))
        self.cluster_radius = STATIC_ZONE_RADIUS_FRACTION * diagonal
        self.exclusion_radius = STATIC_ZONE_EXCLUSION_RADIUS_FRACTION * diagonal
        self.learn_frames = STATIC_ZONE_LEARN_SECONDS * fps
        self.candidates = []  # [{"centre", "first_seen", "last_seen"}, ...] - not yet confirmed static
        self.excluded = []    # [centre, ...] - confirmed static, permanently ignored

    def is_excluded(self, pixel):
        return any(np.linalg.norm(pixel - centre) <= self.exclusion_radius
                   for centre in self.excluded)

    def observe(self, detections, frame_idx):
        """
        Feed this frame's raw detections (already excluded ones should be
        filtered out by the caller) to learn/refresh candidate static spots.
        Returns the centres of any zone newly excluded this call, so the
        caller can reset a tracker that was actively following one of them.
        """
        newly_excluded = []

        for box, _ in detections:
            pixel = centre_of(box)

            match = next((c for c in self.candidates
                         if np.linalg.norm(pixel - c["centre"]) <= self.cluster_radius), None)

            if match is None:
                self.candidates.append({"centre": pixel, "first_seen": frame_idx, "last_seen": frame_idx})
                continue

            match["last_seen"] = frame_idx

            if frame_idx - match["first_seen"] >= self.learn_frames:
                self.excluded.append(match["centre"])
                # list.remove() would fall back to a full dict == comparison
                # (including the numpy "centre" array) against every other
                # entry it isn't identical to, which raises - remove by
                # identity instead of value.
                self.candidates = [c for c in self.candidates if c is not match]
                newly_excluded.append(match["centre"])
                print(f"Ball detection: learned a static non-ball zone near "
                      f"{tuple(match['centre'].astype(int))} (e.g. a wall fixture) - excluding it from now on.")

        # A candidate that's gone quiet for a while probably wasn't a static
        # object after all (or the detector just missed it) - drop it rather
        # than let it linger forever.
        self.candidates = [c for c in self.candidates
                           if frame_idx - c["last_seen"] <= self.learn_frames]

        return newly_excluded


class BallKalmanFilter:
    """
    Constant-velocity Kalman filter over the ball's pixel position. Tracking
    a single point (rather than the multi-object case PlayerDetection deals
    with) makes a plain linear KF a good fit: it gives a principled predicted
    position *and* an uncertainty estimate every frame, which the frame loop
    uses both to gate incoming detections (reject anything too far from where
    the ball should be) and to size the local re-detection search window
    (grow it while coasting, keep it tight while locked on).
    """

    def __init__(self, diagonal):
        accel_std = ACCEL_NOISE_FRACTION * diagonal
        self._accel_var = accel_std ** 2
        measurement_std = MEASUREMENT_NOISE_FRACTION * diagonal
        self.R = np.eye(2) * (measurement_std ** 2)
        self.x = None
        self.P = None

    @property
    def initialized(self):
        return self.x is not None

    def init(self, x, y):
        # Position variance starts small (we just saw it there); velocity is
        # completely unknown at first, so its variance starts large.
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, self._accel_var, self._accel_var]) * 100.0

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        q = self._accel_var
        Q = q * np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2],
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        return self.x[:2].copy()

    def update(self, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = np.asarray(z, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def position(self):
        return self.x[:2].copy()

    def position_std(self):
        return np.sqrt(np.clip(np.diag(self.P)[:2], 0, None))


class BallTracker:
    """
    Owns the Kalman filter tracking the ball's pixel position across frames,
    plus the trail/speed bookkeeping built on top of it. Gating and
    search-window sizing come from the filter's own uncertainty, so both
    tighten automatically while locked on and loosen automatically while
    coasting through a miss.
    """

    def __init__(self, matrix, fps, frame_width, frame_height):
        self.matrix = matrix
        self.fps = fps
        self.diagonal = float(np.hypot(frame_width, frame_height))
        self.history = []
        self.kf = BallKalmanFilter(self.diagonal)
        self.coast_frames = 0
        self.max_coast_frames = max(1, round(MAX_COAST_SECONDS * fps))

    @property
    def is_tracking(self):
        return self.kf.initialized

    def is_lost(self):
        return self.coast_frames > self.max_coast_frames

    def predict(self):
        """Advance the filter by one frame. Returns (position, std), or
        (None, None) if there's no active track to predict from."""
        if not self.kf.initialized:
            return None, None
        return self.kf.predict(1.0 / self.fps), self.kf.position_std()

    def gate_radius(self, std):
        radius = GATE_SIGMA_MULTIPLIER * float(np.mean(std))
        return float(np.clip(radius, GATE_MIN_FRACTION * self.diagonal, GATE_MAX_FRACTION * self.diagonal))

    def zoom_crop(self, predicted, std):
        """Local search window for the next frame - always at least large
        enough to contain the region a detection would actually be accepted
        from (see gate_radius)."""
        half = self.gate_radius(std) + ZOOM_MARGIN_PX
        half = float(np.clip(half, ZOOM_MIN_HALF_SIZE_PX, ZOOM_MAX_HALF_SIZE_PX))
        return predicted[0] - half, predicted[1] - half, predicted[0] + half, predicted[1] + half

    def match(self, candidates, predicted, std):
        """
        Pick which candidate (if any) is the game ball this frame. With an
        active prediction, candidates outside the uncertainty-scaled gate
        are discarded outright as probably an unrelated object; among
        whatever's left inside the gate, the biggest wins (closer to this
        camera, same reasoning as the no-prediction case below) rather than
        just whichever happens to be nearest the predicted point. Without a
        prediction (no track yet, or just abandoned), fall back to the
        biggest of whichever candidates are close to the top confidence -
        see ACQUISITION_SIZE_TIEBREAK_CONF_FRACTION.
        """
        if not candidates:
            return None

        if predicted is None:
            return self._best_by_size(candidates)

        radius = self.gate_radius(std)
        within_gate = [d for d in candidates
                       if np.linalg.norm(centre_of(d[0]) - predicted) <= radius]

        if not within_gate:
            return None

        return self._best_by_size(within_gate)

    def _best_by_size(self, candidates):
        """Among whichever candidates are within
        ACQUISITION_SIZE_TIEBREAK_CONF_FRACTION of the top confidence, the
        one with the largest apparent box."""
        best_confidence = max(conf for _, conf in candidates)
        threshold = best_confidence * ACQUISITION_SIZE_TIEBREAK_CONF_FRACTION
        contenders = [d for d in candidates if d[1] >= threshold]
        return max(contenders, key=lambda d: box_area(d[0]))

    def revalidate(self, candidates, predicted, std):
        """
        Periodic whole-frame sanity check while tracking (see
        REVALIDATION_INTERVAL_SECONDS). A locked-on track normally only
        ever looks at the small zoom crop around its own prediction, so if
        it drifted onto a plausible-but-wrong object that moves enough to
        dodge the static-zone check (a player's head is the case this was
        written for), it has no way to notice on its own. This compares the
        normal gated match against the best candidate anywhere in the full
        frame, and only overrides the lock if that candidate is a clearly
        distinct (outside the current gate), comparably-confident, bigger
        object - not just noisy jitter of the same detection.
        """
        gated = self.match(candidates, predicted, std)

        if not candidates:
            return gated

        best_overall = self._best_by_size(candidates)

        if gated is not None and np.array_equal(best_overall[0], gated[0]):
            return gated

        if predicted is not None:
            radius = self.gate_radius(std)
            outside_gate = np.linalg.norm(centre_of(best_overall[0]) - predicted) > radius
        else:
            outside_gate = True

        if outside_gate and (gated is None or box_area(best_overall[0]) > box_area(gated[0])):
            # Something elsewhere in the frame is a clearly better ball
            # candidate than whatever's currently locked on - the lock has
            # likely drifted onto the wrong object, so drop it and start
            # fresh from this one instead of blending toward it.
            self.lose()
            return best_overall

        return gated

    def observe(self, detection, frame_idx):
        """
        Record this frame's outcome: `detection` is (box, confidence) if a
        match was accepted, or None if the filter is coasting on prediction
        alone. Returns a speed_log reading, or None if there's nothing to
        report yet (no track, or not enough history for a speed window).
        """
        if detection is not None:
            box, confidence = detection
            pixel = centre_of(box)

            if not self.kf.initialized:
                self.kf.init(pixel[0], pixel[1])
            else:
                self.kf.update(pixel)

            self.coast_frames = 0
            interpolated = False
        else:
            if not self.kf.initialized:
                return None

            pixel = self.kf.position()
            confidence = None
            self.coast_frames += 1
            interpolated = True

        court = pixel_to_court(pixel[0], pixel[1], self.matrix) if self.matrix is not None else None
        point = {"frame_idx": frame_idx, "pixel": pixel, "court": court,
                 "confidence": confidence, "interpolated": interpolated}

        self.history.append(point)
        if len(self.history) > TRAIL_LENGTH:
            self.history.pop(0)

        if len(self.history) < 2:
            return None

        window_start = self.history[max(0, len(self.history) - 1 - SPEED_WINDOW_FRAMES)]

        # The homography is only accurate for points on the court's ground
        # plane - a detection well above it (very common now that the search
        # area deliberately reaches the top of frame) can convert a perfectly
        # ordinary pixel movement into a wildly inflated "real-world" distance
        # as the projection approaches the plane's vanishing line. Never
        # report a number that isn't physically possible - fall back to
        # pixels/sec for that reading instead.
        #
        # A real-world speed is only ever computed between two
        # actually-measured endpoints, never a coasted (interpolated) one -
        # that position came from the Kalman filter's own extrapolation, not
        # the detector, and compounding it through this already
        # noise-sensitive projection was producing physically-impossible
        # speed spikes in testing. Plain pixel-space speed doesn't have that
        # nonlinear blowup, so it's still used as the fallback either way.
        endpoints_measured = not point["interpolated"] and not window_start["interpolated"]
        court_speed = self._speed_court(window_start, point) if endpoints_measured else None
        real_units = court_speed is not None and court_speed <= MAX_PLAUSIBLE_REPORTED_SPEED_MS
        speed = court_speed if real_units else self._speed_px(window_start, point)

        return {
            "frame_idx": frame_idx,
            "confidence": confidence,
            "interpolated": interpolated,
            "pixel": pixel.tolist(),
            "court": list(court) if court is not None else None,
            "real_units": real_units,
            "speed_m_per_s": speed if real_units else None,
            "speed_px_per_s": speed if not real_units else None,
        }

    def _speed_court(self, prev, curr):
        """Real-world speed in m/s via the homography, or None if it can't be
        computed. Only trustworthy for points on the calibrated ground plane -
        see pixel_to_court()."""
        if prev["court"] is None or curr["court"] is None:
            return None

        dt = (curr["frame_idx"] - prev["frame_idx"]) / self.fps

        if dt <= 0:
            return None

        distance = float(np.linalg.norm(np.array(curr["court"]) - np.array(prev["court"])))

        return distance / dt

    def _speed_px(self, prev, curr):
        dt = (curr["frame_idx"] - prev["frame_idx"]) / self.fps

        if dt <= 0:
            return 0.0

        distance = float(np.linalg.norm(curr["pixel"] - prev["pixel"]))

        return distance / dt

    def trail(self):
        return [p["pixel"] for p in self.history]

    def court_trail(self):
        """Recent real-world (metres) positions, for the court minimap -
        only the points where a homography was available to project from."""
        return [p["court"] for p in self.history if p["court"] is not None]

    def lose(self):
        """Drop the track entirely - e.g. coasted past the miss limit, or was
        following a spot that turned out to be a newly-learned static zone.
        The next frame starts over with a full-frame acquisition scan."""
        self.kf = BallKalmanFilter(self.diagonal)
        self.coast_frames = 0
        self.history = []


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


def draw_search_box(frame, box):
    """Outline the current local zoom-search crop, so it's easy to see when
    the tracker is locked on and tightly following vs. about to fall back to
    a full-frame reacquisition scan."""

    if box is None:
        return

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    p1 = (int(np.clip(x1, 0, width - 1)), int(np.clip(y1, 0, height - 1)))
    p2 = (int(np.clip(x2, 0, width - 1)), int(np.clip(y2, 0, height - 1)))

    cv2.rectangle(frame, p1, p2, (255, 255, 0), 1)


def annotate_ball(frame, trail, detection, reading):

    for i in range(1, len(trail)):
        fade = i / len(trail)
        colour = (0, int(140 + 100 * fade), int(255 * fade))

        cv2.line(frame,
                 tuple(int(v) for v in trail[i - 1]),
                 tuple(int(v) for v in trail[i]),
                 colour, 2)

    if detection is None:
        return

    box, _ = detection
    x1, y1, x2, y2 = (int(v) for v in box)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    if reading is None:
        return

    if reading["real_units"]:
        label = f"{reading['speed_m_per_s'] * 3.6:.1f} km/h"
    else:
        label = f"{reading['speed_px_per_s']:.0f} px/s"

    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

    cv2.rectangle(frame, (x1, y1 - size[1] - 10), (x1 + size[0] + 6, y1),
                  (0, 200, 255), -1)

    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def detectBall(video_path, output_path, show_preview=False, save_video=False):
    """
    show_preview=False (default) skips the live cv2 preview window entirely -
    needed for running this somewhere with no display/GTK support (a
    headless server, some CI/automation contexts), where cv2.namedWindow
    itself raises. save_video=False (default) skips writing ball.mp4 - the
    JSON logs (ball_speed.json) are the real output once something else
    (see PostProcessed/renderVideo.py) renders a combined annotated video
    from all the detection stages' logs together; per-stage videos are no
    longer needed day to day. Pass either as True to get the old behaviour
    back for debugging a specific stage in isolation.
    """

    model = YOLO(MODEL_PATH)
    fallback_model = YOLO(FALLBACK_MODEL_PATH)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    video_file = str(Path(output_path) / OUTPUT_VIDEO_NAME)
    speed_log_file = Path(output_path) / SPEED_LOG_NAME

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            video_file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height)
        )

    need_annotation = save_video or show_preview

    device = 0 if torch.cuda.is_available() else "cpu"

    matrix = load_homography(output_path)

    if matrix is not None:
        print(f"Loaded court calibration from {Path(output_path) / COURT_FILE_NAME}; "
              f"ball speed will be reported in real-world units (km/h).")
    else:
        print("No court calibration found; ball speed will be reported in pixels/sec instead.")

    corners = load_court_corners(output_path)
    roi = build_ball_roi(corners, frame_width, frame_height) if corners is not None else None

    if roi is not None:
        print(f"Restricting ball detection to the court area (+{BALL_ROI_PADDING_FRACTION:.0%} padding, "
              f"full frame height above it).")
    else:
        print("No court boundary found; ball detection will search the full frame.")

    tiles = generate_tiles(frame_width, frame_height, ACQUISITION_TILE_GRID, ACQUISITION_TILE_OVERLAP_FRACTION)

    tracker = BallTracker(matrix, fps, frame_width, frame_height)
    static_zones = StaticZoneLearner(fps, frame_width, frame_height)
    color_profile = BallColorProfile()
    revalidation_interval_frames = max(1, round(REVALIDATION_INTERVAL_SECONDS * fps))
    frames_since_revalidation = 0

    window_name = "Ball Detection"
    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    speed_log = []
    frame_idx = 0

    target_frame_seconds = 1.0 / fps

    while True:

        frame_start = time.perf_counter()

        success, frame = cap.read()
        if not success:
            break

        predicted, std = tracker.predict()
        tracking_mode = predicted is not None and not tracker.is_lost()
        due_for_revalidation = tracking_mode and frames_since_revalidation >= revalidation_interval_frames
        search_box_for_drawing = None

        if tracking_mode and not due_for_revalidation:
            crop_box = tracker.zoom_crop(predicted, std)
            raw = detect_in_crop(model, frame, crop_box, ZOOM_IMG_SIZE, ZOOM_CONFIDENCE_THRESHOLD, device)
            search_box_for_drawing = crop_box
            frames_since_revalidation += 1
        else:
            if predicted is not None and not due_for_revalidation:
                # Coasted past the miss limit - drop the stale track before
                # rescanning, so matching below falls back to
                # highest-confidence rather than continuity with a position
                # that's no longer trustworthy.
                tracker.lose()
                predicted, std = None, None
            raw = detect_tiled(model, frame, tiles, ACQUISITION_IMG_SIZE, CONFIDENCE_THRESHOLD, device)
            frames_since_revalidation = 0

        candidates = [d for d in raw if box_diagonal(d[0]) >= MIN_BALL_DIAGONAL_PX]
        candidates = [d for d in candidates if box_in_roi(d[0], roi)]
        candidates = [d for d in candidates if not static_zones.is_excluded(centre_of(d[0]))]

        # Learning runs every frame, tracking mode included - not just during
        # acquisition. A bad initial lock onto a static false positive (a
        # wall fixture, a shadow) keeps re-matching within its own zoom crop
        # indefinitely otherwise: it never coasts (so never hits the miss
        # limit that would trigger a fresh full-frame scan), and a genuinely
        # moving ball never clusters into a static zone by construction, so
        # this can't mistakenly flag real play.
        newly_excluded = static_zones.observe(candidates, frame_idx)
        if newly_excluded and tracker.is_tracking:
            last_pixel = tracker.kf.position()
            if any(np.linalg.norm(last_pixel - c) <= static_zones.exclusion_radius for c in newly_excluded):
                # The track just turned out to be following a static false
                # positive, not the ball - drop it, and don't let matching
                # below use the now-abandoned prediction, or it would just
                # gate right back onto whatever's still sitting at that spot.
                tracker.lose()
                predicted, std = None, None

        if due_for_revalidation:
            matched = tracker.revalidate(candidates, predicted, std)
        else:
            matched = tracker.match(candidates, predicted, std)

        if matched is not None:
            # Only ever learn the ball's colour from the fine-tuned model's
            # own picks, never from the fallback's - a wrong fallback pick
            # would otherwise poison the very reference meant to catch it.
            color_profile.update(frame, matched[0])
        else:
            # The fine-tuned model found nothing usable in this frame's
            # search region - try the fallback detector on that same region
            # before giving up on the frame entirely. A generic "any sports
            # ball" classifier and a real volleyball detector tend to miss
            # different frames (different training data, different failure
            # modes on blur/scale), so this recovers some misses a single
            # model - however good - won't catch on its own. Only ever runs
            # on a miss, so it doesn't add cost to the common case.
            if tracking_mode and not due_for_revalidation:
                fallback_raw = detect_in_crop(fallback_model, frame, crop_box, ZOOM_IMG_SIZE,
                                               ZOOM_CONFIDENCE_THRESHOLD, device,
                                               class_id=FALLBACK_BALL_CLASS_ID)
            else:
                fallback_raw = detect_tiled(fallback_model, frame, tiles, ACQUISITION_IMG_SIZE,
                                             CONFIDENCE_THRESHOLD, device,
                                             class_id=FALLBACK_BALL_CLASS_ID)

            fallback_candidates = [d for d in fallback_raw if box_diagonal(d[0]) >= MIN_BALL_DIAGONAL_PX]
            fallback_candidates = [d for d in fallback_candidates if box_in_roi(d[0], roi)]
            fallback_candidates = [d for d in fallback_candidates
                                    if not static_zones.is_excluded(centre_of(d[0]))]

            # The fallback is a generic "any sports ball" classifier, so it
            # readily calls things that don't actually look like this video's
            # ball a match - veto anything that doesn't colour-match the
            # reference built from the fine-tuned model's own detections.
            # distance() returns None (no veto) until enough confident
            # detections have been seen to trust the reference yet.
            fallback_candidates = [
                d for d in fallback_candidates
                if (dist := color_profile.distance(frame, d[0])) is None or dist <= HIST_DISTANCE_THRESHOLD
            ]

            matched = tracker.match(fallback_candidates, predicted, std)

        reading = tracker.observe(matched, frame_idx)

        if need_annotation:
            annotate_ball(frame, tracker.trail(), matched, reading)
            draw_roi(frame, roi)
            draw_search_box(frame, search_box_for_drawing)
            draw_court_minimap(frame, tracker.court_trail())

        if reading is not None:
            speed_log.append(reading)

        if save_video:
            writer.write(frame)

        if show_preview:
            cv2.imshow(window_name, frame)

            # Detection is normally far faster than the source video's own
            # frame rate (especially once locked on and only searching a
            # small zoom crop), so waiting a fixed 1ms would play the
            # preview back much faster than it was actually recorded.
            # Wait out whatever's left of this frame's real-time budget
            # instead, so the preview runs at the source's actual pace -
            # falling back to the minimum delay if processing this frame
            # already took longer than that budget.
            elapsed = time.perf_counter() - frame_start
            remaining_ms = max(1, int((target_frame_seconds - elapsed) * 1000))

            if cv2.waitKey(remaining_ms) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if save_video:
        writer.release()
        print(f"Ball detection video saved: {video_file}")
    if show_preview:
        cv2.destroyAllWindows()

    with open(speed_log_file, "w") as f:
        json.dump(speed_log, f, indent=2)

    print(f"Ball speed log saved: {speed_log_file}")
