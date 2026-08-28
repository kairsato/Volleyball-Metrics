"""
Shared detection-fusion and appearance utilities for player tracking.

The actual tracking entry point is tracker_offline.trackplayers_offline() -
this module holds what it (and nothing else, currently) depends on: running
the 5-tracker ensemble and fusing their agreeing boxes into one, and the
appearance-embedding machinery used to tell visually similar players apart.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

MODEL_PATH = "yolo26x.pt"

# name -> tracker config passed to model.track
TRACKERS = {
    "oc": "ocsort.yaml",
    "deep": str(Path(__file__).parent / "custom_deepocsort.yaml"),
    #"deep": "deepocsort.yaml",
    "tt": "tracktrack.yaml",
    "ft":"fasttrack.yaml",
    "botSort":"botsort.yaml"
}

# Minimum overlap for two trackers to be considered the same player. Kept low
# rather than the more typical ~0.5 because a packed formation (e.g. a team's
# six players converging on a rally) means different trackers often box the
# same partially-occluded player at slightly different sizes/offsets - a
# stricter threshold made those legitimate players fail to reach MIN_AGREEMENT
# and vanish from that frame entirely.
IOU_MATCH_THRESH = 0.10

# How many trackers must agree (the anchor box counts as one) before a box is
# kept. A box only one tracker sees is much more likely to be jitter/noise, and
# feeding that straight into the identity bank is a common source of identity
# swaps, so by default at least one other tracker has to corroborate it.
MIN_AGREEMENT = 2

# Largest gap (in frames) tracker_offline.py's consolidation will bridge
# between two fragments of the same person - lives here because it's a
# property of the detection/tracking cadence (how long a real absence can
# plausibly be), not of any one consumer.
IDENTITY_MAX_AGE = 1800  # 60 sec @ 30fps

# Weight given to appearance vs. position when two fragments are being
# considered for the same identity.
APPEARANCE_WEIGHT = 0.85

# One embedding set per body shape, judged from the box width / height. Crouching,
# diving and upright players look different enough that mixing them hurts matching.
ASPECT_EDGES = (0.45, 0.70, 1.00)
CROSS_SHAPE_PENALTY = 0.2   # cost added when only another shape is stored

# Appearance embeddings: ResNet18 input preprocessing (ImageNet normalisation).
# 128x256 keeps roughly a player's real body proportions instead of squashing
# them into a square, which otherwise distorts the embedding every frame.
CNN_INPUT_WIDTH = 128
CNN_INPUT_HEIGHT = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# How many recent embeddings each identity keeps per body-shape bucket.
# Matching against the nearest of several real snapshots discriminates
# between similar-looking players far better than blending them into one
# running average, which tends to blur different players' looks together.
APPEARANCE_GALLERY_SIZE = 10

# Name of the calibration file CourtDefinition.court saves into output_path
COURT_FILE_NAME = "court.json"


def load_homography(output_path):
    """
    Load the pixel -> real-world-court-metres transform CourtDefinition.court
    saved for this video. Returns None if no calibration has been saved yet -
    player positions are then logged in pixel coordinates only.
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
              f"player positions will be logged in pixels only.")
        return None

    return matrix


def pixel_to_court(x, y, matrix):
    point = np.array([[[x, y]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(point, matrix)

    return float(transformed[0][0][0]), float(transformed[0][0][1])


def calculate_iou(boxA, boxB):
    # Standard coordinate extraction: [x1, y1, x2, y2]
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


def load_court_polygon(output_path):
    """
    Load the four court corners CourtDefinition.court saved for this video, as
    a polygon in frame pixel coordinates. Used only to draw the court boundary
    overlay on the annotated video - everyone in frame gets tracked regardless
    of position, since a player diving or stepping outside the calibrated
    boundary shouldn't lose tracking, and any spurious non-player identity
    (staff, ball kids) can be marked ignored in the player review step instead.

    Returns None (no overlay drawn) if no calibration has been saved yet.
    """
    court_file = Path(output_path) / COURT_FILE_NAME

    if not court_file.exists():
        return None

    try:
        with open(court_file) as f:
            data = json.load(f)

        image_points = [(p["x"], p["y"]) for p in data["image_points"]]

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(f"Could not read court boundary from {court_file} ({e}); "
              f"tracking the full frame instead.")
        return None

    if len(image_points) != 4:
        return None

    return np.array(image_points, dtype=np.float32)


def fuse(tracker_outputs):
    """Cluster boxes that several trackers agree on and average them into one."""

    entries = []

    for name, boxes, ids in tracker_outputs:
        for box, tid in zip(boxes, ids):
            entries.append({
                "name": name,
                "box": np.asarray(box, dtype=float),
                "id": int(tid)
            })

    used = [False] * len(entries)
    fused = []

    for i, anchor in enumerate(entries):

        if used[i]:
            continue

        # Best overlapping box from every other tracker
        best = {}

        for j, other in enumerate(entries):

            if used[j] or j == i or other["name"] == anchor["name"]:
                continue

            iou = calculate_iou(anchor["box"], other["box"])

            if iou <= IOU_MATCH_THRESH:
                continue

            if other["name"] not in best or iou > best[other["name"]][1]:
                best[other["name"]] = (j, iou)

        if len(best) + 1 < MIN_AGREEMENT:
            continue

        used[i] = True

        indices = [i]
        ious = []

        for j, iou in best.values():
            used[j] = True
            indices.append(j)
            ious.append(iou)

        merged = np.mean([entries[k]["box"] for k in indices], axis=0)
        ids = {entries[k]["name"]: entries[k]["id"] for k in indices}

        fused.append((merged, ids, float(np.mean(ious))))

    return fused


def aspect_ratio(box):

    x1, y1, x2, y2 = box

    return float(x2 - x1) / max(1.0, float(y2 - y1))


def shape_bin(ratio):
    """Upright, crouched and diving players get their own embedding."""

    return sum(ratio >= edge for edge in ASPECT_EDGES)


def bin_penalty(stored_bin, current_bin):
    """Cost of comparing embeddings taken at different body shapes."""

    steps = abs(stored_bin - current_bin)

    return CROSS_SHAPE_PENALTY * steps / len(ASPECT_EDGES)


def centre_of(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def cosine_distance(a, b):
    """
    Distance in [0, 1] between two L2-normalised embeddings (0 = identical).

    Clamped with plain min/max rather than np.clip - np.clip is built for
    array inputs and its dispatch overhead (type checks, ufunc machinery)
    dwarfs the cost of clamping a single float. Profiling an offline
    consolidation run (tens of millions of calls, comparing every candidate
    pair's embedding galleries) found this one substitution was ~92% of that
    run's total time - np.clip on a scalar, not any algorithmic cost.
    """

    if a is None or b is None:
        return 1.0

    similarity = max(-1.0, min(1.0, float(np.dot(a, b))))

    return max(0.0, min(1.0, (1.0 - similarity) / 2.0))


def extract_crop(frame, box, exclude_boxes=None):
    """
    Crop a player's box, muting any pixels claimed by a neighbouring detection
    so a crowded frame doesn't bleed one player's appearance into another.
    Muted pixels are replaced with the patch's own average color rather than
    black: a solid black block creates a sharp artificial edge the CNN
    responds strongly to, which made occluded players look more like each
    other (they'd share the same black-block feature) instead of less. Trims
    a thin sliver off the top/bottom to cut down on ball motion blur and
    floor bleed.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_h = max(1.0, y2 - y1)

    top = int(max(0, min(height - 1, y1 + box_h * 0.03)))
    bottom = int(max(0, min(height, y2 - box_h * 0.02)))
    left = int(max(0, min(width - 1, x1)))
    right = int(max(0, min(width, x2)))

    if bottom - top < 8 or right - left < 8:
        return None

    patch = frame[top:bottom, left:right].copy()

    exclude_mask = np.zeros(patch.shape[:2], dtype=bool)

    for ex_box in (exclude_boxes or []):
        ex_x1, ex_y1, ex_x2, ex_y2 = ex_box

        ox1 = int(max(left, ex_x1)) - left
        oy1 = int(max(top, ex_y1)) - top
        ox2 = int(min(right, ex_x2)) - left
        oy2 = int(min(bottom, ex_y2)) - top

        if ox2 > ox1 and oy2 > oy1:
            exclude_mask[oy1:oy2, ox1:ox2] = True

    if exclude_mask.any() and not exclude_mask.all():
        fill_colour = patch[~exclude_mask].mean(axis=0)
        patch[exclude_mask] = fill_colour

    return patch


class AppearanceEncoder:
    """
    Batched CNN feature extractor used to tell visually similar players apart.

    Plain color histograms can't distinguish teammates wearing identical kits,
    which is the biggest source of identity swaps in this pipeline. An
    ImageNet-pretrained ResNet18 (used purely as a fixed feature extractor, not
    fine-tuned) captures texture, build and pose in addition to color, so it is
    far more discriminating between two players in the same uniform.
    """

    def __init__(self, device):
        self.device = device

        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception as e:
            print(f"Could not download pretrained ResNet18 weights ({e}); "
                  f"falling back to randomly initialised weights.")
            model = resnet18(weights=None)

        model.fc = torch.nn.Identity()
        model.eval()

        self.model = model.to(device)

    def _prepare(self, crop):
        # A distant player's crop is usually far smaller than the CNN input, so
        # this is normally an upsample - INTER_CUBIC preserves more of what
        # little detail is there than INTER_LINEAR's flatter blur.
        interpolation = (cv2.INTER_CUBIC if crop.shape[0] < CNN_INPUT_HEIGHT
                         or crop.shape[1] < CNN_INPUT_WIDTH else cv2.INTER_AREA)

        resized = cv2.resize(crop, (CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        return np.transpose(normed, (2, 0, 1))

    @torch.no_grad()
    def encode(self, crops):
        """Batch-encode a list of BGR crops (any entry may be None). Returns a
        list of L2-normalised embeddings aligned with the input, with None
        wherever the crop was missing or too small to embed."""

        valid = [i for i, c in enumerate(crops) if c is not None and c.size > 0]

        results = [None] * len(crops)

        if not valid:
            return results

        batch = np.stack([self._prepare(crops[i]) for i in valid])
        tensor = torch.from_numpy(batch).to(self.device)

        features = self.model(tensor)
        features = F.normalize(features, dim=1)
        features = features.cpu().numpy()

        for slot, i in enumerate(valid):
            results[i] = features[slot]

        return results


def identity_colour(stable_id):

    rng = np.random.default_rng(stable_id * 9781)

    return tuple(int(c) for c in rng.integers(60, 255, size=3))
