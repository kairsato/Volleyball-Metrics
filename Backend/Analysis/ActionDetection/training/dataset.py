"""Shared vocabulary + parsing helpers for the per-player action classifier's
training data. Two kinds of sources feed it (see
MachineLearning/datasetGather.py's ActionDatasets class):

- The Ibrahim et al. Volleyball dataset (https://github.com/mostafa-saad/deep-activity-rec),
  specifically its per-frame player tracking annotations (the
  `volleyball_tracking_annotation` archive) paired with that clip's frame
  images - parse_annotation_file/index_clips below know that format.
- Several Roboflow Universe bounding-box datasets, each with their own class
  vocabulary - handled directly in ActionDatasets since their format (YOLO
  images/+labels/+data.yaml) doesn't need clip/frame indexing like Ibrahim's
  does.

Both kinds ultimately produce the same thing: individual player crops
labeled with one of APP_ACTION_CLASSES, merged into one ImageFolder-style
directory by ActionDatasets.build_train_val_split for training/train.py to
consume.
"""

from pathlib import Path

# The Ibrahim et al. dataset's own 9-class per-player vocabulary (see the
# repo's README).
ACTION_LABELS = [
    "waiting", "setting", "digging", "falling", "spiking",
    "blocking", "jumping", "moving", "standing",
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(ACTION_LABELS)}

# How that vocabulary maps onto this app's own action_type values (see
# Backend/PostProcessing/consolidate.py's ACTION_TYPES). There's no "serve"
# label here - serve is a game-state/event thing, not a per-player pose, so
# it's detected separately by actionDetection.py's own timing rule. States
# like waiting/standing/moving/jumping/falling aren't ball-touch actions at
# all: actionDetection.py only asks "what action was this" on frames it
# already knows from ball trajectory were a touch, so a classifier trained
# on this vocabulary only ever needs to disambiguate among the four real
# touch types below at inference time.
TO_APP_ACTION_TYPE = {
    "setting": "set",
    "digging": "dig",
    "spiking": "spike",
    "blocking": "block",
}

# The classifier's actual output vocabulary - every training source (Ibrahim
# and each Roboflow source's own CLASS_MAPS in datasetGather.ActionDatasets)
# maps its own labels onto these 4 and drops everything else. Sorted alphabetically to
# match torchvision.datasets.ImageFolder's own class-to-index ordering
# (it always sorts folder names), so a checkpoint's saved `classes` list
# and a fresh ImageFolder built from the same directory layout never
# disagree on which index means what.
APP_ACTION_CLASSES = sorted(set(TO_APP_ACTION_TYPE.values()))


def parse_annotation_file(path: Path):
    """Yields one dict per line of a clip's <clip_id>.txt: {track_id, box
    [x1,y1,x2,y2], frame_id, lost, generated, label}. See the tracking
    annotation archive's own README for the exact column meanings - lost
    means the box is outside the frame (not a usable sample), generated
    means it was interpolated rather than hand-annotated."""
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        track_id, x1, y1, x2, y2, frame_id, lost, _grouping, generated = parts[:9]
        label = " ".join(parts[9:]).strip()
        yield {
            "track_id": int(track_id),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "frame_id": int(frame_id),
            "lost": bool(int(lost)),
            "generated": bool(int(generated)),
            "label": label,
        }


def index_clips(root: Path):
    """Walks root/<video_id>/<clip_id>/ and returns (video_id, clip_id,
    clip_dir, annotation_file) for every clip directory that actually has
    an annotation .txt file."""
    clips = []
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        for clip_dir in sorted(video_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            ann_file = clip_dir / f"{clip_dir.name}.txt"
            if ann_file.exists():
                clips.append((video_dir.name, clip_dir.name, clip_dir, ann_file))
    return clips
