"""PyTorch dataset for the Ibrahim et al. Volleyball dataset
(https://github.com/mostafa-saad/deep-activity-rec), specifically its
per-frame player tracking annotations (the `volleyball_tracking_annotation`
archive) paired with that clip's frame images. One sample = one labeled,
non-lost player box, cropped from its frame.

Directory layout expected on disk, matching how the dataset actually ships:
    <annotations_root>/<video_id>/<clip_id>/<clip_id>.txt
    <images_root>/<video_id>/<clip_id>/<frame_id>.jpg

annotations_root and images_root are separate archives from the same
dataset (see the repo's README) and don't need to cover the same set of
clips - only clips present in both contribute samples, so this also works
against a partial image set (e.g. a small sample subset) without any
special-casing.
"""

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

# The dataset's own 9-class per-player vocabulary (see the repo's README).
ACTION_LABELS = [
    "waiting", "setting", "digging", "falling", "spiking",
    "blocking", "jumping", "moving", "standing",
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(ACTION_LABELS)}

# How that vocabulary maps onto this app's own action_type values (see
# Backend/PostProcessed/consolidate.py's ACTION_TYPES). There's no "serve"
# label here - serve is a game-state/event thing, not a per-player pose, so
# it's handled separately. States like waiting/standing/moving/jumping/
# falling aren't ball-touch actions at all: actionDetection.py only asks
# "what action was this" on frames it already knows from ball trajectory
# were a touch, so a classifier trained on this vocabulary only ever needs
# to disambiguate among the four real touch types below at inference time.
TO_APP_ACTION_TYPE = {
    "setting": "set",
    "digging": "dig",
    "spiking": "spike",
    "blocking": "block",
}


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


class PlayerActionCrops(Dataset):
    def __init__(self, annotations_root, images_root, transform=None, skip_generated=True):
        self.transform = transform
        self.samples = []

        annotations_root = Path(annotations_root)
        images_root = Path(images_root)

        for video_id, _clip_id, _ann_clip_dir, ann_file in index_clips(annotations_root):
            image_dir = images_root / video_id / ann_file.parent.name
            if not image_dir.exists():
                continue

            for entry in parse_annotation_file(ann_file):
                if entry["lost"]:
                    continue
                if skip_generated and entry["generated"]:
                    continue
                if entry["label"] not in LABEL_TO_INDEX:
                    continue

                image_path = image_dir / f"{entry['frame_id']}.jpg"
                if not image_path.exists():
                    continue

                self.samples.append({**entry, "image_path": image_path})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(str(sample["image_path"]))
        if image is None:
            raise FileNotFoundError(sample["image_path"])

        h, w = image.shape[:2]
        x1, y1, x2, y2 = sample["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            crop = np.zeros((8, 8, 3), dtype=np.uint8)

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self.transform:
            crop = self.transform(crop)

        return crop, LABEL_TO_INDEX[sample["label"]]
