"""Gathers every model's training data - one namespace class per model,
each owning its own dataset_<model>/ folder (a sibling of this file) and
knowing how to download/prepare/merge its own sources. mainTrainingModels.py
is the only caller: one `<Model>Datasets.download_*`/`extract_*`/`prepare_*`
call per source, then one `merge_*`/`build_*` call to combine them into a
training-ready dataset, per model, wired into train_all() there.

The three classes below are independent (no shared state, no inheritance) -
grouped into one file because they're conceptually the same kind of thing
(this project's per-model data-gathering logic) and all lean on the same
Roboflow-download plumbing in machineLearningCommon.py. Add a further
model's data-gathering logic here the same way: a new `<Model>Datasets`
class owning `MachineLearning/dataset_<model>/`.
"""

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import yaml

ML_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = ML_DIR.parent
BACKEND_DIR = ANALYSIS_DIR.parent
for _directory in (ANALYSIS_DIR, BACKEND_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from machineLearningCommon import download_roboflow_project  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class BallDatasets:
    """Ball-detector training data sources. Every download_*/prepare_*
    method returns the local Path to a YOLO-format dataset folder (an
    `images/` + sibling `labels/` layout, using Roboflow's own train/valid/
    test split for downloaded sources) for one data source - either
    downloaded from Roboflow Universe, a manually-obtained Kaggle export,
    or generated straight from this project's own already-processed
    videos. merge_all combines however many of these a caller gathers into
    one training-ready dataset, used by BallDetection/ballDetection.py
    (MODEL_PATH/SECONDARY_MODEL_PATH) once fine-tuned.

    Every public (non-own-footage) source below is a THIRD PARTY dataset
    the caller needs their own Roboflow/Kaggle account and API key for -
    nothing is bundled with this repo, and nothing downloads without that
    key. Attribution/license notes are on each method; verify them before
    using a model fine-tuned from these commercially.
    """

    DATASETS_DIR = ML_DIR / "dataset_ballDetection"
    BALL_CLASS_ID = 0
    BALL_CLASS_NAMES = {"ball", "volleyball", "volley ball", "the ball"}

    @staticmethod
    def download_primaryws(api_key: str) -> Path:
        """548 ball-only images (RGB + grayscale variants). CC BY 4.0 -
        https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset"""
        return download_roboflow_project(
            api_key, "primaryws", "volleyball_ball_object_detection_dataset", BallDatasets.DATASETS_DIR / "primaryws",
        )

    @staticmethod
    def download_aivolleyballref(api_key: str) -> Path:
        """1091 images, multi-class (players/ball/borders/net) - merge_all
        below keeps only the ball annotations, dropping the rest. CC BY 4.0 -
        https://universe.roboflow.com/aivolleyballref/volleyball_detection"""
        return download_roboflow_project(
            api_key, "aivolleyballref", "volleyball_detection", BallDatasets.DATASETS_DIR / "aivolleyballref",
        )

    @staticmethod
    def download_volleyballtest(api_key: str) -> Path:
        """3070 ball-only images. CC BY 4.0 -
        https://universe.roboflow.com/volleyballtest/volleyball-fdqxb"""
        return download_roboflow_project(
            api_key, "volleyballtest", "volleyball-fdqxb", BallDatasets.DATASETS_DIR / "volleyballtest",
        )

    @staticmethod
    def download_salo_levy(api_key: str) -> Path:
        """324 ball-only images. CC BY 4.0 -
        https://universe.roboflow.com/salo-levy-nlqrn/volley-ball-detection"""
        return download_roboflow_project(
            api_key, "salo-levy-nlqrn", "volley-ball-detection", BallDatasets.DATASETS_DIR / "salo_levy",
        )

    @staticmethod
    def prepare_kaggle_dataset(zip_path: Path) -> Path:
        """Kaggle's API needs its own separate credential setup this module
        doesn't attempt to automate - download
        https://www.kaggle.com/datasets/pythonistasamurai/volleyball-ball-object-detection-dataset
        by hand and pass the downloaded zip here. License unconfirmed at
        time of writing (unlike the CC BY 4.0 Roboflow sources above) -
        verify it before using a model trained on this commercially.
        Unpacked contents are expected to already be in (or be converted
        to, by hand, before calling merge_all) a YOLO images/+labels/
        layout."""
        dest = BallDatasets.DATASETS_DIR / "kaggle_pythonistasamurai"
        if dest.exists():
            print(f"[kaggle_pythonistasamurai] already prepared at {dest}, skipping.")
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(zip_path), str(dest))
        return dest

    @staticmethod
    def pseudo_label_own_footage(
        job_ids: Optional[list] = None,
        min_confidence: float = 0.6,
        stride: int = 15,
        max_per_job: int = 300,
    ) -> Path:
        """Turns this project's own already-processed videos into training
        data with zero manual annotation, using the pipeline's own confident
        ball detections as labels (pseudo-labeling/self-training) - directly
        targets what ballDetection.py's own module docstring names as the
        top next step: the current model has never seen this project's
        actual footage/venues.

        For each job (every already-complete job if job_ids is None), reads
        that job's TRAJECTORY_LOG_NAME (ball_trajectory.json) - the globally
        reconstructed trajectory ballDetection.solve_ball_trajectory
        produced, not the raw per-model candidates - and keeps only "real"
        frames (an actual detection Phase 2 kept, not an interpolated
        guess) at or above min_confidence. This is a better source than
        picking straight from raw candidates would be: a "real" frame
        already benefits from Phase 2's own cross-model-agreement and
        trajectory-consistency filtering, so it's less likely to be a
        confident false positive than a single detector's raw output taken
        in isolation. Every `stride`th eligible frame is kept, for variety
        across a video rather than dense near-duplicate frames, capped at
        max_per_job per video.
        """
        from API import config as api_config
        from API.jobs import STATUS_COMPLETE, store
        from BallDetection.ballDetection import TRAJECTORY_LOG_NAME

        dest = BallDatasets.DATASETS_DIR / "own_footage"
        images_dir = dest / "images"
        labels_dir = dest / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        if job_ids is None:
            job_ids = [job.id for job in store.list() if job.status == STATUS_COMPLETE]

        total_written = 0
        for job_id in job_ids:
            trajectory_file = api_config.output_dir(job_id) / TRAJECTORY_LOG_NAME
            video_path = api_config.find_input_video(job_id)
            if not trajectory_file.exists() or video_path is None:
                print(f"[{job_id}] missing ball_trajectory.json or source video, skipping.")
                continue

            log = json.loads(trajectory_file.read_text())
            frame_width, frame_height = log["frame_w"], log["frame_h"]

            eligible = [
                (frame_idx, entry["box"])
                for frame_idx, entry in enumerate(log["frames"])
                if entry["status"] == "real" and entry["conf"] >= min_confidence
            ]

            sampled = eligible[::stride][:max_per_job]
            if not sampled:
                print(f"[{job_id}] no eligible high-confidence real-detection frames.")
                continue

            cap = cv2.VideoCapture(str(video_path))
            try:
                for frame_idx, box in sampled:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = cap.read()
                    if not success:
                        continue

                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / frame_width
                    cy = ((y1 + y2) / 2) / frame_height
                    w = (x2 - x1) / frame_width
                    h = (y2 - y1) / frame_height

                    stem = f"{job_id}_{frame_idx}"
                    cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)
                    (labels_dir / f"{stem}.txt").write_text(
                        f"{BallDatasets.BALL_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    )
                    total_written += 1
            finally:
                cap.release()

        print(f"Pseudo-labeled {total_written} frame(s) from {len(job_ids)} job(s) into {dest}")
        return dest

    @staticmethod
    def _find_images(source: Path) -> list:
        return [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.parent.name == "images"]

    @staticmethod
    def _matching_label(image_path: Path) -> Path:
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

    @staticmethod
    def _ball_class_index(source: Path) -> Optional[int]:
        """Reads a downloaded/prepared dataset's own data.yaml to find which
        of ITS class indices is actually the ball - a multi-class source
        like aivolleyballref doesn't necessarily agree with this project's
        own "class 0 is the ball" convention. A single-class source with no
        data.yaml at all (e.g. own_footage, which this class writes as
        class 0 itself) is assumed to already be ball-only."""
        data_yaml = source / "data.yaml"
        if not data_yaml.exists():
            return BallDatasets.BALL_CLASS_ID

        names = yaml.safe_load(data_yaml.read_text()).get("names", [])
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names, key=int)]
        for index, name in enumerate(names):
            if str(name).strip().lower() in BallDatasets.BALL_CLASS_NAMES:
                return index
        return BallDatasets.BALL_CLASS_ID if len(names) == 1 else None

    @staticmethod
    def _remap_label_lines(label_path: Path, ball_index: int) -> Optional[str]:
        """Keeps only this label file's ball-class lines (see
        _ball_class_index), remapped to BALL_CLASS_ID and dropping every
        other class a multi-class source might also list for the same
        image."""
        lines_out = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if int(parts[0]) != ball_index:
                continue
            lines_out.append(" ".join([str(BallDatasets.BALL_CLASS_ID), *parts[1:]]))
        return "\n".join(lines_out) + "\n" if lines_out else None

    @staticmethod
    def merge_all(sources: list, out_name: str = "merged", val_fraction: float = 0.1) -> Path:
        """Combines every dataset folder gathered above into one YOLO
        dataset ready for training - remaps each source's own ball class to
        BALL_CLASS_ID (dropping any other class, and any image that turns
        out to have no ball annotation left after that filter), prefixes
        filenames by source folder name to avoid collisions between
        sources, and writes one shuffled train/val split + data.yaml at
        DATASETS_DIR/out_name. Always rebuilds that folder from scratch
        rather than appending, so re-running after adding a new source
        never leaves stale files behind from a previous merge.

        Returns the path to the written data.yaml, ready to pass straight
        to Ultralytics' `YOLO(...).train(data=...)`.
        """
        out_dir = BallDatasets.DATASETS_DIR / out_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        train_images, train_labels = out_dir / "train" / "images", out_dir / "train" / "labels"
        val_images, val_labels = out_dir / "val" / "images", out_dir / "val" / "labels"
        for directory in (train_images, train_labels, val_images, val_labels):
            directory.mkdir(parents=True, exist_ok=True)

        pairs = []  # (image_path, label_path, ball_index, prefix)
        for source in sources:
            ball_index = BallDatasets._ball_class_index(source)
            if ball_index is None:
                print(f"[{source.name}] couldn't identify a 'ball' class in its data.yaml - skipping this source.")
                continue
            for image_path in BallDatasets._find_images(source):
                label_path = BallDatasets._matching_label(image_path)
                if label_path.exists():
                    pairs.append((image_path, label_path, ball_index, source.name))

        if not pairs:
            raise RuntimeError("No image/label pairs found across the given sources - nothing to merge.")

        # Shuffled (not grouped by source) so the val split isn't
        # accidentally all-one-dataset - seeded for a reproducible split
        # across re-runs.
        random.Random(0).shuffle(pairs)
        split_index = max(1, int(len(pairs) * (1 - val_fraction)))

        written = 0
        for i, (image_path, label_path, ball_index, prefix) in enumerate(pairs):
            images_dir, labels_dir = (train_images, train_labels) if i < split_index else (val_images, val_labels)
            remapped = BallDatasets._remap_label_lines(label_path, ball_index)
            if remapped is None:
                continue
            stem = f"{prefix}_{image_path.stem}"
            shutil.copy2(image_path, images_dir / f"{stem}{image_path.suffix}")
            (labels_dir / f"{stem}.txt").write_text(remapped)
            written += 1

        data_yaml = out_dir / "data.yaml"
        data_yaml.write_text("train: train/images\nval: val/images\nnc: 1\nnames: ['ball']\n")
        print(f"Merged {written} labeled image(s) from {len(sources)} source(s) into {out_dir}")
        return data_yaml


class ActionDatasets:
    """Action-classifier training data sources, used by
    ActionDetection/actionDetection.py's classifier once trained.

    Every method below produces labeled player-action CROPS (not the raw
    downloaded dataset) into DATASETS_DIR/raw_crops/<source_name>/<app_class>/ -
    one JPEG per labeled box, already cut out of its source image/frame.
    This differs from BallDatasets (which merges YOLO detection datasets
    directly): the action classifier is a per-crop image classifier, so
    each source's own bounding boxes get cropped out and sorted into this
    app's 4-class vocabulary (see ActionDetection/training/dataset.py's
    APP_ACTION_CLASSES) right away, and build_train_val_split at the
    bottom turns the accumulated crops into one ImageFolder-ready
    train/val directory for ActionDetection/training/train.py's
    train_from_folder().

    Sources and licensing (verified before including - "unclear" or
    conflicting-license candidates found during research were deliberately
    left out, see below):
        - mikhail-klyukin/volleyball_dataset          CC BY 4.0   ~1.2k images
        - vbanalyzer/volleyball-action-recognition    CC BY 4.0   ~30k images
        - activity-graz-uni/volleyball-activity       CC BY 4.0   ~25k images (Roboflow
          re-upload of TU Graz ICG match footage - the Roboflow export itself is
          CC BY 4.0, but the original academic dataset's own license terms
          weren't independently located, so treat this source as lower-confidence
          if redistributing crops derived from it)
        - vballactionrecognition/...-7rnpb            CC BY 4.0   ~1.3k images
        - Ibrahim et al. Volleyball Dataset (CVPR'16) citation-required, no
          explicit commercial grant - see extract_ibrahim_crops(). Optional:
          needs a manual Google Drive download, so it's only included when
          IBRAHIM_ANNOTATIONS/IBRAHIM_IMAGES are actually supplied.

    Deliberately EXCLUDED despite showing up in the same search:
        - actions-players/volleyball-actions: license badge conflicts with a
          third-party report of CC BY-NC-ND 4.0 (no-derivatives) on the same
          dataset - cropping images for training would itself be a derivative,
          so this is skipped rather than guessed at.
        - volleyball-yzp18/better-vbml: no license badge found at all.
        - HAA500: only a code-level MIT LICENSE found; the curated web-sourced
          video content's own rights are unconfirmed, and volleyball-specific
          class names couldn't even be confirmed to exist in its taxonomy.
        - Kaggle's omarmohamedelgharib/resnet50-image-level-volleyball-dataset:
          ships pre-extracted ResNet-50 feature vectors, not images - unusable
          for a crop-based classifier.

    Every Roboflow source below is a THIRD PARTY dataset - nothing is
    bundled with this repo, and nothing downloads without your own
    Roboflow account API key (https://app.roboflow.com/settings/api).
    """

    DATASETS_DIR = ML_DIR / "dataset_actionDetection"
    RAW_CROPS_DIR = DATASETS_DIR / "raw_crops"

    # Each Roboflow source's own class vocabulary (lowercased) -> one of
    # APP_ACTION_CLASSES. A source's classes not listed here (e.g. "serving",
    # "a_serve", "ball", "net", "ref-start") are simply dropped - serve is
    # handled by actionDetection.py's own timing rule, not this classifier
    # (see training/dataset.py), and non-action classes obviously don't
    # belong here.
    CLASS_MAPS = {
        "mikhail_klyukin": {
            "blocking": "block", "digging": "dig", "passing": "dig",
            "setting": "set", "spiking": "spike",
        },
        "vbanalyzer": {
            "block": "block", "receive": "dig", "set": "set", "hit": "spike",
        },
        "graz_uni": {
            "block": "block", "reception": "dig", "setting": "set", "attack": "spike",
        },
        "vballactionrecognition": {
            "block": "block", "bump": "dig", "set": "set", "spike": "spike",
        },
    }

    @staticmethod
    def download_mikhail_klyukin(api_key: str) -> Path:
        """~1.2k images: blocking, digging, passing, serving, setting, spiking.
        CC BY 4.0 - https://universe.roboflow.com/mikhail-klyukin/volleyball_dataset"""
        return download_roboflow_project(
            api_key, "mikhail-klyukin", "volleyball_dataset", ActionDatasets.DATASETS_DIR / "mikhail_klyukin",
        )

    @staticmethod
    def download_vbanalyzer(api_key: str) -> Path:
        """~30k images: block, a_serve, b_serve, hit, receive, ref-start, set.
        CC BY 4.0 - https://universe.roboflow.com/vbanalyzer/volleyball-action-recognition-k6tqv"""
        return download_roboflow_project(
            api_key, "vbanalyzer", "volleyball-action-recognition-k6tqv", ActionDatasets.DATASETS_DIR / "vbanalyzer",
        )

    @staticmethod
    def download_graz_uni(api_key: str) -> Path:
        """~25k images from real Austrian Volley League match footage (TU Graz
        ICG): serve, reception, setting, attack, block, stand, defense/move.
        Roboflow re-upload licensed CC BY 4.0 - see the class docstring's
        licensing note on this one. https://universe.roboflow.com/activity-graz-uni/volleyball-activity-dataset"""
        return download_roboflow_project(
            api_key, "activity-graz-uni", "volleyball-activity-dataset", ActionDatasets.DATASETS_DIR / "graz_uni",
        )

    @staticmethod
    def download_vballactionrecognition(api_key: str) -> Path:
        """~1.3k images: block, bump, set, spike. CC BY 4.0 -
        https://universe.roboflow.com/vballactionrecognition/volleyball-action-recognition-7rnpb"""
        return download_roboflow_project(
            api_key, "vballactionrecognition", "volleyball-action-recognition-7rnpb",
            ActionDatasets.DATASETS_DIR / "vballactionrecognition",
        )

    @staticmethod
    def _write_crop(image, box, out_path: Path) -> bool:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image[y1:y2, x1:x2])
        return True

    @staticmethod
    def extract_bbox_dataset_crops(source_dir: Path, class_map: dict, source_name: str) -> Path:
        """Walks a downloaded Roboflow YOLO-format export (train/valid/test,
        each with images/+labels/, plus one data.yaml at the root listing
        class names) and crops out every box whose class maps onto one of
        APP_ACTION_CLASSES via class_map. Rebuilds RAW_CROPS_DIR/<source_name>/
        from scratch each call, so re-running this after a class_map edit
        never leaves stale crops from the old mapping behind."""
        dest = ActionDatasets.RAW_CROPS_DIR / source_name
        if dest.exists():
            shutil.rmtree(dest)

        data_yaml = source_dir / "data.yaml"
        names = yaml.safe_load(data_yaml.read_text()).get("names", []) if data_yaml.exists() else []
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names, key=int)]

        written = 0
        for split_dir in source_dir.iterdir():
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            if not images_dir.is_dir():
                continue

            for image_path in images_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue

                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                h, w = image.shape[:2]

                for i, line in enumerate(label_path.read_text().splitlines()):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    class_idx = int(parts[0])
                    if class_idx >= len(names):
                        continue
                    app_class = class_map.get(str(names[class_idx]).strip().lower())
                    if app_class is None:
                        continue

                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                    if ActionDatasets._write_crop(image, box, dest / app_class / f"{image_path.stem}_{i}.jpg"):
                        written += 1

        print(f"[{source_name}] extracted {written} labeled crop(s) into {dest}")
        return dest

    @staticmethod
    def extract_ibrahim_crops(annotations_root, images_root, source_name: str = "ibrahim") -> Path:
        """The Ibrahim et al. Volleyball dataset's per-player tracking
        annotations, paired with that clip's frame images (see
        ActionDetection/training/dataset.py's module docstring for the
        exact directory layout and why the two archives don't need to
        cover the same clips). "lost" boxes (outside the frame) and
        "generated" ones (interpolated, not hand-annotated) are skipped -
        only real, visible, hand-labeled boxes are trustworthy enough to
        train on."""
        from ActionDetection.training.dataset import index_clips, parse_annotation_file, TO_APP_ACTION_TYPE

        dest = ActionDatasets.RAW_CROPS_DIR / source_name
        if dest.exists():
            shutil.rmtree(dest)

        annotations_root, images_root = Path(annotations_root), Path(images_root)

        written = 0
        for video_id, _clip_id, _clip_dir, ann_file in index_clips(annotations_root):
            image_dir = images_root / video_id / ann_file.parent.name
            if not image_dir.exists():
                continue

            for entry in parse_annotation_file(ann_file):
                if entry["lost"] or entry["generated"]:
                    continue
                app_class = TO_APP_ACTION_TYPE.get(entry["label"])
                if app_class is None:
                    continue

                image_path = image_dir / f"{entry['frame_id']}.jpg"
                if not image_path.exists():
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                out_path = dest / app_class / f"{video_id}_{ann_file.parent.name}_{entry['frame_id']}_{entry['track_id']}.jpg"
                if ActionDatasets._write_crop(image, entry["box"], out_path):
                    written += 1

        print(f"[{source_name}] extracted {written} labeled crop(s) into {dest}")
        return dest

    @staticmethod
    def build_train_val_split(crop_sources: list, out_name: str = "merged", val_fraction: float = 0.15) -> Path:
        """Combines every source's crops (already sorted into
        RAW_CROPS_DIR/<source>/<app_class>/ by the extract_* methods above)
        into one ImageFolder-ready DATASETS_DIR/<out_name>/{train,val}/<class>/
        directory. Shuffles each class's files with a fixed seed before
        splitting, so the val split is a random sample across sources (not
        accidentally all-one-source) but still reproducible across
        re-runs. Always rebuilds out_name from scratch, matching
        BallDatasets.merge_all's own "never leave stale files" behaviour.
        """
        from ActionDetection.training.dataset import APP_ACTION_CLASSES

        out_dir = ActionDatasets.DATASETS_DIR / out_name
        if out_dir.exists():
            shutil.rmtree(out_dir)

        by_class = {cls: [] for cls in APP_ACTION_CLASSES}
        for source_dir in crop_sources:
            for cls in APP_ACTION_CLASSES:
                class_dir = source_dir / cls
                if class_dir.exists():
                    by_class[cls].extend(sorted(class_dir.iterdir()))

        total_written = 0
        for cls, files in by_class.items():
            if not files:
                print(f"[{cls}] no crops found across any source - this class will be missing from {out_dir}.")
                continue

            rng = random.Random(0)
            rng.shuffle(files)
            split_index = max(1, int(len(files) * (1 - val_fraction)))

            for i, file_path in enumerate(files):
                split = "train" if i < split_index else "val"
                dest_dir = out_dir / split / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                # Prefixed by source name (the crop's grandparent dir) so
                # two sources' filenames never collide.
                shutil.copy2(file_path, dest_dir / f"{file_path.parent.parent.name}_{file_path.name}")
                total_written += 1

        print(f"Built merged classification dataset: {total_written} crop(s) across "
              f"{sum(1 for f in by_class.values() if f)}/{len(APP_ACTION_CLASSES)} classes at {out_dir}")
        return out_dir


class CourtDatasets:
    """Court-keypoint model training data sources, used by
    CourtDefinition/court.py's keypoint model once fine-tuned.

    Unlike ActionDatasets (which crops individual labeled boxes out of
    several sources), a keypoint model needs the WHOLE image plus its
    keypoint annotations - so this class's job is narrower: download each
    source, remap its own keypoint vocabulary onto this project's 4-point
    half-court scheme (net_left, net_right, baseline_left, baseline_right -
    see Backend/API/calibration.py's POINT_NAMES, which this exists to
    eventually auto-fill), and merge the results into one YOLO-pose
    dataset directory.

    Point scheme (KPT_NAMES below) is deliberately NOT tied to any one
    source dataset's own point count or ordering - it's the minimal
    4-point half-court quadrilateral (the net edge + the opposite baseline
    edge of whatever's visible) that both this project's calibration UI
    and homography math actually need. A source with more points (attack
    lines, extrapolated far-side corners, etc.) just has those extra
    points dropped; a source that only ever labels a strict subset of this
    quadrilateral is skipped for images missing any of the 4.

    Sources and licensing:
        - primaryws/volleyball_court_keypoints_regression_dataset  CC BY 4.0
          862 images, 8 keypoints per instance (this project's own ball
          detector already uses another dataset from the same Roboflow
          org). Verified via its own downloaded data.yaml (its Universe
          page blocked automated fetches during research). Its own
          keypoint indices 1/2 are the net-line ends and 5/6 are the
          opposite baseline's corners (confirmed by manual visual
          inspection across several sample images - see
          PRIMARYWS_KEEP_INDICES below); the other 4 (0, 3, 4, 7) are
          dropped as their exact semantics couldn't be confidently
          determined from the export alone.
        - This project's own already-calibrated footage
          (extract_own_footage_keypoints below) - see its own docstring
          for why this turned out to be essential, not optional: primaryws
          alone is exclusively professional broadcast footage (bright
          arena lighting, a dedicated blue/tan court, an elevated wide
          camera), which is a completely different visual domain from this
          project's actual real-world footage (handheld/GoPro-style low
          angles, a shared multi-sport gym floor with overlapping
          badminton/basketball line markings, harsher indoor lighting) - a
          model trained on primaryws alone scored ~0.98 mAP on its own
          held-out split but detected essentially nothing on real job
          footage, a classic narrow-source-distribution trap.

    Every non-own-footage source below is a THIRD PARTY dataset - nothing
    is bundled with this repo, and nothing downloads without your own
    Roboflow account API key (https://app.roboflow.com/settings/api).
    """

    DATASETS_DIR = ML_DIR / "dataset_courtDefinition"

    # This project's own 4-point half-court scheme - net edge, then the
    # opposite baseline edge, left-to-right within each. flip_idx pairs
    # each point with its mirror for horizontal-flip augmentation
    # (net_left<->net_right, baseline_left<->baseline_right).
    KPT_NAMES = ["net_left", "net_right", "baseline_left", "baseline_right"]
    KPT_SHAPE = [len(KPT_NAMES), 3]
    FLIP_IDX = [1, 0, 3, 2]

    # primaryws's own 8-point label order -> which 4 indices are this
    # project's net_left/net_right/baseline_left/baseline_right (see class
    # docstring).
    PRIMARYWS_KEEP_INDICES = [1, 2, 6, 5]

    @staticmethod
    def download_primaryws(api_key: str) -> Path:
        """862 images, 8 keypoints/instance (4 kept - see class docstring).
        CC BY 4.0 - https://universe.roboflow.com/primaryws/volleyball_court_keypoints_regression_dataset"""
        return download_roboflow_project(
            api_key, "primaryws", "volleyball_court_keypoints_regression_dataset",
            CourtDatasets.DATASETS_DIR / "primaryws",
        )

    @staticmethod
    def _remap_label_line(line: str, keep_indices: list) -> Optional[str]:
        """Rewrites one YOLO-pose label line down to just the kept keypoint
        indices, in KPT_NAMES order. Returns None if any kept point is
        entirely unlabeled (visibility 0 with no coordinates - shouldn't
        happen in a well-formed export, but skip rather than write a
        broken sample)."""
        parts = line.split()
        if len(parts) < 5:
            return None

        cls_and_box = parts[:5]
        kpts = parts[5:]
        if len(kpts) < 3 * (max(keep_indices) + 1):
            return None

        kept = []
        for idx in keep_indices:
            triplet = kpts[idx * 3: idx * 3 + 3]
            if len(triplet) < 3:
                return None
            kept.extend(triplet)

        return " ".join(cls_and_box + kept)

    @staticmethod
    def extract_keypoint_source(source_dir: Path, keep_indices: list, source_name: str) -> Path:
        """Rewrites source_dir's own YOLO-pose export down to this
        project's 4-point scheme (see class docstring) into
        DATASETS_DIR/remapped/<source_name>/<split>/{images,labels}/ -
        images are copied as-is, labels are rewritten via
        _remap_label_line. Rebuilds that source's own subtree from scratch
        each call."""
        dest = CourtDatasets.DATASETS_DIR / "remapped" / source_name
        if dest.exists():
            shutil.rmtree(dest)

        written = 0
        for split_dir in source_dir.iterdir():
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            if not images_dir.is_dir():
                continue

            out_images = dest / split_dir.name / "images"
            out_labels = dest / split_dir.name / "labels"
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            for image_path in images_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue

                remapped_lines = [
                    CourtDatasets._remap_label_line(line, keep_indices)
                    for line in label_path.read_text().splitlines()
                    if line.strip()
                ]
                remapped_lines = [line for line in remapped_lines if line is not None]
                if not remapped_lines:
                    continue

                shutil.copy2(image_path, out_images / image_path.name)
                (out_labels / f"{image_path.stem}.txt").write_text("\n".join(remapped_lines) + "\n")
                written += 1

        print(f"[{source_name}] remapped {written} labeled image(s) into {dest}")
        return dest

    @staticmethod
    def _point_to_kpt_line(points: dict, width: int, height: int) -> Optional[str]:
        """One court.json `points` dict -> a single YOLO-pose label line in
        KPT_NAMES order, normalised to [0, 1] against (width, height). A
        point saved slightly outside the frame (court.json tolerates this -
        see calibration.py, a calibrated corner can legitimately sit just
        past the visible edge) is clamped for the label's x/y but marked
        visibility 1 (labeled, not clearly visible) instead of 2, mirroring
        how primaryws's own export marks an occluded-but-estimated point.
        The bounding box wraps the 4 points with a little padding,
        matching how a source dataset's own "court" box is defined -
        approximate is fine, only the keypoints themselves are used
        downstream."""
        source_points = {
            "net_left": points.get("middle_left"),
            "net_right": points.get("middle_right"),
            "baseline_left": points.get("far_left"),
            "baseline_right": points.get("far_right"),
        }
        if any(source_points[name] is None for name in CourtDatasets.KPT_NAMES):
            return None

        xs, ys, kpt_parts = [], [], []
        for name in CourtDatasets.KPT_NAMES:
            x, y = source_points[name]["x"], source_points[name]["y"]
            visible = 0.0 <= x <= width and 0.0 <= y <= height
            cx, cy = min(max(x, 0.0), width), min(max(y, 0.0), height)
            xs.append(cx)
            ys.append(cy)
            kpt_parts.extend([f"{cx / width:.6f}", f"{cy / height:.6f}", "2" if visible else "1"])

        pad = 0.05
        x1, x2 = max(0.0, min(xs) - width * pad), min(width, max(xs) + width * pad)
        y1, y2 = max(0.0, min(ys) - height * pad), min(height, max(ys) + height * pad)
        box_cx, box_cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        box_w, box_h = (x2 - x1) / width, (y2 - y1) / height

        return " ".join([
            "0", f"{box_cx:.6f}", f"{box_cy:.6f}", f"{box_w:.6f}", f"{box_h:.6f}", *kpt_parts,
        ])

    @staticmethod
    def extract_own_footage_keypoints(
        job_ids: Optional[list] = None, frames_per_job: int = 40, val_fraction: float = 0.2,
    ) -> Path:
        """Turns this project's own already-CONFIRMED court calibrations
        into training data - see the class docstring for why this is
        essential, not optional, alongside primaryws: it's the only source
        that actually looks like this project's real footage (a shared
        multi-sport gym floor, handheld/GoPro-style angles), rather than
        professional broadcast video.

        Every job with a `confirmed: true` court.json (see calibration.py's
        save/confirm endpoints - a human has actually signed off on these 4
        points) contributes up to `frames_per_job` frames, evenly sampled
        across its video, all labeled with that SAME job's own 4 points -
        valid because calibration is (for now) one static homography per
        video; a job whose camera genuinely moves mid-recording would need
        this reworked once multi-segment calibration exists. `job_ids`
        defaults to every confirmed job found on disk; pass a specific list
        to limit which ones contribute."""
        from API import config as api_config
        from API.jobs import store

        dest = CourtDatasets.DATASETS_DIR / "remapped" / "own_footage"
        if dest.exists():
            shutil.rmtree(dest)

        if job_ids is None:
            job_ids = [job.id for job in store.list()]

        written = 0
        for job_id in job_ids:
            court_file = api_config.output_dir(job_id) / "court.json"
            video_path = api_config.find_input_video(job_id)
            if not court_file.exists() or video_path is None:
                continue

            try:
                court = json.loads(court_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not court.get("confirmed"):
                continue

            cap = cv2.VideoCapture(str(video_path))
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if total_frames <= 0 or width <= 0 or height <= 0:
                    continue

                kpt_line = CourtDatasets._point_to_kpt_line(court.get("points", {}), width, height)
                if kpt_line is None:
                    continue

                stride = max(1, total_frames // frames_per_job)
                sampled_indices = list(range(0, total_frames, stride))[:frames_per_job]
                random.Random(0).shuffle(sampled_indices)
                split_index = max(1, int(len(sampled_indices) * (1 - val_fraction)))

                for i, frame_idx in enumerate(sampled_indices):
                    split = "train" if i < split_index else "val"
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = cap.read()
                    if not success:
                        continue

                    images_dir = dest / split / "images"
                    labels_dir = dest / split / "labels"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    labels_dir.mkdir(parents=True, exist_ok=True)

                    stem = f"{job_id}_{frame_idx}"
                    cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)
                    (labels_dir / f"{stem}.txt").write_text(kpt_line + "\n")
                    written += 1
            finally:
                cap.release()

        print(f"[own_footage] extracted {written} labeled frame(s) from confirmed calibrations into {dest}")
        return dest

    @staticmethod
    def build_pose_dataset(remapped_sources: list, out_name: str = "merged") -> Path:
        """Combines every remapped source's train/valid/test splits into
        one YOLO-pose dataset directory with a single data.yaml, ready for
        ultralytics' `YOLO(...).train(data=..., task="pose")`. Copies
        rather than symlinks (Windows-friendly, and keeps the merged dir
        self-contained), prefixing filenames by source name to avoid
        collisions. Always rebuilds out_name from scratch."""
        out_dir = CourtDatasets.DATASETS_DIR / out_name
        if out_dir.exists():
            shutil.rmtree(out_dir)

        split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
        counts = {"train": 0, "val": 0, "test": 0}

        for source_dir in remapped_sources:
            for split_dir in source_dir.iterdir():
                out_split = split_map.get(split_dir.name)
                if out_split is None or not (split_dir / "images").is_dir():
                    continue

                out_images = out_dir / out_split / "images"
                out_labels = out_dir / out_split / "labels"
                out_images.mkdir(parents=True, exist_ok=True)
                out_labels.mkdir(parents=True, exist_ok=True)

                for image_path in (split_dir / "images").iterdir():
                    label_path = split_dir / "labels" / f"{image_path.stem}.txt"
                    if not label_path.exists():
                        continue
                    stem = f"{source_dir.name}_{image_path.stem}"
                    shutil.copy2(image_path, out_images / f"{stem}{image_path.suffix}")
                    shutil.copy2(label_path, out_labels / f"{stem}.txt")
                    counts[out_split] += 1

        data_yaml = out_dir / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({
            "train": "train/images",
            "val": "val/images",
            "test": "test/images" if counts["test"] else None,
            "kpt_shape": CourtDatasets.KPT_SHAPE,
            "flip_idx": CourtDatasets.FLIP_IDX,
            "names": {0: "court"},
            "nc": 1,
        }))

        print(f"Built merged pose dataset: {counts} at {out_dir}")
        return data_yaml
