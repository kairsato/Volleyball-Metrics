"""Fine-tunes this project's own tracker models against combined training
data - the ball detector (see datasetGather.BallDatasets for its
individual data sources), the per-player action classifier (see
datasetGather.ActionDatasets), and the court-keypoint model (see
datasetGather.CourtDatasets) - structured so a further model can be added
the same way later: one `<model>_datasets()` function to gather that
model's training data, one `train_<model>()` function to actually
fine-tune it, both wired into train_all() below alongside the existing
ones.

Usage: edit the CONFIGURATION constants below, then run (from
Backend/Analysis/MachineLearning/):
    python mainTrainingModels.py             # trains everything
    python mainTrainingModels.py ball         # ball detector only
    python mainTrainingModels.py action       # action classifier only
    python mainTrainingModels.py court        # court-keypoint model only

Same "edit the constants, then run" shape as RunEverything.py's own
hardcoded video_path/output_path - no other CLI flags to pass.
"""

import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = ML_DIR.parent
for _directory in (ML_DIR, ANALYSIS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from datasetGather import ActionDatasets as action_datasets  # noqa: E402
from datasetGather import BallDatasets as ball_datasets  # noqa: E402
from datasetGather import CourtDatasets as court_datasets  # noqa: E402
from BallDetection.ballDetection import MODEL_PATH as BALL_BASE_MODEL_PATH  # noqa: E402

# --- Configuration --------------------------------------------------------
# Your own Roboflow account API key - https://app.roboflow.com/settings/api
# - tied to your account, not something this repo can supply.
ROBOFLOW_API_KEY = "jdOqG07uQQgd1juGzcN8"
# Path to a manually-downloaded Kaggle dataset zip (see
# datasetGather.BallDatasets.prepare_kaggle_dataset), or None to skip it.
KAGGLE_ZIP = None
# Specific job IDs to pseudo-label from (see
# datasetGather.BallDatasets.pseudo_label_own_footage), or None for every
# already-complete job.
OWN_FOOTAGE_JOBS = None
# Checkpoint to continue fine-tuning from - defaults to this project's
# already-fine-tuned ball detector rather than a stock YOLO model, since
# that checkpoint (see ballDetection.py's module docstring: precision
# 0.929, recall 0.758, mAP50 0.856, mAP50-95 0.525 on its own original
# validation split) is itself a reasonable starting point this only needs
# to build on, not replace outright.
BASE_MODEL = BALL_BASE_MODEL_PATH
EPOCHS = 50
IMGSZ = 640
OUT_PATH = ML_DIR / "models" / "ball_detector_finetuned.pt"

# Optional 5th action-classifier source - the Ibrahim et al. Volleyball
# dataset (see datasetGather.ActionDatasets module docstring). It ships as
# a manual Google Drive download, not something this script can fetch on
# its own, so both need to be set to Paths for it to be included; leave
# either as None to train from just the 4 Roboflow sources.
IBRAHIM_ANNOTATIONS = None  # path to the extracted volleyball_tracking_annotation/ root
IBRAHIM_IMAGES = None  # path to the extracted videos/ (frame images) root
ACTION_EPOCHS = 30
ACTION_BATCH_SIZE = 384
ACTION_ARCHITECTURE = "resnet50"
ACTION_OUT_PATH = ANALYSIS_DIR / "ActionDetection" / "action_classifier.pt"

COURT_EPOCHS = 100
COURT_BATCH_SIZE = 64
COURT_IMGSZ = 640
COURT_OUT_PATH = ANALYSIS_DIR / "CourtDefinition" / "court_keypoints.pt"
# ---------------------------------------------------------------------------


def ball_detector_datasets() -> Path:
    """Gathers every ball-detector data source - public Roboflow datasets,
    an optional manually-downloaded Kaggle export, and this project's own
    already-processed footage (pseudo-labeled from its own confident
    detections, see BallDatasets.pseudo_label_own_footage) - then merges
    them into one YOLO dataset ready for train_ball_detector() below.

    Each individual source's own download/prepare logic lives in
    datasetGather.BallDatasets, not here - this just calls each of them in
    turn and hands the results to BallDatasets.merge_all()."""
    sources = [
        ball_datasets.download_primaryws(ROBOFLOW_API_KEY),
        ball_datasets.download_aivolleyballref(ROBOFLOW_API_KEY),
        ball_datasets.download_volleyballtest(ROBOFLOW_API_KEY),
        ball_datasets.download_salo_levy(ROBOFLOW_API_KEY),
        ball_datasets.pseudo_label_own_footage(OWN_FOOTAGE_JOBS),
    ]
    if KAGGLE_ZIP is not None:
        sources.append(ball_datasets.prepare_kaggle_dataset(KAGGLE_ZIP))

    return ball_datasets.merge_all(sources)


def train_ball_detector(data_yaml: Path) -> Path:
    """Fine-tunes the ball detector against the merged dataset, from
    BASE_MODEL, for EPOCHS at IMGSZ - see the CONFIGURATION constants
    above."""
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    results = model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ)

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        best.replace(OUT_PATH)
        print(f"Ball detector fine-tuning complete. Best checkpoint saved to {OUT_PATH}")
    else:
        print(f"Ball detector fine-tuning complete, but no best.pt was found - check {results.save_dir}")
    return OUT_PATH


def action_classifier_datasets() -> Path:
    """Gathers every action-classifier data source - 4 public Roboflow
    datasets plus the optional Ibrahim et al. dataset (see the
    CONFIGURATION constants above) - crops each source's labeled boxes,
    remaps them onto this app's 4-class vocabulary, and merges the result
    into one ImageFolder-ready directory for train_action_classifier()
    below. Each individual source's own download/crop logic lives in
    datasetGather.ActionDatasets, not here."""
    crop_sources = [
        action_datasets.extract_bbox_dataset_crops(
            action_datasets.download_mikhail_klyukin(ROBOFLOW_API_KEY),
            action_datasets.CLASS_MAPS["mikhail_klyukin"], "mikhail_klyukin",
        ),
        action_datasets.extract_bbox_dataset_crops(
            action_datasets.download_vbanalyzer(ROBOFLOW_API_KEY),
            action_datasets.CLASS_MAPS["vbanalyzer"], "vbanalyzer",
        ),
        action_datasets.extract_bbox_dataset_crops(
            action_datasets.download_graz_uni(ROBOFLOW_API_KEY),
            action_datasets.CLASS_MAPS["graz_uni"], "graz_uni",
        ),
        action_datasets.extract_bbox_dataset_crops(
            action_datasets.download_vballactionrecognition(ROBOFLOW_API_KEY),
            action_datasets.CLASS_MAPS["vballactionrecognition"], "vballactionrecognition",
        ),
    ]
    if IBRAHIM_ANNOTATIONS is not None and IBRAHIM_IMAGES is not None:
        crop_sources.append(action_datasets.extract_ibrahim_crops(IBRAHIM_ANNOTATIONS, IBRAHIM_IMAGES))

    return action_datasets.build_train_val_split(crop_sources)


def train_action_classifier(data_dir: Path) -> Path:
    """Trains the action classifier against the merged dataset, from
    scratch (ImageNet-pretrained backbone) since - unlike the ball
    detector - there's no previous fine-tune of this model to continue
    from. See the ACTION_* CONFIGURATION constants above."""
    from ActionDetection.training.train import train_from_folder

    return train_from_folder(
        data_dir, epochs=ACTION_EPOCHS, batch_size=ACTION_BATCH_SIZE,
        architecture=ACTION_ARCHITECTURE, out_path=ACTION_OUT_PATH,
    )


def court_keypoint_datasets() -> Path:
    """Gathers the court-keypoint model's data sources and remaps each
    onto this project's 4-point half-court scheme (see
    datasetGather.CourtDatasets docstring), then merges the result into
    one YOLO-pose dataset ready for train_court_keypoints() below."""
    remapped_sources = [
        court_datasets.extract_keypoint_source(
            court_datasets.download_primaryws(ROBOFLOW_API_KEY),
            court_datasets.PRIMARYWS_KEEP_INDICES, "primaryws",
        ),
    ]
    return court_datasets.build_pose_dataset(remapped_sources)


def train_court_keypoints(data_yaml: Path) -> Path:
    """Fine-tunes the court-keypoint model against the merged dataset. See
    the COURT_* CONFIGURATION constants above."""
    from CourtDefinition.keypoints.training.train import train_keypoints

    return train_keypoints(
        data_yaml, epochs=COURT_EPOCHS, batch=COURT_BATCH_SIZE, imgsz=COURT_IMGSZ, out_path=COURT_OUT_PATH,
    )


def train_all(targets=("ball", "action", "court")) -> None:
    """Runs every requested model's training end-to-end. `targets` picks
    which of "ball"/"action"/"court" to run - see main()'s CLI handling
    below."""
    if "ball" in targets:
        data_yaml = ball_detector_datasets()
        train_ball_detector(data_yaml)
    if "action" in targets:
        data_dir = action_classifier_datasets()
        train_action_classifier(data_dir)
    if "court" in targets:
        data_yaml = court_keypoint_datasets()
        train_court_keypoints(data_yaml)


def main():
    if not ROBOFLOW_API_KEY:
        raise SystemExit("Set ROBOFLOW_API_KEY at the top of this file before running.")

    targets = tuple(sys.argv[1:]) or ("ball", "action", "court")
    unknown = set(targets) - {"ball", "action", "court"}
    if unknown:
        raise SystemExit(f"Unknown target(s) {sorted(unknown)} - choose from 'ball', 'action', 'court'.")

    train_all(targets)


if __name__ == "__main__":
    main()
