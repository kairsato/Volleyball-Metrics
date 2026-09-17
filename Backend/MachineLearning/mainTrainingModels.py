"""Fine-tunes this project's own tracker models against combined training
data - the ball detector (see datasetGather.BallDatasets for its
individual data sources), the full-frame action detector (see
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
    python mainTrainingModels.py action       # action detector only
    python mainTrainingModels.py court        # court-keypoint model only

Same "edit the constants, then run" shape as RunEverything.py's own
hardcoded video_path/output_path - no other CLI flags to pass.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ML_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = ML_DIR.parent
for _directory in (ML_DIR, ANALYSIS_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from Backend.MachineLearning.datasetGather import ActionDatasets as action_datasets  # noqa: E402
from Backend.MachineLearning.datasetGather import BallDatasets as ball_datasets  # noqa: E402
from Backend.MachineLearning.datasetGather import CourtDatasets as court_datasets  # noqa: E402
from BallDetection.ballDetection import MODEL_PATH as BALL_BASE_MODEL_PATH  # noqa: E402

# --- Configuration --------------------------------------------------------
# Credentials come from Backend/.env.local (gitignored - see .env.example
# for the keys this reads) rather than being hardcoded here, since they're
# tied to your own Roboflow/Kaggle accounts, not something this repo can
# supply.
load_dotenv(ANALYSIS_DIR / ".env.local")

# Your own Roboflow account API key - https://app.roboflow.com/settings/api
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
if not ROBOFLOW_API_KEY:
    print("Warning: ROBOFLOW_API_KEY not set in Backend/.env.local - Roboflow downloads will fail.")

# Your own Kaggle account credentials - https://www.kaggle.com/settings
# ("Create New Token"). Not currently used to automate downloads (see
# datasetGather.BallDatasets.prepare_kaggle_dataset - Kaggle exports are
# still fetched by hand), kept here for parity with ROBOFLOW_API_KEY and
# for the kaggle package to pick up automatically if that changes.
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")
if not KAGGLE_KEY:
    print("Warning: KAGGLE_KEY not set in Backend/.env.local - Kaggle downloads will fail.")

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

# No prior action-detector fine-tune exists to continue from (unlike the
# ball detector's BASE_MODEL above), so this starts from a stock pretrained
# YOLO checkpoint already vendored in the repo - the same way court
# keypoints starts from a stock yolo11n-pose.pt (see
# CourtDetection/keypoints/training/model.py's BASE_CHECKPOINT).
ACTION_BASE_MODEL = ANALYSIS_DIR.parent / "yolo11m.pt"
ACTION_EPOCHS = 50
ACTION_IMGSZ = 640
ACTION_BATCH_SIZE = 384
ACTION_OUT_PATH = ANALYSIS_DIR / "ActionDetection" / "action_detector.pt"

COURT_EPOCHS = 100
COURT_BATCH_SIZE = 64
COURT_IMGSZ = 640
COURT_OUT_PATH = ANALYSIS_DIR / "CourtDetection" / "court_keypoints.pt"
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


def action_detector_datasets() -> Path:
    """Gathers every action-detector data source - 4 public Roboflow
    datasets (see the CONFIGURATION constants above) - and merges them
    directly as YOLO detection data (each source's own bounding boxes kept
    intact, remapped onto this app's 4-class vocabulary) into one dataset
    ready for train_action_detector() below. Each individual source's own
    download logic lives in datasetGather.ActionDatasets, not here."""
    sources = [
        action_datasets.download_mikhail_klyukin(ROBOFLOW_API_KEY),
        action_datasets.download_vbanalyzer(ROBOFLOW_API_KEY),
        action_datasets.download_graz_uni(ROBOFLOW_API_KEY),
        action_datasets.download_vballactionrecognition(ROBOFLOW_API_KEY),
    ]
    return action_datasets.merge_all(sources)


def train_action_detector(data_yaml: Path) -> Path:
    """Fine-tunes the action detector against the merged dataset, from
    ACTION_BASE_MODEL (a stock pretrained YOLO checkpoint - see that
    constant's comment above for why, unlike the ball detector, there's no
    prior fine-tune to continue from), for ACTION_EPOCHS at ACTION_IMGSZ."""
    from ultralytics import YOLO

    model = YOLO(str(ACTION_BASE_MODEL))
    results = model.train(
        data=str(data_yaml), epochs=ACTION_EPOCHS, imgsz=ACTION_IMGSZ, batch=ACTION_BATCH_SIZE,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        ACTION_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        best.replace(ACTION_OUT_PATH)
        print(f"Action detector fine-tuning complete. Best checkpoint saved to {ACTION_OUT_PATH}")
    else:
        print(f"Action detector fine-tuning complete, but no best.pt was found - check {results.save_dir}")
    return ACTION_OUT_PATH


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
        court_datasets.extract_own_footage_keypoints(),
    ]
    return court_datasets.build_pose_dataset(remapped_sources)


def train_court_keypoints(data_yaml: Path) -> Path:
    """Fine-tunes the court-keypoint model against the merged dataset. See
    the COURT_* CONFIGURATION constants above."""
    from CourtDetection.keypoints.training.train import train_keypoints

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
        data_yaml = action_detector_datasets()
        train_action_detector(data_yaml)
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
