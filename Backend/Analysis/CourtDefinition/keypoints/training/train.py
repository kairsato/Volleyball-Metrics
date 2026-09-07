"""Trains the court-keypoint model (net_left/net_right/baseline_left/
baseline_right - see MachineLearning/datasetGather.py's CourtDatasets
class) against the merged dataset that class builds.

Usage (from Backend/Analysis/, so CourtDefinition is importable as a
top-level package):
    python -m CourtDefinition.keypoints.training.train \\
        --data MachineLearning/dataset_courtDefinition/merged/data.yaml --epochs 100

Ordinarily reached indirectly via MachineLearning/mainTrainingModels.py,
which gathers+merges every source and then calls train_keypoints() below
directly; this CLI is for retraining against an already-merged data.yaml
without re-downloading anything.
"""

import argparse
from pathlib import Path

from .model import BASE_CHECKPOINT


def train_keypoints(
    data_yaml,
    epochs: int = 100,
    batch: int = 64,
    imgsz: int = 640,
    out_path=Path("court_keypoints.pt"),
) -> Path:
    """Fine-tunes BASE_CHECKPOINT (COCO-pretrained, 17 human keypoints) on
    this project's own 4-point court scheme - ultralytics transparently
    reinitialises the pose head to match data.yaml's kpt_shape when it
    differs from the checkpoint's own, the same transfer-learning path this
    project's ball detector already relies on for its 1-class detection
    head. Large batch + a small backbone (see model.py) keeps this fast
    even on CPU-bound data loading; bump `batch` further if GPU
    utilisation looks low during a run.
    """
    from ultralytics import YOLO

    model = YOLO(BASE_CHECKPOINT)
    results = model.train(data=str(data_yaml), task="pose", epochs=epochs, imgsz=imgsz, batch=batch)

    best = Path(results.save_dir) / "weights" / "best.pt"
    out_path = Path(out_path)
    if best.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        best.replace(out_path)
        print(f"Court-keypoint training complete. Best checkpoint saved to {out_path}")
    else:
        print(f"Court-keypoint training complete, but no best.pt was found - check {results.save_dir}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=Path("court_keypoints.pt"))
    args = parser.parse_args()

    train_keypoints(args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, out_path=args.out)


if __name__ == "__main__":
    main()
