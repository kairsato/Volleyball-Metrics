# dataset_courtDefinition

Training data for the **court-keypoint model** used by
[`CourtDefinition/court.py`](../../CourtDefinition/court.py)
(loads `CourtDefinition/court_keypoints.pt`).

- Gathered/merged by `datasetGather.py`'s `CourtDatasets` class (in this
  same `MachineLearning/` folder).
- Trained by `mainTrainingModels.py`'s `court_keypoint_datasets()` /
  `train_court_keypoints()`, which delegates the actual training loop to
  [`CourtDefinition/keypoints/training/train.py`](../../CourtDefinition/keypoints/training/train.py).
- This folder's contents (raw Roboflow exports, `remapped/`, `merged/`)
  are gitignored - regenerate them by running `mainTrainingModels.py`
  rather than expecting them to already be here after a fresh clone.
