# dataset_actionDetection

Training data for the per-player **action classifier** used by
[`ActionDetection/actionDetection.py`](../../ActionDetection/actionDetection.py)
(loads `ActionDetection/action_classifier.pt`).

- Gathered/merged by `datasetGather.py`'s `ActionDatasets` class (in this
  same `MachineLearning/` folder).
- Trained by `mainTrainingModels.py`'s `action_classifier_datasets()` /
  `train_action_classifier()`, which delegates the actual training loop to
  [`ActionDetection/training/train.py`](../../ActionDetection/training/train.py).
- This folder's contents (raw Roboflow exports, `raw_crops/`, `merged/`)
  are gitignored - regenerate them by running `mainTrainingModels.py`
  rather than expecting them to already be here after a fresh clone.
