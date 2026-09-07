# dataset_ballDetection

Training data for the production **ball detector** used by
[`BallDetection/ballDetection.py`](../../BallDetection/ballDetection.py)
(`MODEL_PATH`/`SECONDARY_MODEL_PATH` load the fine-tuned checkpoints from
`../models/`) and by the experimental hybrid pipeline in
[`ballDetection/`](../ballDetection/) (`hybrid_ball_detector.py`,
`model_benchmark.py`).

- Gathered/merged by `datasetGather.py`'s `BallDatasets` class (in this
  same `MachineLearning/` folder).
- Trained by `mainTrainingModels.py`'s `ball_detector_datasets()` /
  `train_ball_detector()`, and separately benchmarked across several
  architectures by `ballDetection/model_benchmark.py`.
- This folder's contents (raw Roboflow exports, `own_footage/`, `merged/`)
  are gitignored - regenerate them by running `mainTrainingModels.py`
  rather than expecting them to already be here after a fresh clone.
