# dataset_gameStatusDetection

Cached training frames for the **game-status model** used by
[`GameStatusDetection/gameStatusDetection.py`](../../GameStatusDetection/gameStatusDetection.py)
(deployed checkpoints live in `GameStatusDetection/Models/Custom/` and
`GameStatusDetection/Models/VolleyballAnalytics/`, which are separate from
this folder).

- Populated/read by
  [`GameStatusDetection/trainGameStatus.py`](../../GameStatusDetection/trainGameStatus.py)'s
  `CACHE_DIR`, and converted from source annotation formats by
  `Backend/convert_game_status_dataset.py`.
- Unlike the other three `dataset_*` folders here, this one isn't gathered
  via `datasetGather.py` - game-status training has its own caching
  convention (`PersonalVolleyballDataset/`, `VolleyballAnalyticsDatasetConverted/`)
  predating that module.
- This folder's contents are gitignored - regenerate them by running
  `trainGameStatus.py` rather than expecting them to already be here after
  a fresh clone.
