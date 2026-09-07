<div align="center">

# Volleyball Metrics

</div>

<!--
  Demo GIFs go here. Drop capture files into docs/assets/ using the names
  below and these will render — nothing else in the README needs to change.
-->

### Demo

| Upload & tracking | Rally & action detection | Stats dashboard |
| :---: | :---: | :---: |
| ![Video tracking demo](docs/assets/demo-tracking.gif) | ![Action detection demo](docs/assets/demo-actions.gif) | ![Stats dashboard demo](docs/assets/demo-stats.gif) |

## About

I got fed up with volleyball analytics platforms that charge clubs and individual players absurd subscription fees for something a laptop and a few open-source models can already do. Coaches and players who just want to know their hitting efficiency or where their serve receive breaks down shouldn't have to pay per-video or per-seat for it. So I built this: a self-hosted pipeline that takes a normal match recording — a phone on a tripod, a wall-mounted camera, whatever — and turns it into the same kind of stats the paid tools sell, for free, running on your own machine.

**NOTE:** This project had the use of generative AI pipeline and code development.

<details>
<summary><strong>Inspiration</strong></summary>

This project builds directly on ideas and open datasets from:

- **[shukkkur/VolleyVision](https://github.com/shukkkur/VolleyVision)** — the original inspiration for tackling volleyball with a staged detection/tracking pipeline (ball → players/actions → court), and the source of several of the ball and action detection datasets below.
- **[masouduut94/volleyball_analytics](https://github.com/masouduut94/volleyball_analytics)** — inspiration for treating rally/game-status segmentation as its own video-classification stage, and the source of the fine-tuned VideoMAE checkpoint this project's rally detector is built on.

</details>

## What it gives you

- **Player positioning & movement** — see where each player tends to stand, cover, and move to over the course of a match, not just where they ended up.
- **Set locations & serve/receive patterns** — where sets tend to go and how cleanly serve/receive is handled, rally after rally.
- **Standout performances** — surfaces what individual players did well across the match, rolling up into a general MVP read instead of a gut feeling.
- **Annotated video trajectories** — tracking overlays on the rendered video make it easy to follow one specific player, or the ball itself, through an entire rally.
- **Match & game statistics** — win/loss records, hitting efficiency, unforced errors, and other game stats generated automatically instead of requiring a manual stat-taker.
- **Court-relative metrics** — ball speed, jump height, and shot placement, computed in real court coordinates from a one-time court calibration rather than raw pixels.
- **Team-level rollups** — win rate by action type and hit counts aggregated across every video a team has played, so trends show up over a season, not just one match.
- **All of it for free, on your own hardware** — a decent GPU and an evening of setup instead of a per-seat subscription, built in the open on top of the datasets and prior work that made it possible.

## Frontend

The frontend is a React + TypeScript SPA.

### Screenshots & demos

| Videos & upload | Teams | Players |
| :---: | :---: | :---: |
| ![Videos page demo](docs/assets/frontend-videos.gif) | ![Teams page demo](docs/assets/frontend-teams.gif) | ![Players page demo](docs/assets/frontend-players.gif) |

Per-video setup (the **Setup** tab):

| Scoring | Court calibration | Player identification |
| :---: | :---: | :---: |
| ![Scoring setup demo](docs/assets/frontend-scoring.gif) | ![Court calibration demo](docs/assets/frontend-court-calibration.gif) | ![Player identification demo](docs/assets/frontend-player-id.gif) |

<details>
<summary><strong>Main areas</strong></summary>

- **Videos** — upload a match, watch pipeline stage progress live, and jump into any past video's results.
- **Teams / Players** — roster management, team win/loss and win-rate-by-action radars, per-player stats across every video they've appeared in.
- **Scoring** — set scoring mode (manual / automatic / scoreboard OCR), pick the score-region on screen for OCR, and correct any rally-to-game grouping afterward.
- **Court calibration** — click the four court corners and (optionally) the net-top points once per video; used for every downstream court-relative and height-based calculation.
- **Player identification** — review detected players, name them from the roster, merge duplicates, or ignore false positives (refs, coaches, spectators).

</details>

## Backend

FastAPI app orchestrating a multi-stage pipeline; each stage is a self-contained module under `Backend/Analysis/`.

```
Backend/
  API/                    FastAPI app: auth, job orchestration, stage pipeline, routers per resource
  Analysis/
    PlayerDetection/      YOLO detection + custom multi-tracker ensemble with ReID (tracker.py)
    CourtDefinition/      Court corner/net keypoint model + homography (court.py)
    BallDetection/        Volleyball-specific fine-tuned detector + trajectory smoothing
    GameStatusDetection/  Rally segmentation (fine-tuned VideoMAE classifier)
    ActionDetection/      Per-touch action classification (serve/set/spike/dig/block)
    PostProcessing/       Stats consolidation, dashboard generation, annotated video rendering
    MachineLearning/      Dataset gathering/merging + training entry points for every model above
```

<details>
<summary><strong>Stage-by-stage details</strong></summary>

- **API** (`Backend/API/`) — auth, video upload, job/stage orchestration (`stage_runner.py`, `pipeline.py`), roster/team/player CRUD, scoring (manual/automatic/OCR via `score_cv.py`), sharing, and a router per resource under `routers/`.
- **Player detection & tracking** (`PlayerDetection/tracker.py`) — Ultralytics YOLO for detection, fed into a custom 5-tracker ensemble with a ResNet18-based appearance encoder for re-identifying players who briefly leave the frame or get occluded. Heuristics on top: track-merge scoring by IoU + appearance-embedding distance + motion continuity, and confidence gating to suppress crowd/spectator false positives.
- **Court definition** (`CourtDefinition/court.py`) — a keypoint model locates the court corners/net points; a heuristic homography solve maps pixel space to real court coordinates, and a single-view camera pose (from the two net-top points) drives height estimation for the ball and jumps.
- **Ball detection** (`BallDetection/ballDetection.py`) — a fine-tuned detector plus a trajectory smoother/interpolator (heuristic Kalman-style gap filling) to bridge frames where the ball is occluded or motion-blurred, and a real-world speed estimate from the court homography.
- **Game status detection** (`GameStatusDetection/gameStatusDetection.py`) — a fine-tuned VideoMAE video classifier labels each window as play / no-play / serve, which is stitched (with heuristic minimum-duration and boundary-snapping rules) into rally start/end boundaries.
- **Action detection** (`ActionDetection/actionDetection.py`) — a classifier attributes each touch to a type and, combined with player-track proximity heuristics, to a specific player.
- **Post-processing** (`PostProcessing/`) — consolidates all per-frame detections into rally/game/player stats (`consolidate.py`), renders the annotated video (`renderVideo.py`), transcodes for playback (`transcode.py`), and generates a standalone HTML dashboard (`generate_dashboard.py`).

</details>

## Datasets & credits

Every fine-tuned model below is a **YOLOv8-format** dataset merged from one or more open [Roboflow Universe](https://universe.roboflow.com) datasets via `Backend/Analysis/MachineLearning/datasetGather.py`, then trained by `mainTrainingModels.py`. Full attribution is also shown in-app under **About**.

<details>
<summary><strong>Ball detection</strong></summary>

YOLO11m, fine-tuned on ~2,260 images merged from 4 sources (all CC BY 4.0):

- [aivolleyballref/volleyball_detection](https://universe.roboflow.com/aivolleyballref/volleyball_detection) — 771 images
- [primaryws/volleyball_ball_object_detection_dataset](https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset) — 548 images
- [salo-levy-nlqrn/volley-ball-detection](https://universe.roboflow.com/salo-levy-nlqrn/volley-ball-detection) — 120 images
- [volleyballtest/volleyball-fdqxb](https://universe.roboflow.com/volleyballtest/volleyball-fdqxb) — 820 images
- **My own footage** — supplemented automatically: `BallDatasets.pseudo_label_own_footage()` turns my own already-processed matches' high-confidence ball detections into new training labels, so accuracy keeps improving the more I use the app.

</details>

<details>
<summary><strong>Action detection</strong></summary>

Trained from scratch for this project, on ~29,000 images merged from 4 sources (all CC BY 4.0):

- [shukur-sabzaliev-zc3en/volleyball-activity-dataset](https://universe.roboflow.com/shukur-sabzaliev-zc3en/volleyball-activity-dataset) — 25,000 images, uploaded by [Shakhansho Sabzaliev](https://github.com/shukkkur) (VolleyVision), sourced from Graz University of Technology's [Volleyball Activity Dataset](https://www.tugraz.at/index.php?id=17751) (Austrian Volley League 2011/12)
- [vbanalyzer/volleyball-action-recognition-k6tqv](https://universe.roboflow.com/vbanalyzer/volleyball-action-recognition-k6tqv) — 1,806 images
- [vballactionrecognition/volleyball-action-recognition-7rnpb](https://universe.roboflow.com/vballactionrecognition/volleyball-action-recognition-7rnpb) — 1,003 images
- [mikhail-klyukin/volleyball_dataset](https://universe.roboflow.com/mikhail-klyukin/volleyball_dataset) — 1,236 images
- **My own footage** — not yet automated for this model; for now this class grows only through the open sources above.

</details>

<details>
<summary><strong>Court keypoints</strong></summary>

862 images (CC BY 4.0):

- [primaryws/volleyball_court_keypoints_regression_dataset](https://universe.roboflow.com/primaryws/volleyball_court_keypoints_regression_dataset)
- **My own footage** — supplemented automatically: `CourtDatasets.extract_own_footage_keypoints()` turns my own already-confirmed court calibrations into new labeled keypoint frames.

</details>

<details>
<summary><strong>Game status / rally detection</strong></summary>

A [masouduut94/volleyball_analytics](https://github.com/masouduut94/volleyball_analytics) fine-tuned VideoMAE checkpoint, base model `MCG-NJU/videomae-base-finetuned-kinetics` (Hugging Face, CC-BY-NC-4.0 — non-commercial use only).

- **My own footage** — not yet automated for this model; this stage currently relies solely on the fine-tuned checkpoint above.

</details>

<details>
<summary><strong>Player detection</strong></summary>

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0 / commercial license); re-identification is a ResNet18 appearance encoder trained for this project.

</details>

## Tech stack

- **Backend**: Python, FastAPI, PyTorch, Ultralytics YOLO, Hugging Face Transformers (VideoMAE), OpenCV
- **Frontend**: React, TypeScript, Vite, MUI (Material UI), React Router

## Getting started

### Prerequisites

- Python 3.11+ with a virtual environment at `Backend/.venv`
- Node.js 18+
- A CUDA-capable GPU is strongly recommended (tracking and the game-status classifier both run PyTorch models per frame/window)

### Backend setup

```bash
cd Backend
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

The game-status classifier expects a fine-tuned VideoMAE checkpoint at `Backend/Analysis/GameStatusDetection/Models/VolleyballAnalytics/3-states/checkpoint/` (see [Datasets & credits](#datasets--credits) above — it's not bundled in this repo).

### Frontend setup

```bash
cd Frontend
npm install
```

### Running

From the repo root:

```bash
start.bat
```

This launches the backend (`uvicorn API.main:app --reload --host 0.0.0.0 --port 8000`) and the frontend dev server (`npm run dev`, Vite — defaults to `http://localhost:5173`) each in their own window. Open the frontend URL and upload a video to get started.

Both bind to `0.0.0.0` rather than just `localhost`, so another device on the same network (a phone, a laptop) can reach them via this machine's own IP — e.g. `http://192.168.1.27:5173`. If you launch the backend by hand instead of via `start.bat`, include `--host 0.0.0.0` yourself, or it'll silently fall back to loopback-only and be unreachable from anywhere but this machine.

## License

This project's code is licensed under the [MIT License](LICENSE).

**Third-party licenses**, by component:

| Component | License |
| --- | --- |
| Ball, action, and court datasets (Roboflow Universe sources) | CC BY 4.0 |
| Game-status base model (`MCG-NJU/videomae-base-finetuned-kinetics`) | CC-BY-NC-4.0 (non-commercial only) |
| Player detection (Ultralytics YOLO) | AGPL-3.0, or a commercial Ultralytics license |

See [Datasets & credits](#datasets--credits) above for the full per-dataset attribution.
