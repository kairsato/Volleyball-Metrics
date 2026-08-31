# Volleyball-Metrics

Turns a raw volleyball match recording into player stats, rally breakdowns, and an annotated video - automatic court calibration, player/ball tracking, and action detection under the hood.

Upload a video, click the four court corners once, and the pipeline takes it from there: it tracks every player and the ball, segments the match into rallies, classifies each touch (serve/set/spike/dig/block), and rolls all of that up into a per-player stats dashboard and an annotated copy of the video.

## What it does

- **Court calibration** - click the four court corners and net points once; everything downstream (player positions, ball speed, court-relative stats) is computed in real court coordinates, not raw pixels.
- **Player detection & tracking** - a YOLO detector plus a custom multi-tracker ensemble with appearance-based re-identification, so a player keeps their identity even after being briefly occluded or leaving the frame.
- **Ball detection & speed** - a volleyball-specific fine-tuned detector tracks the ball and estimates its real-world speed.
- **Game status / rally detection** - a fine-tuned video classifier segments the match into rallies vs. dead-ball stretches.
- **Action detection** - classifies each touch by type (serve, set, spike, dig, block) and attributes it to a player.
- **Player review** - name detected players from a reusable roster, merge duplicate detections, or ignore false positives (refs, coaches) - with a hint when two separate detections might be the same person.
- **Scoring** - track which side won each rally and how rallies group into games, either by hand, automatically (inferred from ball/player position), or by reading a scoreboard on screen (OCR) - all correctable afterward.
- **Teams** - group roster players into named teams (e.g. "Varsity") independent of any one video; a team's page rolls up its win/loss record, a win-rate-by-action radar, and hit-count stats across every video it's played in.
- **Results page** - per video: an Overview tab (hit counts plus a weighted quality score for serve/receive/set/spike, factoring in speed, placement, and - when the court's net-top points are calibrated - trajectory height), a Stats tab (win/loss record and win-rate radar), a Rallies tab (grouped by game, with the winner highlighted), an Actions tab (every detected touch, filterable), and a Setup tab (court calibration, player review, scoring).
- **Annotated video & dashboard** - a rendered copy of the video with tracking overlays, plus a standalone HTML stats dashboard.

## Tech stack

- **Backend**: Python, FastAPI, PyTorch, Ultralytics YOLO, Hugging Face Transformers (VideoMAE), OpenCV
- **Frontend**: React, TypeScript, Vite, MUI (Material UI), React Router

## Project structure

```
Backend/
  API/                  FastAPI app: routes, job orchestration, stage pipeline
  Analysis/
    PlayerDetection/    Tracking (detection fusion, re-identification, tracklet consolidation)
    CourtDefinition/    Court calibration + ball detection
    GameStatusDetection/  Rally segmentation (VideoMAE classifier)
    ActionDetection/    Per-touch action classification
    PostProcessing/     Stats consolidation, dashboard generation, video rendering
Frontend/
  src/
    pages/              Home, Videos, Players, PlayerStats, Teams, TeamStats, About,
                         Video (per-job workspace), CourtCalibration, PlayerIdentification,
                         ScoringDetermination
    components/         Shared UI - player review, stage progress, results tabs, etc.
```

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

The game-status classifier expects a fine-tuned VideoMAE checkpoint at `Backend/Analysis/GameStatusDetection/Models/VolleyballAnalytics/3-states/checkpoint/` (see [Credits](#credits) below, also available in-app under **About** - it's not bundled in this repo).

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

This launches the backend (`uvicorn API.main:app --reload --host 0.0.0.0 --port 8000`) and the frontend dev server (`npm run dev`, Vite - defaults to `http://localhost:5173`) each in their own window. Open the frontend URL and upload a video to get started.

Both bind to `0.0.0.0` rather than just `localhost`, so another device on the same network (a phone, a laptop) can reach them via this machine's own IP - e.g. `http://192.168.1.27:5173`. If you launch the backend by hand instead of via `start.bat`, include `--host 0.0.0.0` yourself, or it'll silently fall back to loopback-only and be unreachable from anywhere but this machine.

## Credits

- **Player detection**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0 / commercial license)
- **Ball detection**: a YOLO11m detector fine-tuned on the open [VolleyVision / Volleyball_v2 dataset](https://universe.roboflow.com) (Roboflow, CC BY 4.0, credit: shukur-sabzaliev1)
- **Game status / rally detection**: a VideoMAE checkpoint fine-tuned by [masouduut94/volleyball_analytics](https://github.com/masouduut94/volleyball_analytics), base model `MCG-NJU/videomae-base-finetuned-kinetics` (Hugging Face, CC-BY-NC-4.0 - non-commercial use only)
- **Action detection**: trained from scratch for this project
- Full attribution also available in-app under **About**.
