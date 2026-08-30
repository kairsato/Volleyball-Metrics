from pathlib import Path

# Backend/api/data/<job_id>/input<ext>   - uploaded video
# Backend/api/data/<job_id>/output/...   - everything the pipeline stages write
DATA_DIR = Path(__file__).parent / "data"

JOB_METADATA_NAME = "job.json"
PLAYER_NAMES_NAME = "player_names.json"
PLAYER_IGNORED_NAME = "player_ignored.json"
# Sits directly in DATA_DIR, not under any job_dir - this is a global roster
# of real people's names shared across every job, not job-specific data.
PLAYER_ROSTER_NAME = "player_roster.json"
STATS_FILE_NAME = "player_stats.json"
DASHBOARD_FILE_NAME = "dashboard.html"
ANNOTATED_VIDEO_NAME = "analysis.mp4"
COURT_FILE_NAME = "court.json"
# A cached "clean plate" of the court with people erased (see
# calibration.build_clean_court_frame) - computed once per job and reused,
# since it's expensive (samples ~20 frames across the whole video) and the
# source video itself never changes after upload.
COURT_BACKGROUND_FILE_NAME = "court_background.jpg"
GAME_STATUS_FILE_NAME = "game_status.json"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Only one heavy pipeline run (GPU/CPU bound) is allowed at a time.
MAX_CONCURRENT_PIPELINE_RUNS = 1

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


def input_video_path(job_id: str, suffix: str = "") -> Path:
    return job_dir(job_id) / f"input{suffix}"


def output_dir(job_id: str) -> Path:
    return job_dir(job_id) / "output"


def find_input_video(job_id: str) -> Path | None:
    matches = sorted(job_dir(job_id).glob("input.*"))
    return matches[0] if matches else None
