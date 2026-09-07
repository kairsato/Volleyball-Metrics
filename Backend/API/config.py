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
# Raw per-frame ball candidates ballDetection.detectBall saves alongside its
# chosen trajectory - lets a later calibration change re-pick the ball (see
# ballDetection.reselect_ball/pipeline.start_recalibration) without
# re-running detection. A job whose ball detection ran before this file
# existed has no such log - jobs_router.recalibrate_job falls back to a full
# reprocess for those.
BALL_CANDIDATES_FILE_NAME = "ball_candidates.json"
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

# Lets another device on the same local network (a phone, a laptop) reach
# this API when the frontend is loaded via the host machine's LAN IP
# instead of localhost - e.g. http://192.168.1.27:5173. Scoped to private
# address ranges (RFC 1918) rather than allow_origins="*", since this is a
# dev server with no auth of its own; a device outside the LAN still can't
# reach it at all unless the host's firewall/router expose it further.
CORS_ORIGIN_REGEX = (
    r"^http://(localhost|127\.0\.0\.1"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"):(5173|3000)$"
)


def job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


def input_video_path(job_id: str, suffix: str = "") -> Path:
    return job_dir(job_id) / f"input{suffix}"


def output_dir(job_id: str) -> Path:
    return job_dir(job_id) / "output"


def find_input_video(job_id: str) -> Path | None:
    matches = sorted(job_dir(job_id).glob("input.*"))
    return matches[0] if matches else None


# Multi-video jobs: segment 0 (the first uploaded file) always lives at the
# flat paths above (input<ext>, output/) - the exact same paths a plain
# single-video job has always used. Only segments 1+ get this new
# videos/<segment_id>/ subtree, so an existing single-video job on disk, or
# a new one, never has to know multi-video jobs exist at all. See
# resolve_video_and_output below, the one seam every segment-aware call
# site should go through instead of reasoning about the order==0 case
# itself.
def video_dir(job_id: str, segment_id: str) -> Path:
    return job_dir(job_id) / "videos" / segment_id


def segment_video_path(job_id: str, segment_id: str, suffix: str = "") -> Path:
    return video_dir(job_id, segment_id) / f"input{suffix}"


def find_segment_video(job_id: str, segment_id: str) -> Path | None:
    matches = sorted(video_dir(job_id, segment_id).glob("input.*"))
    return matches[0] if matches else None


def segment_output_dir(job_id: str, segment_id: str) -> Path:
    return output_dir(job_id) / "segments" / segment_id


def segment_destination(job_id: str, segment: dict) -> Path:
    """Where to WRITE a segment's video file during upload - the
    deterministic counterpart to resolve_video_and_output below, which
    instead LOOKS UP an already-uploaded file by globbing (and returns None
    if nothing's there yet - exactly wrong for "where should the upload
    handler stream this to"). Needs segment["suffix"] (the extension,
    including the dot) since, unlike a lookup, there's nothing on disk yet
    to glob for."""
    suffix = segment["suffix"] or ""
    if segment["order"] == 0:
        return input_video_path(job_id, suffix)
    return segment_video_path(job_id, segment["id"], suffix)


def resolve_video_and_output(job_id: str, segment: dict) -> tuple[Path | None, Path]:
    """(video_path, output_path) for one segment dict (see jobs.job_videos)
    - segment["order"] == 0 always resolves to the legacy flat paths,
    regardless of how many segments the job has or what that segment's own
    id is."""
    if segment["order"] == 0:
        return find_input_video(job_id), output_dir(job_id)
    segment_id = segment["id"]
    return find_segment_video(job_id, segment_id), segment_output_dir(job_id, segment_id)
