"""Runs a single pipeline stage as a standalone process.

Invoked as `python -m API.stage_runner <stage> <video_path> <output_path>`
by API/pipeline.py, one stage at a time. Running each stage as its own
process (rather than a thread inside the API server) is what makes
cancellation actually work: a YOLO inference loop has no cooperative
"should I stop?" checks anywhere in it, so the only reliable way to stop one
mid-frame is to kill the OS process running it.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from CourtDefinition.BallDetection.ballDetection import detectBall  # noqa: E402
from PlayerDetection.tracker_offline import trackplayers_offline  # noqa: E402
from GameStatusDetection.gameStatusDetection import detectGameStatus  # noqa: E402
from ActionDetection.actionDetection import detectActions  # noqa: E402
from PostProcessing.consolidate import consolidateStats  # noqa: E402
from PostProcessing.renderVideo import renderAnnotatedVideo  # noqa: E402
from PostProcessing.generate_dashboard import generateDashboard  # noqa: E402
from API.players import write_grouped_actions  # noqa: E402


def _consolidate_with_groups(output: str):
    grouped_file = write_grouped_actions(Path(output))
    consolidateStats(output, actions_filename=grouped_file.name)


STAGE_FUNCS = {
    "player_tracking": lambda video, output: trackplayers_offline(video, output, show_preview=False, save_video=False),
    "ball_detection": lambda video, output: detectBall(video, output, show_preview=False, save_video=False),
    "game_status": lambda video, output: detectGameStatus(video, output),
    "action_detection": lambda video, output: detectActions(video, output),
    "consolidating": lambda video, output: _consolidate_with_groups(output),
    "dashboard": lambda video, output: generateDashboard(output),
    "rendering": lambda video, output: renderAnnotatedVideo(video, output),
}


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: stage_runner.py <stage> <video_path> <output_path>", file=sys.stderr)
        return 2

    stage, video_path, output_path = sys.argv[1:4]
    func = STAGE_FUNCS.get(stage)
    if func is None:
        print(f"Unknown stage: {stage}", file=sys.stderr)
        return 2

    func(video_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
