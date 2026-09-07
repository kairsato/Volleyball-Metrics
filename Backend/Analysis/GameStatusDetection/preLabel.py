"""
Runs the trained game-state model over a video and writes a draft
game_status_labels.json - the exact file label.py's manual GUI reads and
writes. Opening that video in label.py afterwards loads the model's guesses
as the starting breakpoints, so correcting a video is a matter of fixing
whatever the model got wrong instead of labeling from scratch - much faster,
especially now that most chunks are already right.

    python preLabel.py --video path/to/video.mp4 [--output path/to/folder]

Then: python ../../label.py --video path/to/video.mp4 --output path/to/folder
"""
import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from label import _export  # noqa: E402

from gameStatusDetection import STRIDE_FRAMES, classifyVideoChunks  # noqa: E402


def preLabel(video_path, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    fps, total_frames, smoothed_labels = classifyVideoChunks(video_path, progress_label="pre-label")

    breakpoints = [(i * STRIDE_FRAMES, state) for i, state in enumerate(smoothed_labels)
                   if i == 0 or state != smoothed_labels[i - 1]]

    labels_file = _export(breakpoints, total_frames, fps, output_path)
    print(f"Draft labels ready - open in label.py to review/correct: {video_path}")
    return labels_file


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to the video's own folder.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = args.output or args.video.parent
    preLabel(str(args.video), str(output))
