"""Generates downscaled renditions of the source video for the Results
page's video quality selector - reuses cv2 (already a hard dependency for
every other video I/O in this pipeline, including this same package's own
renderVideo.py) rather than adding ffmpeg as a new external binary
dependency, the same reasoning that kept UPnP off miniupnpc.

Never upscales: a tier is only generated when its target height is smaller
than the source's own height, so a lower-resolution source doesn't get a
padded-out "480p" file that's really just a relabeled copy. The untouched
source video itself always doubles as the "original" quality tier - no
rendition file is ever written for that one.

Best-effort by design: this stage runs after the analysis pipeline's real
work is done, so a transcoding failure (corrupt frame, disk full, whatever)
should never fail the whole job over what's ultimately a nice-to-have -
every failure path here is caught and logged, never raised.
"""
from pathlib import Path

import cv2

# height in pixels -> filename suffix. Order doesn't affect anything (each
# tier is written independently in the same single pass over the source).
TRANSCODE_TIERS = {"1080p": 1080, "720p": 720, "480p": 480}


def rendition_filename(tier: str) -> str:
    return f"source_{tier}.mp4"


def _even(value: float) -> int:
    return max(2, int(round(value / 2) * 2))


def generate_renditions(video_path, output_path) -> list[str]:
    """Also called on every recalibration (see pipeline._phase_two) even
    though the source video itself never changes there - each tier is
    skipped when its output file already exists, so a repeat call is a
    cheap no-op rather than re-encoding the whole video again."""
    output_path = Path(output_path)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"transcode: could not open {video_path}; skipping quality renditions.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        already_done = [tier for tier in TRANSCODE_TIERS if (output_path / rendition_filename(tier)).exists()]
        tiers = [
            (tier, h) for tier, h in TRANSCODE_TIERS.items()
            if h < source_height and tier not in already_done
        ]
        if source_width <= 0 or source_height <= 0:
            cap.release()
            return already_done
        if not tiers:
            cap.release()
            return already_done

        writers = []
        for tier, target_height in tiers:
            target_width = _even(source_width * (target_height / source_height))
            writer = cv2.VideoWriter(
                str(output_path / rendition_filename(tier)),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (target_width, target_height),
            )
            writers.append((tier, target_width, target_height, writer))

        while True:
            success, frame = cap.read()
            if not success:
                break
            for _, target_width, target_height, writer in writers:
                writer.write(cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA))

        cap.release()
        for _, _, _, writer in writers:
            writer.release()

        return already_done + [tier for tier, _, _, _ in writers]
    except Exception as exc:  # noqa: BLE001 - best-effort, see module docstring
        print(f"transcode: failed to generate quality renditions: {exc}")
        return []
