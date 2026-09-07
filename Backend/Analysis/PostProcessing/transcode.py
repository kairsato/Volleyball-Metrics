"""Generates downscaled renditions of the source video for the Results
page's video quality selector.

Shells out to a bundled ffmpeg binary (via imageio-ffmpeg - a pip-installed
package, not a system-wide dependency the way a "real" ffmpeg install would
be) to encode genuine H.264, rather than cv2.VideoWriter: this used to write
mp4v/MPEG-4 Part 2 instead, which browsers don't reliably support at all -
confirmed directly (a real <video> element loading one of those renditions
never even fired loadedmetadata in Chrome), not just "lower quality than
ideal." cv2.VideoWriter's own H.264 path isn't a fix either - it depends on
a system OpenH264 codec DLL that isn't guaranteed to be present, and silently
produces a 0-byte file when it's missing rather than failing loudly.

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
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

# height in pixels -> filename suffix. Order doesn't affect anything (each
# tier is encoded independently, in its own ffmpeg invocation).
TRANSCODE_TIERS = {"1080p": 1080, "720p": 720, "480p": 480}

# Standard libx264 quality/size tradeoff - visually near-lossless for footage
# like this (a static wide shot, not high-motion close-ups), well-compressed.
# Resolution reduction alone already does most of the file-size work between
# tiers, so one CRF for all of them keeps this simple rather than needing a
# separate "how much worse should the smallest tier look" judgment call.
CRF = 23
# "veryfast" trades some compression efficiency for encode speed - this runs
# as a post-processing nice-to-have after the real analysis work is already
# done (see module docstring), so minutes of extra encode time for a
# marginally smaller file isn't worth it; "ultrafast" was skipped as too
# lossy a tradeoff the other direction.
PRESET = "veryfast"


def rendition_filename(tier: str) -> str:
    return f"source_{tier}.mp4"


def generate_renditions(video_path, output_path) -> list[str]:
    """Also called on every recalibration (see pipeline._phase_two) even
    though the source video itself never changes there - each tier is
    skipped when its output file already exists, so a repeat call is a
    cheap no-op rather than re-encoding the whole video again."""
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"transcode: could not open {video_path}; skipping quality renditions.")
        return []
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if source_height <= 0:
        return []

    already_done = [tier for tier in TRANSCODE_TIERS if (output_path / rendition_filename(tier)).exists()]
    tiers = [
        (tier, h) for tier, h in TRANSCODE_TIERS.items()
        if h < source_height and tier not in already_done
    ]
    if not tiers:
        return already_done

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    done = list(already_done)
    for tier, target_height in tiers:
        out_file = output_path / rendition_filename(tier)
        tmp_file = out_file.with_suffix(".tmp.mp4")
        try:
            # scale=-2:H: keep the source aspect ratio, only constrain
            # height - "-2" rounds the computed width to the nearest even
            # number, which libx264's 4:2:0 chroma subsampling requires.
            # Written to a .tmp file first and renamed into place once
            # ffmpeg exits cleanly, so a mid-encode crash/kill never leaves
            # a partial file behind that a later "already exists" check
            # would wrongly treat as a finished rendition.
            subprocess.run(
                [
                    ffmpeg_exe, "-y", "-i", str(video_path),
                    "-vf", f"scale=-2:{target_height}",
                    "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
                    "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart",
                    str(tmp_file),
                ],
                capture_output=True, text=True, check=True,
            )
            tmp_file.replace(out_file)
            done.append(tier)
        except subprocess.CalledProcessError as exc:
            print(f"transcode: failed to generate {tier} rendition: {exc.stderr.strip()[-2000:]}")
            tmp_file.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - best-effort, see module docstring
            print(f"transcode: failed to generate {tier} rendition: {exc}")
            tmp_file.unlink(missing_ok=True)

    return done
