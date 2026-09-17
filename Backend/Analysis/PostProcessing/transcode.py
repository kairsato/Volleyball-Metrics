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

Never upscales: a downscale tier is only generated when its target height is
smaller than the source's own height, so a lower-resolution source doesn't
get a padded-out "480p" file that's really just a relabeled copy. The
untouched source video itself always doubles as the "original" quality tier -
no rendition file is ever written for that one.

Best-effort by design: this stage runs after the analysis pipeline's real
work is done, so a transcoding failure (corrupt frame, disk full, whatever)
should never fail the whole job over what's ultimately a nice-to-have -
every failure path here is caught and logged, never raised.
"""
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

# tier name -> (target height in pixels, capped average video bitrate in
# kbps). Exactly the qualities the Results page's selector offers alongside
# "original" (the untouched upload, never re-encoded - see module
# docstring). Each lower resolution gets a lower bitrate cap too, on top of
# needing fewer bits per pixel just from being smaller - a 480p rendition
# that was still allowed to spend 1080p-sized bits on a static wide shot like
# this would barely save anything over the tier above it. Order doesn't
# affect anything (each tier is encoded independently, in its own ffmpeg
# invocation).
TRANSCODE_TIERS: dict[str, tuple[int, int]] = {
    "1080p": (1080, 5000),
    "720p": (720, 2500),
    "480p": (480, 1200),
}

# Standard libx264 quality target - visually near-lossless for footage like
# this (a static wide shot, not high-motion close-ups), well-compressed.
# Combined with each tier's own maxrate/bufsize below (a "capped CRF"): CRF
# picks the actual bitrate scene-by-scene for consistent visual quality,
# while the cap only kicks in to clip rare spikes so a tier's file size stays
# predictable and each lower tier is genuinely lower-bitrate than the one
# above it, not just "lower resolution, same bits-per-pixel budget."
CRF = 23
# "veryfast" trades some compression efficiency for encode speed - this runs
# as a post-processing nice-to-have after the real analysis work is already
# done (see module docstring), so minutes of extra encode time for a
# marginally smaller file isn't worth it; "ultrafast" was skipped as too
# lossy a tradeoff the other direction.
PRESET = "veryfast"


def rendition_filename(tier: str) -> str:
    return f"source_{tier}.mp4"


def _encode_rendition(ffmpeg_exe: str, video_path, out_file: Path, target_height: int, maxrate_kbps: int, tier: str) -> bool:
    """Encodes one rendition, returning whether it now exists (already did,
    or was just successfully written). Written to a .tmp file first and
    renamed into place once ffmpeg exits cleanly, so a mid-encode crash/kill
    never leaves a partial file behind that a later "already exists" check
    would wrongly treat as a finished rendition."""
    if out_file.exists():
        return True

    tmp_file = out_file.with_suffix(".tmp.mp4")
    try:
        # scale=-2:H: keep the source aspect ratio, only constrain height -
        # "-2" rounds the computed width to the nearest even number, which
        # libx264's 4:2:0 chroma subsampling requires. maxrate/bufsize cap
        # this tier's peak bitrate (bufsize = 2x maxrate is the usual
        # capped-CRF rule of thumb) without abandoning CRF's per-scene
        # quality targeting - see CRF's own comment above.
        subprocess.run(
            [
                ffmpeg_exe, "-y", "-i", str(video_path),
                "-vf", f"scale=-2:{target_height}",
                "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
                "-maxrate", f"{maxrate_kbps}k", "-bufsize", f"{maxrate_kbps * 2}k",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(tmp_file),
            ],
            capture_output=True, text=True, check=True,
        )
        tmp_file.replace(out_file)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"transcode: failed to generate {tier} rendition: {exc.stderr.strip()[-2000:]}")
        tmp_file.unlink(missing_ok=True)
        return False
    except Exception as exc:  # noqa: BLE001 - best-effort, see module docstring
        print(f"transcode: failed to generate {tier} rendition: {exc}")
        tmp_file.unlink(missing_ok=True)
        return False


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

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    done = []

    for tier, (target_height, maxrate_kbps) in TRANSCODE_TIERS.items():
        if target_height < source_height and _encode_rendition(
            ffmpeg_exe, video_path, output_path / rendition_filename(tier), target_height, maxrate_kbps, tier
        ):
            done.append(tier)

    return done
