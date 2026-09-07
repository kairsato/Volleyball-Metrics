"""One-off converter: turns the CVAT-exported XML annotations in
game_status_dataset.zip (labels: serving-start / serving-end / in-game-end,
one tag per transition frame) into this project's own game_status_labels.json
format (see label.py) - remapping those labels to this project's own
service / play / no-play vocabulary and reusing label.py's own _export so
the output is byte-for-byte the same schema label.py itself writes.

Usage:
    python convert_game_status_dataset.py --xml-dir DIR --dataset-dir DIR
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from label import _export

LABEL_MAP = {
    "serving-start": "service",
    "serving-end": "play",
    "in-game-end": "no-play",
}

VIDEO_EXTENSIONS = (".mp4", ".webm")


def _xml_total_frames(root):
    task_or_job = root.find("meta/task")
    if task_or_job is None:
        task_or_job = root.find("meta/job")
    return int(task_or_job.findtext("size"))


def _extract_breakpoints(root):
    breakpoints = []
    unmapped = set()
    for image in root.findall("image"):
        tags = image.findall("tag")
        if not tags:
            continue
        label = tags[0].get("label")
        state = LABEL_MAP.get(label)
        if state is None:
            unmapped.add(label)
            continue
        breakpoints.append((int(image.get("id")), state))
    breakpoints.sort(key=lambda bp: bp[0])
    return breakpoints, unmapped


def convert_one(video_id, xml_dir, dataset_dir):
    xml_path = xml_dir / f"{video_id}.xml"
    folder = dataset_dir / video_id
    video_path = next((folder / f"{video_id}{ext}" for ext in VIDEO_EXTENSIONS
                        if (folder / f"{video_id}{ext}").exists()), None)
    if video_path is None:
        print(f"{video_id}: SKIP - no video found in {folder}")
        return

    root = ET.parse(xml_path).getroot()
    breakpoints, unmapped = _extract_breakpoints(root)
    if unmapped:
        print(f"{video_id}: WARNING - unmapped labels seen and skipped: {unmapped}")

    xml_frames = _xml_total_frames(root)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cv2_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    total_frames = xml_frames
    if cv2_frames and abs(cv2_frames - xml_frames) > 1:
        print(f"{video_id}: NOTE - XML declares {xml_frames} frames, "
              f"the video file itself decodes {cv2_frames}; using the XML's own "
              f"count since the breakpoint frame indices are relative to it.")

    _export(breakpoints, total_frames, fps, folder)
    print(f"{video_id}: {len(breakpoints)} points, fps={fps:.3f}, frames={total_frames} -> {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=["2", "3", "4", "6", "8", "9", "10", "13"])
    args = parser.parse_args()

    for video_id in args.ids:
        convert_one(video_id, args.xml_dir, args.dataset_dir)


if __name__ == "__main__":
    main()
