"""Reads a video file's own recording-date metadata, for defaulting the
upload flow's "date played" field (see jobs_router.upload_video) without
requiring the user to type it in for every file.

Deliberately dependency-free rather than shelling out to ffprobe - this
codebase avoids adding ffmpeg as an external binary dependency wherever
possible (see Backend/Analysis/PostProcessing/transcode.py's own docstring),
and reading one box out of an MP4/MOV container doesn't need it. Only
.mp4/.mov are supported (both use the same ISO-BMFF/QuickTime box format);
.avi/.mkv callers should expect None and fall back to today's date the same
as any file whose metadata doesn't have this tag at all.
"""

from __future__ import annotations

import struct
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Optional



def _read_box_header(f: BinaryIO) -> Optional[tuple[str, int, int]]:
    """(box_type, payload_size, header_size) for the box starting at the
    file's current position, or None at EOF. Handles the 64-bit
    extended-size case (size field == 1)."""
    header = f.read(8)
    if len(header) < 8:
        return None
    size, box_type = struct.unpack(">I4s", header)
    box_type = box_type.decode("ascii", errors="replace")
    if size == 1:
        ext = f.read(8)
        if len(ext) < 8:
            return None
        (size,) = struct.unpack(">Q", ext)
        return box_type, size - 16, 16
    if size == 0:
        # "extends to end of file" - not meaningful for our purposes, and
        # box_type here is never one we're walking into (moov/mvhd are
        # always explicitly sized in practice), so just stop.
        return None
    return box_type, size - 8, 8


def _find_child_box(f: BinaryIO, end: int, target: str) -> Optional[tuple[int, int]]:
    """(payload_offset, payload_size) for `target` within [f.tell(), end),
    or None. Returning the size here too (rather than making the caller
    re-derive it by seeking back and re-reading the header) sidesteps
    having to know target's own header size - which varies between 8 and
    16 bytes depending on whether it used a 64-bit extended size."""
    while f.tell() < end:
        start = f.tell()
        header = _read_box_header(f)
        if header is None:
            return None
        box_type, payload_size, header_size = header
        if box_type == target:
            return start + header_size, max(payload_size, 0)
        f.seek(start + header_size + max(payload_size, 0))
    return None


def _parse_mvhd_creation_time(f: BinaryIO, mvhd_payload_offset: int) -> Optional[int]:
    f.seek(mvhd_payload_offset)
    version_flags = f.read(4)
    if len(version_flags) < 4:
        return None
    version = version_flags[0]
    if version == 1:
        data = f.read(8)
        if len(data) < 8:
            return None
        (creation_time,) = struct.unpack(">Q", data)
    else:
        data = f.read(4)
        if len(data) < 4:
            return None
        (creation_time,) = struct.unpack(">I", data)
    return creation_time


def extract_creation_date(video_path: Path) -> Optional[date]:
    """The video's own recording date from its container metadata, or None
    if the file isn't MP4/MOV, has no readable moov/mvhd box, or its
    creation_time is the common "never stamped" sentinel of 0 (which would
    otherwise decode to 1904-01-01 - not a real date, just an absent one).

    Walking all the way to the file's actual end when needed (a
    non-"faststart" export can put moov after a multi-GB mdat - this is the
    common case, not a rare one, for footage straight off a camera/phone
    that hasn't been through a web-optimization pass) is fine performance-
    wise: this only ever reads small box headers and seeks past payloads by
    their declared size, never reads a payload's own bytes, so the cost is
    one seek + one small read per top-level box regardless of how large
    mdat is."""
    if video_path.suffix.lower() not in (".mp4", ".mov"):
        return None
    try:
        with video_path.open("rb") as f:
            end = video_path.stat().st_size
            moov = _find_child_box(f, end, "moov")
            if moov is None:
                return None
            moov_offset, moov_payload_size = moov
            f.seek(moov_offset)
            mvhd = _find_child_box(f, moov_offset + moov_payload_size, "mvhd")
            if mvhd is None:
                return None
            mvhd_offset, _ = mvhd
            creation_time = _parse_mvhd_creation_time(f, mvhd_offset)
    except (OSError, struct.error):
        return None

    if not creation_time:
        return None
    try:
        return (datetime(1904, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=creation_time)).date()
    except (OverflowError, OSError):
        return None
