"""Computer-vision score reading: crops the user-marked scoreboard region
near the end of every rally and works out who won each rally and where
each game/set boundary sits, purely from *when a side's digit changes* -
not from needing a clean, simultaneous read of both digits together. This
is the one method of the four in score.py that doesn't reuse teams.py's
ball-out-of-bounds heuristic - everything downstream (game grouping, then
mapping a side to a named team) reuses score.py's existing machinery.

The left and right digits are tracked independently precisely because they
usually *can't* both be read together reliably: a hand, a glare, a referee
walking past, or just motion blur regularly obscures one side while the
other stays legible. Requiring both at once (the original approach) threw
away every one of those partial reads; tracking each side's own last-known
value means a single-side read is still useful - if it differs from what
that side last showed, that side just won the point, full stop, whether or
not we also caught the other side's digit in the same instant. Whatever
raw digits *do* get read are still saved on each rally as a reference
(RallyWinnerOut.cv_left/cv_right) even though the winner-determination
logic above no longer strictly depends on having both - useful for a human
spot-checking a stretch of uncertain rallies, without pretending the exact
number itself must be known.

The reading engine is easyocr, a general-purpose text reader - not a
digit-specific model. Two pretrained digit-detection models were tried
first (a seven-segment/LCD-display detector, then a handwritten-digit
detector) and both failed outright on real footage from this project: 0
detections and 1 wrong detection respectively, tested against a physical
flip-tile scoreboard. General OCR turned out to be the right tool after
all, once given real preprocessing - see _preprocess below - since "read
printed digits in a photo" is what it's actually built for, unlike either
specialized model's training domain.

The score region is a single fixed box for the whole video (typical
broadcast/fixed-camera overlays, and this project's own physical
flip-tile scorer, don't move). Each detected digit run is assigned to the
left or right side by its own horizontal position within the region, not
by "leftmost of however many were found" - that's what lets a single
legible digit still count even when the other side wasn't read at all.
Which physical side of the *frame* the left half corresponds to is
assumed to line up with the calibrated court's near/far net-crossing
split (side A = left, side B = right) - the same "sideline view"
assumption CourtSummary/teams.py already documents for the geometric
Team A/B split.
"""

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import score

# How many points to sample near a rally's end, and how far back that
# window reaches. Sampling has to stay anchored to this end-of-rally
# window rather than spread across the whole rally - the scoreboard
# usually updates a beat *after* the ball is actually dead, so a mid-rally
# read would just catch the *previous* point's score, not the one that
# just happened. A small, fixed number of evenly-spread points within that
# window (not every frame in it) is what keeps this fast: several chances
# for OCR to land on a clean, unobstructed read (a transient overlay,
# motion blur, or a referee's hand can spoil any single frame) without
# scaling with the video's actual frame rate.
SAMPLE_COUNT = 5
SAMPLE_WINDOW_S = 1.2


def _sample_offsets() -> list[float]:
    if SAMPLE_COUNT <= 1:
        return [0.0]
    return [SAMPLE_WINDOW_S * (1 - i / (SAMPLE_COUNT - 1)) for i in range(SAMPLE_COUNT)]


# Scoreboard digits are small in a full broadcast/gym-camera frame - OCR
# does much better on a large, high-contrast crop than on the raw pixels.
# CLAHE (adaptive local contrast) rather than a flat threshold copes better
# with uneven gym lighting across the region without blowing out
# highlights. Padding the crop was tried and made things *worse* in
# testing (more background context, more false-positive digit-like text),
# so the crop stays exactly the user-marked region.
UPSCALE_FACTOR = 6
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        import torch

        _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    return _reader


def _read_frame(cap: cv2.VideoCapture, timestamp_s: float, region: dict) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000)
    success, frame = cap.read()
    if not success:
        return None

    x, y, w, h = int(region["x"]), int(region["y"]), int(region["width"]), int(region["height"])
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    return crop if crop.size > 0 else None


def _preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    big = cv2.resize(
        crop_bgr,
        (crop_bgr.shape[1] * UPSCALE_FACTOR, crop_bgr.shape[0] * UPSCALE_FACTOR),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(gray)


def _ocr_crop(image: Optional[np.ndarray]) -> tuple[Optional[int], Optional[int]]:
    """Reads the left- and right-side digit groups from an already-cropped
    scoreboard image, independently - each is None if that side's digits
    weren't legible in this particular sample, rather than the whole
    reading being discarded just because the *other* side wasn't. Side is
    decided by each detection's horizontal center within the crop (left
    half vs right half), not by "leftmost of however many were found" -
    that's what lets a single legible digit still count even when the
    other side wasn't read at all (a hand, a glare, a referee walking past
    one half of the board).

    Only detections that are ALREADY a clean run of digits (no internal
    whitespace or other characters) count - a detected block containing a
    space is almost always OCR merging both digit tiles into one spurious
    reading (observed repeatedly in testing, e.g. "0 0" appearing as an
    extra detection alongside two correct standalone "5" and "0" reads),
    not a genuine multi-digit number. A real two-digit score reads as one
    clean run ("12"), not two characters with a gap between them.

    Takes an image rather than a timestamp/region - see compute_cv, which
    decodes every frame it needs up front and only calls this (the actual
    model-inference step) afterward, over all of them in one pass."""
    if image is None:
        return None, None

    preprocessed = _preprocess(image)
    detections = _get_reader().readtext(preprocessed)
    mid_x = preprocessed.shape[1] / 2

    left_candidates: list[tuple[float, int]] = []
    right_candidates: list[tuple[float, int]] = []
    for bbox, text, _confidence in detections:
        if not re.fullmatch(r"\d+", text):
            continue
        center_x = sum(p[0] for p in bbox) / len(bbox)
        value = int(text)
        (left_candidates if center_x < mid_x else right_candidates).append((center_x, value))

    # If a side somehow has more than one digit-run detected (stray noise -
    # a jersey number, a clock, a scoreboard label), keep the one closest
    # to that side's own edge of the region, since a genuine score digit
    # sits at the outer edge of its half, not near the middle.
    left = min(left_candidates, key=lambda c: c[0])[1] if left_candidates else None
    right = max(right_candidates, key=lambda c: c[0])[1] if right_candidates else None
    return left, right


def _modal_side(values: list[Optional[int]]) -> Optional[int]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _existing_baseline_before(existing: Optional[dict], rally_index: int) -> tuple[int, int]:
    """Reconstructs what the running (x, y) score would read at the start
    of rally_index, purely from previously-determined winners - lets a
    range-scoped re-run pick up mid-game without needing to re-read
    anything before the range. Resets to (0, 0) at the start of whichever
    game rally_index falls in, exactly like the main scoring loop below
    does on a real game-break reading."""
    if existing is None:
        return (0, 0)

    game_index = next(
        (g["game_index"] for g in existing["games"] if g["start_rally_index"] <= rally_index <= g["end_rally_index"]),
        None,
    )
    if game_index is None:
        return (0, 0)

    x = y = 0
    for r in existing["rallies"]:
        if r["rally_index"] >= rally_index:
            break
        if r["game_index"] != game_index:
            continue
        if r["winner"] == "x":
            x += 1
        elif r["winner"] == "y":
            y += 1
    return (x, y)


def compute_cv(output_path: Path, video_path: Optional[Path], rally_range: Optional[tuple[int, int]] = None) -> dict:
    """rally_range, if given *and* there's an existing result to seed a
    starting score from, restricts which rallies actually get re-read from
    the video at all - not just which ones the output can change. A
    narrow range (e.g. "Games 3-4" out of ten) used to still sample every
    rally in the whole match regardless, which defeated the point of
    specifying a range in the first place. The rallies before the range
    are instead replayed from the existing result (_existing_baseline_
    before) to reconstruct the running score the range starts from, since
    each stored winner is just +1 for one side - that gives the same
    starting point a fresh read would have found, without touching the
    video for them. A first-ever run (no existing result yet) still reads
    the whole match, since there's nothing to seed a mid-match baseline
    from."""
    if video_path is None:
        raise ValueError("No uploaded video found for this job")

    cfg = score.load_config(output_path)
    region = cfg.get("ocr_region")
    if not region:
        raise ValueError("No scoreboard region has been marked yet")

    rallies = score.load_rallies(output_path)
    existing = score.load_result(output_path)

    # Which geometric side (see attribute_sides_per_game) the left-read
    # digit actually belongs to - the "sideline view" assumption in this
    # module's docstring can be backwards for some camera angles/region
    # placements, so this is user-adjustable rather than hardcoded.
    left_side, right_side = ("B", "A") if cfg.get("cv_reverse_direction") else ("A", "B")

    if rally_range is not None and existing is not None:
        target_rallies = [r for r in rallies if rally_range[0] <= r["rally_index"] <= rally_range[1]]
        seed_baseline = _existing_baseline_before(existing, rally_range[0])
    else:
        target_rallies = rallies
        seed_baseline = (0, 0)

    offsets = _sample_offsets()

    # Phase 1: decode every needed frame up front (video seek/read only,
    # no model inference) and hold the crops in memory, all through one
    # shared VideoCapture rather than reopening the video per read - that
    # reopen was the single biggest cost here, far more than the
    # seek+decode+OCR that follows it. Phase 2 then runs OCR over all of
    # them in one pass. Splitting decode from inference like this (instead
    # of decoding one frame, OCRing it, decoding the next, ...) means the
    # video decoder and the OCR reader each just run straight through their
    # own work instead of constantly trading off.
    crops: list[Optional[np.ndarray]] = []
    crop_indexes: dict[int, tuple[int, int]] = {}  # rally_index -> (start, end) into crops
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("Could not open video")
        for rally in target_rallies:
            start = len(crops)
            for offset in offsets:
                timestamp_s = max(rally["start_time_s"], rally["end_time_s"] - offset)
                crops.append(_read_frame(cap, timestamp_s, region))
            crop_indexes[rally["rally_index"]] = (start, len(crops))
    finally:
        cap.release()

    # Per rally: the modal left reading and modal right reading, aggregated
    # *independently* across this rally's samples - not the modal (left,
    # right) *pair*. A sample that only caught one side still contributes
    # to that side's own modal value, instead of the whole sample being
    # thrown out for not having both.
    readings: dict[int, tuple[Optional[int], Optional[int]]] = {}
    for rally in target_rallies:
        start, end = crop_indexes[rally["rally_index"]]
        samples = [_ocr_crop(crop) for crop in crops[start:end]]
        left_val = _modal_side([s[0] for s in samples])
        right_val = _modal_side([s[1] for s in samples])
        readings[rally["rally_index"]] = (left_val, right_val)

    split_after: set[int] = set()
    winners: dict[int, Optional[str]] = {}
    uncertain: set[int] = set()

    # last_left/last_right are each side's own last-known digit, tracked
    # independently - a side's winner-worthy "change" only needs *that
    # side's* new reading to differ from what it last showed, regardless
    # of whether the other side was legible in the same rally at all. The
    # exact number/size of the jump is never used to decide *who* won,
    # only *that* one side moved and the other (as far as we can tell)
    # didn't - the raw values are still saved per rally as a reference
    # (see cv_left/cv_right below), just not relied on for this.
    last_left, last_right = seed_baseline
    for rally in target_rallies:
        left_val, right_val = readings[rally["rally_index"]]

        # A side's reading dropping below what it last showed means a new
        # game started somewhere before this rally - the physical/digital
        # scoreboard doesn't go backwards otherwise.
        reset = (left_val is not None and left_val < last_left) or (
            right_val is not None and right_val < last_right
        )
        if reset:
            prev_index = rally["rally_index"] - 1
            if prev_index >= 0:
                split_after.add(prev_index)
            last_left, last_right = 0, 0

        left_changed = left_val is not None and left_val != last_left
        right_changed = right_val is not None and right_val != last_right

        if left_changed and not right_changed:
            winners[rally["rally_index"]] = left_side
        elif right_changed and not left_changed:
            winners[rally["rally_index"]] = right_side
        else:
            # Neither side changed, both appeared to (only one side can
            # genuinely score a rally, so that's more likely a misread
            # than a real double-change), or nothing was legible at all -
            # none of those can be confidently attributed to either side.
            winners[rally["rally_index"]] = None
            uncertain.add(rally["rally_index"])

        if left_val is not None:
            last_left = left_val
        if right_val is not None:
            last_right = right_val

    if rally_range is None:
        merged_split_after = split_after
    else:
        lo, hi = rally_range
        existing_boundaries = {g["end_rally_index"] for g in existing["games"][:-1]} if existing and existing["games"] else set()
        # Boundaries outside the requested range are carried over as-is;
        # only the ones inside it are allowed to come from this fresh pass.
        merged_split_after = {b for b in existing_boundaries if not (lo <= b <= hi)}
        merged_split_after |= {b for b in split_after if lo <= b <= hi}

    games = score.build_games_from_boundaries(rallies, merged_split_after)
    attribution = score.attribute_sides_per_game(output_path, rallies, games, cfg["team_x_id"], cfg["team_y_id"])

    game_by_rally = {}
    for game in games:
        for idx in range(game["start_rally_index"], game["end_rally_index"] + 1):
            game_by_rally[rallies[idx]["rally_index"]] = game["game_index"]

    existing_by_rally = {r["rally_index"]: r for r in existing["rallies"]} if existing else {}

    result_rallies = []
    for rally in rallies:
        rally_index = rally["rally_index"]
        game_index = game_by_rally[rally_index]
        in_range = rally_range is None or (rally_range[0] <= rally_index <= rally_range[1])

        if not in_range and rally_index in existing_by_rally:
            previous = existing_by_rally[rally_index]
            result_rallies.append({**previous, "game_index": game_index})
            continue

        ab_winner = winners.get(rally_index)
        game_attribution = attribution.get(game_index, {})
        winner = game_attribution.get(ab_winner) if ab_winner else None
        confidence = "uncertain" if (rally_index in uncertain or winner is None) else "auto"
        cv_left, cv_right = readings.get(rally_index, (None, None))
        result_rallies.append({
            "rally_index": rally_index,
            "game_index": game_index,
            "winner": winner,
            "confidence": confidence,
            # The raw digits actually read for this rally, saved purely as
            # a reference for a human spot-checking an uncertain stretch -
            # the winner above never depends on these being complete or
            # even present (see the change-detection loop).
            "cv_left": cv_left,
            "cv_right": cv_right,
        })

    return score.save_result(output_path, {"games": games, "rallies": result_rallies})
