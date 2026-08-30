"""
Player tracking, done as a two-phase offline batch job rather than a live,
frame-by-frame decision process. Deciding each player's identity live (as an
earlier version of this pipeline did) means deciding with only what happened
*before* the current frame - which forces velocity extrapolation, trust
thresholds and lost-player grace periods just to guess through a gap with no
way to look past it. Processing a whole pre-recorded video has no such
constraint, so tracking is split into two passes instead:

  1. Detect + fuse every frame (reusing tracker.py's 5-tracker ensemble and
     fusion), but only chain detections into short local tracklets using the
     raw trackers' own frame-to-frame IDs. No identity decision is made yet,
     so a tracklet simply ends the instant its trackers stop agreeing
     (occlusion, a crossing, anything) - fragmentation is expected and fine
     here.
  2. Once the whole video has been seen, consolidate those tracklets into
     final player identities by matching fragments against each other using
     their full appearance galleries and the gap between them - bridging a
     gap using what happens on *both* sides of it, not extrapolating
     forward from only one side.

Every call consolidates this video's tracklets from scratch - no identity
gallery is persisted across separate runs.
"""
import json
import time
from pathlib import Path
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from PlayerDetection.tracker import (
    TRACKERS,
    MODEL_PATH,
    APPEARANCE_GALLERY_SIZE,
    APPEARANCE_WEIGHT,
    IDENTITY_MAX_AGE,
    fuse,
    aspect_ratio,
    shape_bin,
    bin_penalty,
    cosine_distance,
    centre_of,
    extract_crop,
    AppearanceEncoder,
    load_homography,
    load_court_polygon,
    pixel_to_court,
    identity_colour,
)

OUTPUT_VIDEO_NAME = "consolidated.mp4"
POSITIONS_LOG_NAME = "player_positions.json"
CANDIDATE_MATCHES_LOG_NAME = "player_candidate_matches.json"

# Appearance-only distance bar for flagging two *separate* final identities
# as "might be the same person" in the player review UI - pairs
# consolidate() did NOT merge, usually because the gap between them was too
# long or the cost's spatial component pushed it over, not because they
# look different. This was originally 0.6, which on real footage flagged
# something like every other pair in the video (dozens of hints for a
# 5-minute clip) - nowhere near "might be the same person", just "isn't
# wildly different", which isn't a useful signal. Tightened to roughly
# MERGE_MAX_COST's own bar: below this, two players plausibly look like the
# same person; above it, the appearance match is too weak to be worth a
# human's time.
CANDIDATE_MAX_APPEARANCE_DISTANCE = 0.45

# A 1-2 frame tracklet is almost always jitter that survived fuse()'s own
# 2-tracker agreement check by luck; anything shorter than this never gets a
# chance to become a real player identity.
MIN_TRACKLET_FRAMES = 2

# Cost bar a pair of tracklets must clear to be merged into one person.
# Consolidation compares whole galleries (up to APPEARANCE_GALLERY_SIZE
# samples per side, nearest-neighbour) instead of tracker.py's single live
# frame, so it can afford to be stricter than tracker.py's MAX_MATCH_COST
# (0.55) without losing recall.
MERGE_MAX_COST = 0.45

# Largest gap (in frames) consolidation will bridge between two fragments -
# matches tracker.py's IDENTITY_MAX_AGE so both approaches forget a person
# after the same real-world absence.
MAX_MERGE_GAP_FRAMES = IDENTITY_MAX_AGE

# How much a fragment's boundary position is allowed to have drifted per
# frame of gap before the spatial term stops trusting it - same scale
# tracker.py's live position cost uses.
POSITION_RELAXATION_FRAMES = 25


def _store_embedding(tracklet, embedding, box):
    if embedding is None:
        return

    bucket = shape_bin(aspect_ratio(box))
    gallery = tracklet["embeddings"].setdefault(bucket, [])
    gallery.append(embedding)

    if len(gallery) > APPEARANCE_GALLERY_SIZE:
        gallery.pop(0)


def _new_tracklet(box, ids, embedding, frame_idx):
    tracklet = {"keys": set(ids.items()), "entries": [(frame_idx, box)], "embeddings": {}}
    _store_embedding(tracklet, embedding, box)

    return tracklet


def _extend_tracklet(tracklet, box, ids, embedding, frame_idx):
    tracklet["keys"] = set(ids.items())
    tracklet["entries"].append((frame_idx, box))
    _store_embedding(tracklet, embedding, box)


def _unpack_tracks(tracks):
    """
    A tracker's raw update() return: an (N, 8) array of
    [x1, y1, x2, y2, track_id, score, cls, idx] rows (idx isn't used here -
    it's the source detection's row, meaningful only to the caller that
    still holds that detection array). Converts to the (boxes, ids) shape
    fuse() expects, matching tracker.py's unpack() for an Ultralytics
    Results object.
    """
    if tracks is None or len(tracks) == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=int)

    return np.asarray(tracks[:, :4], dtype=float), np.asarray(tracks[:, 4], dtype=int)


def _advance_tracklets(open_tracklets, fused, embeddings, frame_idx):
    """
    One frame of pure local chaining: a tracklet only continues when a
    detection shares a raw tracker id with it. Every open tracklet not
    extended this frame closes, and every detection that didn't extend one
    starts a new tracklet - no identity decision happens here, that's
    entirely _consolidate()'s job once the whole video has been seen.
    """
    candidate_pairs = []

    for oi, tracklet in enumerate(open_tracklets):
        for di, (_, ids, _) in enumerate(fused):
            overlap = sum(1 for key in ids.items() if key in tracklet["keys"])
            if overlap:
                candidate_pairs.append((overlap, oi, di))

    candidate_pairs.sort(key=lambda p: -p[0])

    matched_open = {}
    used_detections = set()

    for overlap, oi, di in candidate_pairs:
        if oi in matched_open or di in used_detections:
            continue
        matched_open[oi] = di
        used_detections.add(di)

    closed = [t for oi, t in enumerate(open_tracklets) if oi not in matched_open]

    next_open = []
    for oi, di in matched_open.items():
        box, ids, _ = fused[di]
        tracklet = open_tracklets[oi]
        _extend_tracklet(tracklet, box, ids, embeddings[di], frame_idx)
        next_open.append(tracklet)

    for di, (box, ids, _) in enumerate(fused):
        if di not in used_detections:
            next_open.append(_new_tracklet(box, ids, embeddings[di], frame_idx))

    return next_open, closed


def _gallery_distance(gallery_a, gallery_b):
    """
    Nearest-neighbour cosine distance between two tracklets' embedding
    galleries - same shape-bucket logic as tracker.py's identity matching,
    preferring same-bucket comparisons but allowing cross-bucket ones with a
    mismatch penalty rather than refusing to compare at all.
    """
    best = None

    for bucket_a, embeddings_a in gallery_a.items():
        for bucket_b, embeddings_b in gallery_b.items():

            if not embeddings_a or not embeddings_b:
                continue

            penalty = bin_penalty(bucket_a, bucket_b)

            for ea in embeddings_a:
                for eb in embeddings_b:
                    dist = min(1.0, cosine_distance(ea, eb) + penalty)
                    if best is None or dist < best:
                        best = dist

    return best


def _bridge(cluster_a, cluster_b):
    """
    None if the two clusters' time spans overlap anywhere - they were on
    screen simultaneously as distinct detections, so they can never be the
    same person. Otherwise (gap_frames, pos_a, pos_b) for the closest pair of
    fragment boundaries across the two clusters, since a cluster that has
    already absorbed earlier merges may itself span several disjoint spans.

    Fast path: once a cluster has absorbed many merges, its own span list
    can run to dozens of entries, and consolidate() calls this for every
    remaining candidate after every single merge. If cluster_a entirely
    precedes cluster_b (its latest span ends before cluster_b's earliest
    one starts), that pair of edges IS the closest possible pair by
    construction - no per-span comparison can beat it - so that case
    resolves in O(1).

    Slow path: real match footage has two different players both on screen
    across nearly the whole video, so their overall ranges interleave far
    more often than a synthetic well-separated test suggests - meaning this
    is hit a lot, not just as a rare fallback. A naive scan of every
    (span_a, span_b) pair is O(len(spans_a) * len(spans_b)) and was still
    the dominant cost on real data even with the fast path above. Since
    both span lists are kept sorted by start frame (see _consolidate and
    _merge_into), the same problem "find the closest pair of disjoint
    sorted intervals, or detect a real overlap" is solvable in a single
    merge-style sweep: O(len(spans_a) + len(spans_b)).
    """
    if cluster_a["end_frame"] < cluster_b["start_frame"]:
        gap = cluster_b["start_frame"] - cluster_a["end_frame"]
        return gap, centre_of(cluster_a["end_pos"]), centre_of(cluster_b["start_pos"])

    if cluster_b["end_frame"] < cluster_a["start_frame"]:
        gap = cluster_a["start_frame"] - cluster_b["end_frame"]
        return gap, centre_of(cluster_b["end_pos"]), centre_of(cluster_a["start_pos"])

    spans_a, spans_b = cluster_a["spans"], cluster_b["spans"]
    i, j = 0, 0
    best = None

    while i < len(spans_a) and j < len(spans_b):
        a_start, a_end = spans_a[i]
        b_start, b_end = spans_b[j]

        if a_start <= b_end and b_start <= a_end:
            return None  # a genuine overlap - disqualified, no matter the rest

        if a_end < b_start:
            gap = b_start - a_end
            if best is None or gap < best[0]:
                best = (gap, cluster_a["by_frame"][a_end], cluster_b["by_frame"][b_start])
            # spans_b[j] has the smallest start left in B, so this WAS
            # spans_a[i]'s best possible pairing - it can only get a worse
            # gap against any later (larger-start) span in B, so move on.
            i += 1
        else:
            gap = a_start - b_end
            if best is None or gap < best[0]:
                best = (gap, cluster_b["by_frame"][b_end], cluster_a["by_frame"][a_start])
            j += 1

    if best is None:
        return None

    gap, pos_a, pos_b = best
    return gap, centre_of(pos_a), centre_of(pos_b)


def _pair_cost(cluster_a, cluster_b, diagonal):
    """Merge cost for two clusters, or None if they can't/shouldn't merge."""

    bridge = _bridge(cluster_a, cluster_b)
    if bridge is None or bridge[0] > MAX_MERGE_GAP_FRAMES:
        return None

    gap, pos_a, pos_b = bridge

    appearance = _gallery_distance(cluster_a["embeddings"], cluster_b["embeddings"])
    if appearance is None:
        return None

    raw_distance = np.linalg.norm(pos_a - pos_b) / diagonal
    spatial = min(1.0, raw_distance / (1.0 + gap / POSITION_RELAXATION_FRAMES))

    cost = APPEARANCE_WEIGHT * appearance + (1.0 - APPEARANCE_WEIGHT) * spatial

    return cost if cost <= MERGE_MAX_COST else None


def _merge_into(cluster_a, cluster_b):
    cluster_a["entries"] = sorted(cluster_a["entries"] + cluster_b["entries"], key=lambda e: e[0])
    # _bridge's sweep over spans depends on both lists being sorted by start
    # frame - each side is already sorted coming in, so this is a cheap
    # one-time sort, not the repeated O(n*m) scan that used to dominate.
    cluster_a["spans"] = sorted(cluster_a["spans"] + cluster_b["spans"])
    cluster_a["by_frame"].update(cluster_b["by_frame"])

    if cluster_b["start_frame"] < cluster_a["start_frame"]:
        cluster_a["start_frame"] = cluster_b["start_frame"]
        cluster_a["start_pos"] = cluster_b["start_pos"]

    if cluster_b["end_frame"] > cluster_a["end_frame"]:
        cluster_a["end_frame"] = cluster_b["end_frame"]
        cluster_a["end_pos"] = cluster_b["end_pos"]

    for bucket, embeddings in cluster_b["embeddings"].items():
        gallery = cluster_a["embeddings"].setdefault(bucket, [])
        gallery.extend(embeddings)

        if len(gallery) > APPEARANCE_GALLERY_SIZE:
            # Even subsample rather than chronological truncation, so the
            # kept samples still represent the merged cluster's full span
            # instead of collapsing onto whichever fragment merged last.
            keep = sorted(set(np.linspace(0, len(gallery) - 1, APPEARANCE_GALLERY_SIZE).round().astype(int)))
            gallery[:] = [gallery[i] for i in keep]


def _consolidate(tracklets, diagonal):
    """
    Greedily merges the globally cheapest compatible pair of tracklets,
    repeating until no pair is left under MERGE_MAX_COST. Global-cheapest-
    first (rather than a single left-to-right pass) is what lets this use
    the whole video's context: a fragment picks its best match anywhere in
    the video, not just the nearest one in time.
    """
    clusters = []

    for t in tracklets:
        if len(t["entries"]) < MIN_TRACKLET_FRAMES:
            continue

        start_frame, start_box = t["entries"][0]
        end_frame, end_box = t["entries"][-1]

        clusters.append({
            "entries": list(t["entries"]),
            "embeddings": {bucket: list(embs) for bucket, embs in t["embeddings"].items()},
            "spans": [(start_frame, end_frame)],
            "by_frame": dict(t["entries"]),
            "start_frame": start_frame,
            "start_pos": start_box,
            "end_frame": end_frame,
            "end_pos": end_box,
        })

    alive = list(range(len(clusters)))
    cost_cache = {}

    for a in range(len(alive)):
        for b in range(a + 1, len(alive)):
            cost = _pair_cost(clusters[alive[a]], clusters[alive[b]], diagonal)
            if cost is not None:
                cost_cache[(alive[a], alive[b])] = cost

    while cost_cache:
        (i, j), cost = min(cost_cache.items(), key=lambda kv: kv[1])

        # Printed so a merge can be checked against the annotated video by
        # frame number - a cost near MERGE_MAX_COST was accepted but wasn't a
        # confident match, and is the first place to look if two different
        # players ever end up sharing one id.
        flag = " <- borderline, worth checking" if cost > MERGE_MAX_COST * 0.7 else ""
        print(f"consolidate: merging frames {clusters[j]['spans']} into "
              f"{clusters[i]['spans']} (cost {cost:.3f}/{MERGE_MAX_COST}){flag}")

        _merge_into(clusters[i], clusters[j])
        clusters[j] = None
        alive.remove(j)

        for key in [k for k in cost_cache if i in k or j in k]:
            del cost_cache[key]

        for k in alive:
            if k == i:
                continue
            lo, hi = (i, k) if i < k else (k, i)
            cost = _pair_cost(clusters[lo], clusters[hi], diagonal)
            if cost is not None:
                cost_cache[(lo, hi)] = cost

    return [clusters[k] for k in alive]


def _find_candidate_matches(clusters):
    """Pairs of *separate* final identities whose appearance galleries are
    still fairly similar even though consolidate() didn't merge them -
    surfaced to the player review UI as a "might be the same person" hint
    rather than an automatic merge, since a false merge silently corrupts
    stats while a false hint just costs the user one glance. A pair whose
    time spans overlap is skipped outright: they were on screen
    simultaneously, so they provably can't be the same person no matter how
    similar they look."""
    candidates = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if _bridge(clusters[i], clusters[j]) is None:
                continue

            distance = _gallery_distance(clusters[i]["embeddings"], clusters[j]["embeddings"])
            if distance is None or distance > CANDIDATE_MAX_APPEARANCE_DISTANCE:
                continue

            candidates.append((i, j, distance))

    return candidates


def _annotate(frame, entries):
    for stable_id, box in entries:
        x1, y1, x2, y2 = (int(v) for v in box)
        colour = identity_colour(stable_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        label = f"P{stable_id}"
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        cv2.rectangle(frame, (x1, y1 - size[1] - 8), (x1 + size[0] + 6, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def trackplayers_offline(video_path, output_path, show_preview=False, save_video=False):
    """
    See module docstring for the two-phase design. show_preview/save_video
    only apply to phase 3 (the final replay with consolidated ids) - phase 1
    has nothing stable to show yet, since identity isn't decided until the
    whole video has been seen.
    """
    tracker_names = list(TRACKERS)

    # One shared model does the actual (expensive) detection forward pass -
    # model.track() normally couples detection to a single tracker instance,
    # but Ultralytics runs them as two separate internal steps (detect, then
    # hand the boxes to a tracker.update() call), so nothing requires
    # re-running detection per tracker. Verified byte-identical output to the
    # old one-model-per-tracker approach on real frames, at 2.6x the speed -
    # this was the single biggest cost in the whole pipeline (5 full YOLO26x
    # passes per frame instead of 1).
    detector = YOLO(MODEL_PATH)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diagonal = float(np.hypot(frame_width, frame_height))

    cuda_available = torch.cuda.is_available()
    track_device = 0 if cuda_available else "cpu"
    torch_device = torch.device("cuda:0" if cuda_available else "cpu")
    encoder = AppearanceEncoder(torch_device)

    # Each tracker algorithm is a plain object fed the shared detections
    # below - built once (matching persist=True's old effect: state carries
    # across frames) using the exact same config resolution model.track()
    # uses internally, so behaviour matches what each TRACKERS[name] yaml
    # always specified.
    trackers = {}
    for name in tracker_names:
        cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKERS[name])))
        cfg.device = torch_device
        trackers[name] = TRACKER_MAP[cfg.tracker_type](args=cfg)

    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", cuda_available)

    run_started = time.monotonic()

    # ---- Phase 1: detect + fuse every frame, chain into local tracklets ----
    phase1_started = time.monotonic()
    open_tracklets = []
    finished_tracklets = []

    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        detection = detector.predict(
            source=frame, classes=[0], conf=0.25, iou=0.45,
            device=track_device, verbose=False,
        )[0]
        det = detection.boxes.cpu().numpy()

        tracker_outputs = []
        for name in tracker_names:
            tracks = trackers[name].update(det, detection.orig_img, feats=None)
            tracker_outputs.append((name, *_unpack_tracks(tracks)))

        fused = fuse(tracker_outputs)

        boxes = [entry[0] for entry in fused]
        crops = [
            extract_crop(frame, box, exclude_boxes=[boxes[j] for j in range(len(boxes)) if j != i])
            for i, box in enumerate(boxes)
        ]
        embeddings = encoder.encode(crops)

        open_tracklets, closed = _advance_tracklets(open_tracklets, fused, embeddings, frame_idx)
        finished_tracklets.extend(closed)

        if frame_idx % 200 == 0:
            print(f"phase 1 - frame {frame_idx}: {len(fused)} detections, "
                  f"{len(open_tracklets)} open tracklets, {len(finished_tracklets)} finished")

        frame_idx += 1

    finished_tracklets.extend(open_tracklets)
    cap.release()

    phase1_elapsed = time.monotonic() - phase1_started
    print(f"Phase 1 done in {phase1_elapsed:.1f}s: {len(finished_tracklets)} raw tracklets over {frame_idx} frames")

    # ---- Phase 2: consolidate tracklets into final player identities ----
    phase2_started = time.monotonic()
    clusters = _consolidate(finished_tracklets, diagonal)
    clusters.sort(key=lambda c: c["entries"][0][0])

    # clusters[i] becomes stable_id i+1 below, so compute candidates against
    # this exact order - each pair's (a, b) is final stable_ids directly.
    candidate_matches = _find_candidate_matches(clusters)
    candidate_matches_file = Path(output_path) / CANDIDATE_MATCHES_LOG_NAME
    with open(candidate_matches_file, "w") as f:
        json.dump(
            [{"a": i + 1, "b": j + 1, "confidence": round(1.0 - distance, 3)} for i, j, distance in candidate_matches],
            f, indent=2,
        )

    phase2_elapsed = time.monotonic() - phase2_started
    print(f"Phase 2 done in {phase2_elapsed:.1f}s: consolidated into {len(clusters)} player identities, "
          f"{len(candidate_matches)} candidate same-person hint(s)")

    # ---- Phase 3: replay the video, writing the annotated output + log with final ids ----
    phase3_started = time.monotonic()
    frame_lookup = {}
    for stable_id, cluster in enumerate(clusters, start=1):
        for f_idx, box in cluster["entries"]:
            frame_lookup.setdefault(f_idx, []).append((stable_id, box))

    homography = load_homography(output_path)
    court_polygon = load_court_polygon(output_path)

    writer = None
    need_annotation = save_video or show_preview
    if save_video:
        writer = cv2.VideoWriter(
            str(Path(output_path) / OUTPUT_VIDEO_NAME),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height),
        )

    window_name = "Consolidated Tracking"
    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    positions_log = []
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        entries = frame_lookup.get(frame_idx, [])

        if need_annotation:
            if court_polygon is not None:
                cv2.polylines(frame, [court_polygon.astype(np.int32)],
                              isClosed=True, color=(255, 0, 255), thickness=2)

            _annotate(frame, entries)

            cv2.putText(frame, f"frame {frame_idx}   players {len(entries)}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if entries:
            frame_players = []
            for stable_id, box in entries:
                cx, cy = centre_of(box)
                court = pixel_to_court(cx, cy, homography) if homography is not None else None
                frame_players.append({
                    "stable_id": int(stable_id),
                    "box": [float(v) for v in box],
                    "pixel": [float(cx), float(cy)],
                    "court": list(court) if court is not None else None,
                })

            positions_log.append({
                "frame_idx": frame_idx,
                "timestamp_s": frame_idx / fps,
                "players": frame_players,
            })

        if save_video:
            writer.write(frame)

        if show_preview:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if save_video:
        writer.release()
        print(f"Consolidated tracking video saved: {Path(output_path) / OUTPUT_VIDEO_NAME}")
    if show_preview:
        cv2.destroyAllWindows()

    positions_log_file = Path(output_path) / POSITIONS_LOG_NAME
    with open(positions_log_file, "w") as f:
        json.dump(positions_log, f, indent=2)

    phase3_elapsed = time.monotonic() - phase3_started
    total_elapsed = time.monotonic() - run_started
    print(f"Phase 3 done in {phase3_elapsed:.1f}s")
    print(f"Total: {total_elapsed:.1f}s (phase 1: {phase1_elapsed:.1f}s, "
          f"phase 2: {phase2_elapsed:.1f}s, phase 3: {phase3_elapsed:.1f}s)")
    print(f"Player positions saved: {positions_log_file}")
