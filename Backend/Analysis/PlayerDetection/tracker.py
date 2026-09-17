"""
Player detection, tracking, and identity - detect (YOLO26x) + track
(BoT-SORT) + re-identify (an external ReID model), then consolidate into
stable player identities using volleyball-specific heuristics, all in one
module.

Tracking itself is done as a two-phase offline batch job rather than a live,
frame-by-frame decision process. Deciding each player's identity live (as an
earlier version of this pipeline did) means deciding with only what happened
*before* the current frame - which forces velocity extrapolation, trust
thresholds and lost-player grace periods just to guess through a gap with no
way to look past it. Processing a whole pre-recorded video has no such
constraint, so tracking is split into two stages, each its own top-level
function:

  1. track_and_chain() - detect (YOLO26x) + track (BoT-SORT) every frame,
     but only chain detections into short local tracklets using the
     tracker's own frame-to-frame ids. No identity decision is made yet, so
     a tracklet simply ends the instant BoT-SORT loses it (occlusion, a
     crossing, anything) or the appearance-continuity guard below judges a
     "continuation" to actually be a different person - fragmentation is
     expected and fine here. Saves its raw tracklets to TRACKLETS_RAW_NAME.
     Runs as a 3-stage threaded pipeline (read -> detect ->
     track+embed+chain) so one frame's decode+detect overlaps the previous
     frame's track+embed+chain instead of running strictly sequentially -
     see _run_pipeline.

  2. consolidate_tracklets() - once the whole video has been seen, matches
     those raw tracklets against each other by treating them as an explicit
     weighted graph: one node per tracklet, one edge per pair judged
     compatible, the edge weight a cost built from their full appearance
     galleries and the gap between them - with a pair disqualified outright
     if they're confidently on opposite sides of the court (see
     _consolidate/_pair_cost - real players physically cannot cross the net
     to the other team's side, so unlike appearance/spatial cost this is
     enforced as a hard rule, not a soft prior). Repeatedly
     contracting the globally cheapest edge - not deciding each tracklet's
     identity independently against a gallery - is what lets a fragment
     bridge a gap using what happens on *both* sides of it, and lets a
     confident match anywhere in the video outrank a merely-plausible one
     nearby in time. A roster-size pass then caps each side at
     MAX_PLAYERS_PER_SIDE simultaneously-active identities (see
     _enforce_roster_cap), re-merging over-fragmented excess where a relaxed
     cost bar allows it and flagging genuine leftovers (bench/staff who
     wandered into the service-area buffer) as likely_non_player rather than
     silently counting them as rostered players. That graph, and every merge
     it actually performed, is saved to IDENTITY_GRAPH_NAME so a bad merge
     (or a split that should have merged) can be inspected after the fact
     rather than guessed at. Then replays the source video with the final
     ids, optionally live (the "player identification preview" -
     show_preview) as well as to disk.

Splitting these into separate functions - rather than one monolithic pass -
means consolidate_tracklets can be re-run on its own (different matching
constants, preview on/off, etc.) straight from TRACKLETS_RAW_NAME, without
repeating track_and_chain's expensive detect+track+embed pass over every
frame. trackplayers_offline() at the bottom is a single-process convenience
wrapper running both stages back to back - that's what API/stage_runner.py's
"player_tracking" stage calls; RunEverything.py calls the two stages
separately so the preview can be toggled and consolidation re-tuned cheaply.
Every call consolidates this video's tracklets from scratch - no identity
gallery is persisted across separate runs (that's API/player_gallery.py's
job, a separate cross-video system built on top of this module's
AppearanceEncoder).
"""
import json
import queue
import threading
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from GameStatusDetection.rallyWindows import compute_track_windows

# YOLO26x alone does the detecting. Fusing a second detector (RF-DETR)
# into every frame was tried and measured against this on both a side-view
# and a back-view match, and removed again: it found no additional real
# players (identities tracked across most of the clip were unchanged at 17
# and 24-25 respectively), added only ~1-2% more raw detections, produced
# the same or MORE identity fragments after consolidation (18 -> 21 finals
# on the side view), and roughly doubled detection time. A second detector's
# extra boxes are overwhelmingly the marginal, heavily-occluded cases that
# fragment into short tracklets rather than new people.
MODEL_PATH = "yolo26x.pt"

# Longest side the detector actually runs at, however large the source frame
# is. Not a quality/speed tradeoff - at 4K it is faster AND better.
#
# This used to run at the frame's native size on the reasoning that full
# resolution gives the detector "a real shot at the distant players a
# downscaled pass would blur away". That holds at 1080p, where native IS
# about this size, and is wrong at 4K. Measured on a 51-minute 3840x2160
# recording:
#
#     imgsz            s/frame   projected   mean detections/frame
#     native (4K)       0.125      194 min          17.4
#     1920              0.031       48 min          22.2
#     1280              0.018       27 min          21.1
#
# Four times faster and finding MORE people, because a detector trained on
# roughly 640-1280px inputs sees a 4K person as far larger than anything in
# its training distribution. Capping the long side puts 4K footage at the
# same working resolution 1080p footage already had, so nothing regresses
# for the sizes this pipeline was tuned on - a 1080p frame is already under
# the cap and is passed through untouched.
#
# half=True and batching were measured on the same footage and did nothing
# (0.126 vs 0.125 s/frame, and 0.029 vs 0.031 at batch 4), so neither is
# used - this stage is not compute-bound in the way either of those helps.
MAX_INFERENCE_LONG_SIDE = 1920

# Bounded queue sizes for track_and_chain's read -> detect ->
# track+embed+chain pipeline (see _run_pipeline). Small on purpose: this is
# about overlapping consecutive frames' work, not buffering deep into the
# video - a large queue would just let one stage race ahead and hold whole
# BGR frames (several MB each) in memory for no wall-clock benefit.
FRAME_QUEUE_SIZE = 8
DETECTION_QUEUE_SIZE = 4

# Ultralytics' own pretrained person-ReID encoder (auto-downloaded from their
# release assets on first use) - see AppearanceEncoder for why this replaced
# a generic ImageNet classifier. "m" (medium) balances embedding quality
# against per-frame cost - benchmarked at ~54ms for a full 12-player frame
# batch on CPU (this pipeline's fallback - see AppearanceEncoder), which is
# comfortably inside an offline batch job's budget; "n"/"s" are faster but
# measurably less discriminating between similarly-dressed players, which is
# exactly the failure mode this swap exists to fix.
REID_MODEL_NAME = "yolo26m-reid.onnx"

# Tracker config passed to model.track. BoT-SORT alone (stock config, its own
# `with_reid: False` by default so it never duplicates the external
# AppearanceEncoder below) - this used to be one of a 5-tracker ensemble
# whose boxes were fused together per frame, which bought some resilience to
# any one tracker's jitter but at real complexity and runtime cost. Simpler
# now: one tracker's own raw id chains a tracklet locally (track_and_chain
# below), and cross-checking against reality is the appearance-continuity
# guard (_appearance_breaks_continuity) plus consolidate_tracklets's full-
# gallery graph matching - not agreement from four other trackers.
TRACKER_CONFIG = "botsort.yaml"

# Largest gap (in frames) consolidation (consolidate_tracklets below) will
# bridge between two fragments of the same person.
IDENTITY_MAX_AGE = 1800  # 60 sec @ 30fps

# Weight given to appearance vs. position when two fragments are being
# considered for the same identity.
# Split of the blended merge cost between how alike two clusters look, where
# on the court they were (ZONE_WEIGHT), and how far apart the two fragment
# boundaries are in pixels. Appearance no longer dominates: it is the weakest
# of the three at this crop size (see ZONE_COLUMNS' measurements and
# PHYSICAL_CONTINUITY_MAX_COURT_M's), and it keeps an absolute veto of its own
# in APPEARANCE_MAX_DISTANCE regardless of what this weighting says.
APPEARANCE_WEIGHT = 0.45

# The same split for an uncalibrated video, where there is no court and so no
# zone term to give that weight to. Left at what the two-way appearance/
# spatial blend used before zones existed, rather than letting the rebalance
# above silently hand appearance's share to the pixel-space spatial term in a
# case none of the zone measurements apply to.
APPEARANCE_WEIGHT_WITHOUT_ZONES = 0.85

# One embedding set per body shape, judged from the box width / height. Crouching,
# diving and upright players look different enough that mixing them hurts matching.
ASPECT_EDGES = (0.45, 0.70, 1.00)
CROSS_SHAPE_PENALTY = 0.2   # cost added when only another shape is stored

# How many recent embeddings each identity keeps per body-shape bucket.
# Matching against the nearest of several real snapshots discriminates
# between similar-looking players far better than blending them into one
# running average, which tends to blur different players' looks together.
APPEARANCE_GALLERY_SIZE = 10

# Which crops earn one of those gallery slots: the least crowded ones the
# tracklet ever saw, not the most recent.
#
# Almost every crop on real volleyball footage has another player in it even
# when the detector drew no box there - measured on a real match, the
# nearest other detection's box centre sits a median of 0.2 subject widths
# away and only ~2% of crops are isolated by any useful definition - and the
# ReID encoder is reading whoever else is in frame along with the subject.
# That contamination, not the encoder, is what makes appearance weak here:
# on the isolated crops the same encoder separates people at AUC 0.873
# against 0.744 across crops generally.
#
# Two things that look like fixes are not. Insetting or narrowing the crop
# to cut the neighbour out measured at AUC 0.733-0.760, i.e. no better than
# leaving it alone; and simply refusing crowded crops is not available,
# since at 2% isolated most tracklets would end up with no gallery at all.
# Choosing the best of what there is costs nothing and does work - keeping
# the least crowded N rather than the last N measured AUC 0.841 vs 0.819,
# and beats the even-subsample _merge_into used to do on 0.777.
#
# Isolation is measured as the distance to the nearest other detection's box
# centre in multiples of the subject's own box width, so it does not change
# meaning with camera distance.
GALLERY_ISOLATION_WHEN_ALONE = 99.0

# Name of the calibration file CourtDetection.court saves into output_path
COURT_FILE_NAME = "court.json"

# Court dimensions the saved homography's court-metre axes are built against
# (CourtDetection.court.COURT_LENGTH/COURT_WIDTH) - duplicated here rather
# than imported since PlayerDetection has no dependency on CourtDetection
# otherwise, and these are a fixed rule-book constant, not something that
# varies per video.
COURT_LENGTH_M = 18.0
COURT_WIDTH_M = 9.0

# How far behind each baseline (in court metres) a detection is still
# accepted as a player serving/receiving serve rather than a spectator,
# bench player, or staff member standing in the crowd/sideline area. A
# standard serving zone has no fixed depth limit (it runs to the end of the
# free zone), but real venues rarely give a server more than about this much
# room before hitting seating or a wall, so it's a practical rather than a
# rulebook bound. Width is never extended past the sidelines: the service
# zone is exactly as wide as the court, not wider.
SERVICE_AREA_DEPTH_M = 6.0

OUTPUT_VIDEO_NAME = "consolidated.mp4"
POSITIONS_LOG_NAME = "player_positions.json"
CANDIDATE_MATCHES_LOG_NAME = "player_candidate_matches.json"

# track_and_chain's raw output - every tracklet before any identity has been
# decided, plus the frame size/fps consolidate_tracklets needs. Lets
# consolidate_tracklets be re-run on its own later (see module docstring)
# instead of only ever running right after track_and_chain in the same
# process.
TRACKLETS_RAW_NAME = "player_tracklets_raw.json"

# Appearance embeddings live beside that file as raw float32 rather than
# inside it as decimal text. A 128-d embedding costs about 2.5 KB written
# out as JSON numbers and 512 bytes as float32, and json has to parse every
# one of those digits back into a Python float object on load.
#
# On a 51-minute 4K recording the combined file reached 5.0 GB, which took
# minutes to write and expanded to tens of GB of Python objects on the way
# back in - enough to exhaust a 62 GB machine before consolidation could
# start. Splitting the embeddings out leaves a JSON file of boxes that stays
# in the low hundreds of MB and an array that memory-maps straight into
# numpy.
TRACKLET_EMBEDDINGS_NAME = "player_tracklets_embeddings.npz"

# The tracklet-similarity graph consolidate_tracklets matched identities
# from - every compatible pair it considered, the merges it actually
# performed and in what order, and which raw tracks each final stable_id was
# built from. Written purely for inspection (e.g. why two tracklets that
# look alike didn't merge, or which raw BoT-SORT ids a player's stats come
# from) - nothing downstream reads it back in.
IDENTITY_GRAPH_NAME = "player_identity_graph.json"

# Distance bars for flagging two *separate* final identities as "might be
# the same person" in the player review UI - pairs consolidate() did NOT
# merge, usually because the gap between them was too long or the appearance
# gate was too tight, not because they are really different people. Both
# sit looser than the merge-time bars they mirror (APPEARANCE_MAX_DISTANCE
# and ZONE_MAX_DISTANCE): a pair between the two is one the evidence likes
# but not enough to merge unreviewed, which is exactly what a human hint is
# for. A false merge silently corrupts stats with no way back; a false hint
# costs the reviewer one glance and a click.
#
# Appearance can be this loose specifically BECAUSE the zone and side
# evidence now gate the hint too - it no longer has to carry the decision on
# its own. Measured on a real match with the zone/side gates in front of it,
# against the full different-person population:
#
#     appearance   zone     same-person caught   different-person admitted
#        0.30      0.85           58.2%                    0.0%
#        0.34      0.85           76.1%                    1.1%
#        0.34      0.95           82.1%                    1.1%   <- here
#        0.38      0.95           86.6%                   10.7%
#
# 0.34/0.95 is the knee: a 24-point recall gain over the old appearance-only
# 0.30 bar for about one false hint in ninety. The next step out costs ten
# times the error for five more points, which is where a hint stops being
# worth looking at.
#
# The zone bar in particular is looser here than for a merge because a
# player legitimately rotates between zones over a set, and a hint the
# reviewer rejects is cheap - measured on the three fragment groups of one
# real player that stayed separate under the merge-time 0.85, where the
# zone gate alone blocked 36 of the 86 cross-pairs and appearance the other
# 50. Loosening either one on its own reunited only part of him.
CANDIDATE_MAX_APPEARANCE_DISTANCE = 0.34
CANDIDATE_MAX_ZONE_DISTANCE = 0.95

# A 1-2 frame tracklet is almost always a stray detection BoT-SORT briefly
# picked up (a spectator's arm, a reflection) rather than a real person;
# anything shorter than this never gets a chance to become a real player
# identity.
MIN_TRACKLET_FRAMES = 2

# What fraction of a raw tracklet's own frames must put its foot point
# inside the play area (see is_in_play_area) for that tracklet to be
# eligible to become a player identity at all.
#
# is_in_play_area already gates detection in track_and_chain, but on a real
# job that gate is a no-op: court calibration is a Setup-tab step that runs
# AFTER player_tracking (see API/services/pipeline._phase_one), so the first
# pass has no homography and keeps everyone in frame - the crowd, the bench,
# people walking behind the court. Consolidation is re-run once the court
# IS known (stage_runner._recalibrate), so this is the point where the
# court-boundary rule can actually be applied, and applying it here - rather
# than only auto-ignoring the results afterwards in
# API/services/players.recalibrate_players - is what stops off-court people
# becoming identities at all. Measured on a 2:30 clip that produced 1340
# identities for 12 players on court: 2287 of its 2569 raw tracklets never
# had a single in-play frame, and gating them out here cut the final count
# to 197.
#
# A fraction rather than "in play at least once", for the same reason
# recalibrate_players uses MIN_IN_RANGE_FRACTION: a spectator in the front
# row leaning towards the sideline clips the play area for a handful of
# frames across a whole video, while a real player - even one chasing a wide
# ball into the crowd - is on court for the overwhelming majority of their
# screen time, so half is a bar the two cannot both clear.
MIN_IN_PLAY_FRACTION = 0.5

# Hard appearance gate: two clusters this far apart on _gallery_distance's
# averaged scale are never merged, whatever the rest of the cost says.
#
# Calibrated against measured data rather than picked by feel. Using labels
# that need no human (two halves of one unbroken tracklet are the same
# person; two tracklets overlapping in time are different people), the
# accept/reject tradeoff on a real match runs:
#
#     bar    correct merges kept    wrong merges allowed
#     0.22          48.5%                   0.7%
#     0.26          71.3%                   4.2%
#     0.30          92.4%                  19.0%
#     0.45         100.0%                  98.9%   <- the old effective bar
#
# 0.22 is deliberately the precision end of that curve. The two failure
# modes are not symmetrical: a wrong merge silently fuses two people into
# one identity and corrupts their stats with no way for a reviewer to
# unpick it, while a missed merge just leaves an extra fragment to name -
# and naming two fragments the same already reunites them downstream (see
# API/services/players.build_canonical_mapping). So this errs toward more
# fragments over any chance of mixing people.
APPEARANCE_MAX_DISTANCE = 0.22

# When two clusters share frames, how small and how box-coincident that
# overlap has to be to read as one person detected twice rather than two
# people on screen together (see _is_duplicate_detection). A real
# double-detection is a handful of frames with the boxes nearly on top of
# each other; the measured case that motivated this was 5 frames (0.33% of
# the shorter cluster) averaging 0.48 IoU, while the same match's genuinely
# different-player small overlaps averaged 0.06 and 0.00 IoU - so the IoU
# bar, not the frame count, is what actually separates the two cases.
DUPLICATE_OVERLAP_MAX_FRAMES = 30       # 1 second @ 30fps
DUPLICATE_OVERLAP_MAX_FRACTION = 0.02
DUPLICATE_OVERLAP_MIN_IOU = 0.40

# How a cluster's court occupancy is summarised for _zone_distance: each
# side of the net split into front/back row and three columns along it -
# the six volleyball rotation positions, twice. Players hold a rotation
# slot for a stretch of play rather than roaming the whole court, so where
# someone stood is real evidence about who they are, and unlike appearance
# it does not degrade as the gap between two fragments grows.
#
# Measured with the same label-free protocol as the appearance bars (two
# halves of one tracklet separated by a synthetic gap are the same person;
# two tracklets that overlap in time are provably different people), at the
# long gaps where the physical-continuity path below no longer applies and
# appearance was previously the only signal left:
#
#     gap     zone bar 0.70            zone bar 0.85
#      2s     83.5% kept / 7.7% wrong  88.2% kept / 12.5% wrong
#     10s     77.8% kept / 7.1% wrong  83.3% kept / 11.3% wrong
#     20s     74.1% kept / 12.7% wrong 81.5% kept / 14.5% wrong
#
# Compare appearance at its own 0.22 gate: 48.5% kept. Zone occupancy holds
# up across a 20-second gap where appearance is close to random, which is
# exactly the re-linking case that was fragmenting one player into many.
ZONE_COLUMNS = 3
ZONE_FRONT_ROW_DEPTH_M = 4.5      # attack line to net, i.e. the front row

# Absolute gate: two clusters whose court occupancy overlaps less than this
# are never merged and never offered as the same person. Deliberately set at
# the generous end (0.85, not the 0.30 that would maximise precision on the
# table above) because rotation is a real thing that moves a player between
# zones legitimately - this is here to reject the pairs that share no court
# at all, like a left-back and a right-front who were never in the same
# place, not to insist a player never moved.
ZONE_MAX_DISTANCE = 0.85

# How much the graded zone distance counts in the blended merge cost. Large
# on purpose: on the measurements above it is the better discriminator of
# the two at every gap length past a second or so, so the cost ordering
# should be led by where people were, with appearance refining rather than
# deciding.
ZONE_WEIGHT = 0.40

# Cost added when two clusters' dominant sides disagree but not confidently
# enough for _pair_cost's outright side veto to fire. Real players almost
# never change sides mid-set - measured here at 100.0% side agreement for
# same-person pairs against 45.7% for different-person pairs, which makes
# side the single cleanest signal available - but the veto only fires when
# BOTH clusters clear SIDE_MIN_CONFIDENCE, so below that the evidence used
# to be discarded entirely rather than merely softened. Scaled by how
# confident the weaker of the two actually is, so a cluster hovering at the
# net contributes almost nothing and a nearly-certain one contributes
# nearly all of it.
SIDE_MISMATCH_PENALTY = 0.35

# Physical-continuity fast path: a gap this short with the two fragments'
# foot points this close together on the court is merged WITHOUT having to
# clear APPEARANCE_MAX_DISTANCE, because nobody can teleport - a person
# standing 1 metre away half a second later is the same person whatever a
# noisy 128-d embedding thinks of the two crops.
#
# This exists because appearance alone is the weaker evidence by a wide
# margin at this crop size, and gating on it was throwing away the stronger
# evidence the tracker already had. Measured on a real match with labels
# that need no human (two halves of one unbroken tracklet separated by a
# synthetic gap are the same person; two tracklets that overlap in time are
# provably different people, cut so they no longer overlap):
#
#     rule                              real continuations kept   wrong merges
#     appearance <= 0.22                        48.5%                 0.7%
#     court <= 1.0m over <= 15 frames            94.4%                 0.4%
#
# i.e. roughly twice the recall at a lower error rate. The measured 0.4% is
# also an overestimate of what happens in situ: those negatives had to have
# their real time overlap cut away to be testable at all, and _bridge still
# disqualifies any genuinely overlapping pair outright, so two players who
# are ever on court together can never reach this path.
#
# Both bars are deliberately at the knee of that curve rather than past it.
# Widening the gap is what costs accuracy: at 60 frames the same 1.0m bar
# keeps only 49.5% of real continuations, because a player covers real
# ground in two seconds - so the gap stays short and the distance bar stays
# tight instead of trading one for the other.
PHYSICAL_CONTINUITY_MAX_GAP_FRAMES = 15   # 0.5 sec @ 30fps
PHYSICAL_CONTINUITY_MAX_COURT_M = 1.0

# Cost band the physical-continuity path scores into, scaled by how close
# the two foot points actually were. Sits entirely below the lowest cost any
# appearance-driven merge can reach (APPEARANCE_WEIGHT * the smallest
# measured gallery distance, ~0.16) so _consolidate's global-cheapest-first
# loop always spends the certain merges before the merely plausible ones,
# and one fragment can chain through several hops of physical certainty
# before appearance gets a say.
PHYSICAL_CONTINUITY_MAX_COST = 0.05

# Cost bar a pair of tracklets must clear to be merged into one person -
# ordering only now that APPEARANCE_MAX_DISTANCE is the real gate. Kept on
# the same scale as the appearance/spatial blend it sums (see _pair_cost) so
# a pair that clears the appearance gate isn't then rejected for a merely
# mediocre spatial term.
MERGE_MAX_COST = 0.45

# Real players never cross the net to the other team's side - the x = 9.0m
# midpoint of the homography's court-metre axes (see create_half_court_
# homography). Matches API/services/teams.py's own NET_X_M, duplicated here
# rather than imported for the same reason COURT_LENGTH_M/COURT_WIDTH_M are:
# PlayerDetection has no dependency on the API layer otherwise, and this is a
# fixed rule-book constant, not something that varies per video.
NET_X_M = COURT_LENGTH_M / 2.0

# A cluster's side (see _cluster_side) only counts as "confidently" on one
# side of the court if at least this fraction of its tracked frames agree -
# below that (mostly hovering near the net, or too few frames to judge) it's
# treated as unknown and exempt from both the side-mismatch disqualification
# and the roster cap, rather than guessing.
SIDE_MIN_CONFIDENCE = 0.7

# Indoor volleyball fields 6 players per side at once. Enforced as a
# simultaneous-activity cap (_enforce_roster_cap), not a total-distinct-
# identities-per-video cap, so a libero swap or mid-match substitution (whose
# tracklet only starts once the player they replaced is off court) is never
# penalised - only a side that has more than 6 identities genuinely
# overlapping in time is over the limit.
MAX_PLAYERS_PER_SIDE = 6

# Looser cost bar _enforce_roster_cap allows when trying to re-merge a side's
# excess identities into one of its primary MAX_PLAYERS_PER_SIDE - this pass
# only runs on a side that's already over the roster cap, so a merge here is
# recovering a real player MERGE_MAX_COST's stricter bar over-fragmented,
# not a first-pass matching decision. Note this only relaxes the blended
# cost: APPEARANCE_MAX_DISTANCE is an absolute gate that no pass loosens, so
# the roster cap can never buy a merge between two people who don't look
# alike just because a side is over capacity.
RELAXED_MERGE_MAX_COST = 0.65

# How much of its own tracked life an identity must spend as a side's
# "extra" (beyond MAX_PLAYERS_PER_SIDE) one before it's called a
# likely_non_player rather than a real player caught in a momentary overlap.
# Measured on a real match: over-capacity moments there were 44 separate
# bursts, none lasting even a second, totalling 4.5% of frames - so a plain
# "was it ever over capacity" test flagged identities with 1400 good tracked
# frames (7-8% of their life over capacity) right alongside genuine
# artefacts, and left other identities with an indistinguishable 6-7%
# unflagged, which is close to arbitrary. The real interlopers on that
# footage sat at 42% and 61%, so anything in that gap separates them
# cleanly; 0.25 keeps a wide margin on both sides rather than hugging
# either population.
NON_PLAYER_MIN_EXCESS_FRACTION = 0.25

# Largest gap (in frames) consolidation will bridge between two fragments -
# matches IDENTITY_MAX_AGE so both the local chaining and the global
# consolidation forget a person after the same real-world absence.
MAX_MERGE_GAP_FRAMES = IDENTITY_MAX_AGE

# How much a fragment's boundary position is allowed to have drifted per
# frame of gap before the spatial term stops trusting it.
POSITION_RELAXATION_FRAMES = 25

# A raw tracker briefly latching onto the wrong physical person after a
# crossing or occlusion - a well-known failure mode for any tracker, not
# just a rare glitch - lets an open tracklet silently absorb a different
# person's crops without ever closing, since _advance_tracklets below only
# needs BoT-SORT's own raw id to still match. Left unchecked, that makes one
# stable_id genuinely be two different real players across the video (the
# "same number, sometimes the wrong person" symptom) and can just as easily
# block a later, correct consolidation merge because the poisoned gallery no
# longer looks like either person. Looser than APPEARANCE_MAX_DISTANCE since
# an early tracklet's gallery is small and noisier than a full-video
# comparison - this only needs to catch a clear mismatch, not fine-grained
# matching (that's still consolidate_tracklets's job), and a false break
# here just costs an extra fragment. Sits above the measured 95th percentile
# of genuine same-person distance (0.372) on _gallery_distance's averaged
# scale; it was 0.7 when that function returned a nearest-neighbour distance
# instead, which on this scale would essentially never fire.
APPEARANCE_CONTINUITY_MAX_DISTANCE = 0.45
MIN_GALLERY_FOR_CONTINUITY_CHECK = 3


def load_homography(output_path):
    """
    Load the pixel -> real-world-court-metres transform CourtDetection.court
    saved for this video. Returns None if no calibration has been saved yet -
    player positions are then logged in pixel coordinates only.
    """
    court_file = Path(output_path) / COURT_FILE_NAME

    if not court_file.exists():
        return None

    try:
        with open(court_file) as f:
            data = json.load(f)

        matrix = np.array(data["homography"], dtype=np.float64)

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(f"Could not read court calibration from {court_file} ({e}); "
              f"player positions will be logged in pixels only.")
        return None

    return matrix


def pixel_to_court(x, y, matrix):
    point = np.array([[[x, y]]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(point, matrix)

    return float(transformed[0][0][0]), float(transformed[0][0][1])


def foot_point(box):
    """Bottom-centre of a detection box - where a standing player actually
    touches the ground, which is what the flat pixel<->court homography
    (a ground-plane-only mapping, see create_half_court_homography) expects.
    The box centre used elsewhere for logged positions sits roughly at
    waist height, which the homography would misread as a ground point
    further from the camera than the player really is - fine for a rough
    position log, but exactly the kind of error that would wrongly exclude
    or include a player near the in-court/service-area boundary."""
    x1, y1, x2, y2 = box
    return float((x1 + x2) / 2.0), float(y2)


def is_in_play_area(box, homography):
    """
    Whether a detection's foot point falls inside the court or its service
    area behind either baseline - see SERVICE_AREA_DEPTH_M. Used to keep
    player detection limited to people actually playing, rather than
    coaches, bench players, referees, ball kids and spectators standing
    further back or to the sides.

    Returns True (no filtering) when there's no homography yet - without a
    calibrated court there's no way to tell in-play from off-court, and
    refusing to detect anyone would be worse than detecting everyone (see
    load_court_polygon's docstring for the same tradeoff on the drawn
    overlay).
    """
    if homography is None:
        return True

    x, y = pixel_to_court(*foot_point(box), homography)

    return (
        -SERVICE_AREA_DEPTH_M <= x <= COURT_LENGTH_M + SERVICE_AREA_DEPTH_M
        and 0.0 <= y <= COURT_WIDTH_M
    )


def in_play_fraction(entries, homography):
    """
    What fraction of a tracklet's own frames put it inside the play area -
    see is_in_play_area for the boundary itself and MIN_IN_PLAY_FRACTION for
    what _consolidate does with this.

    Returns 1.0 (nothing is off-court) when there's no homography, matching
    is_in_play_area's own "no calibration -> don't filter" tradeoff: without
    a calibrated court there's no way to tell a player from a spectator, and
    dropping everyone would be far worse than keeping everyone.
    """
    if homography is None or not entries:
        return 1.0

    in_play = sum(1 for _frame_idx, box in entries if is_in_play_area(box, homography))

    return in_play / len(entries)


def load_court_polygon(output_path):
    """
    Load the four court corners CourtDetection.court saved for this video, as
    a polygon in frame pixel coordinates. Used only to draw the court boundary
    overlay on the annotated video - is_in_play_area is what actually decides
    which detections get tracked (court + service area behind each baseline),
    this is purely visual.

    Returns None (no overlay drawn) if no calibration has been saved yet.
    """
    court_file = Path(output_path) / COURT_FILE_NAME

    if not court_file.exists():
        return None

    try:
        with open(court_file) as f:
            data = json.load(f)

        image_points = [(p["x"], p["y"]) for p in data["image_points"]]

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
        print(f"Could not read court boundary from {court_file} ({e}); "
              f"tracking the full frame instead.")
        return None

    if len(image_points) != 4:
        return None

    return np.array(image_points, dtype=np.float32)


def aspect_ratio(box):

    x1, y1, x2, y2 = box

    return float(x2 - x1) / max(1.0, float(y2 - y1))


def shape_bin(ratio):
    """Upright, crouched and diving players get their own embedding."""

    return sum(ratio >= edge for edge in ASPECT_EDGES)


def bin_penalty(stored_bin, current_bin):
    """Cost of comparing embeddings taken at different body shapes."""

    steps = abs(stored_bin - current_bin)

    return CROSS_SHAPE_PENALTY * steps / len(ASPECT_EDGES)


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def centre_of(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _box_overlap(a, b):
    """Intersection-over-union of two boxes - 1.0 when they coincide, 0.0
    when they don't touch. Used to tell one person detected twice from two
    people standing near each other (see _is_duplicate_detection)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _count_sides(entries, homography):
    """How many of a tracklet's frames fall on each side of the net (foot
    point's court-x vs. NET_X_M) - the raw material _cluster_side reduces to
    a single dominant side + confidence. {"A": 0, "B": 0} (no opinion, never
    penalised or capped) when there's no calibration to judge by."""
    counts = {"A": 0, "B": 0}

    if homography is None:
        return counts

    for _frame_idx, box in entries:
        x, _y = pixel_to_court(*foot_point(box), homography)
        counts["A" if x < NET_X_M else "B"] += 1

    return counts


def _count_zones(entries, homography):
    """A tracklet's court occupancy as raw counts over the ZONE_COLUMNS-wide
    front/back grid on each side of the net - the six volleyball rotation
    positions, twice. Counts rather than a normalised histogram so merging
    two clusters is a plain element-wise add, the same way side_counts is
    kept up to date incrementally by _merge_into instead of being recomputed
    from entries inside _pair_cost's O(n^2) scan.

    All zeroes when there's no calibration, which _zone_distance reads as
    "no opinion" rather than "nowhere in common".
    """
    counts = [0] * (2 * 2 * ZONE_COLUMNS)

    if homography is None:
        return counts

    for _frame_idx, box in entries:
        x, y = pixel_to_court(*foot_point(box), homography)
        side = 0 if x < NET_X_M else 1
        depth = 0 if abs(x - NET_X_M) < ZONE_FRONT_ROW_DEPTH_M else 1
        column = int(min(ZONE_COLUMNS - 1, max(0, y / (COURT_WIDTH_M / ZONE_COLUMNS))))
        counts[side * 2 * ZONE_COLUMNS + depth * ZONE_COLUMNS + column] += 1

    return counts


def _zone_distance(cluster_a, cluster_b):
    """How little two clusters' court occupancy overlaps, in [0, 1] - 0 when
    they spent their time in exactly the same mix of zones, 1 when they
    never shared a zone at all. Histogram intersection of the two normalised
    occupancy profiles (see _count_zones).

    None when either side has no calibrated positions to judge by, so
    callers can fall back to appearance alone rather than treat "unknown" as
    "nothing in common".
    """
    total_a, total_b = sum(cluster_a["zone_counts"]), sum(cluster_b["zone_counts"])
    if total_a == 0 or total_b == 0:
        return None

    overlap = sum(
        min(a / total_a, b / total_b)
        for a, b in zip(cluster_a["zone_counts"], cluster_b["zone_counts"])
    )

    return 1.0 - overlap


def _cluster_side(cluster):
    """A cluster's dominant court side and the fraction of its tracked
    frames that agree with it - (None, 0.0) if it has no side_counts opinion
    at all (no calibration). Only ever reads cluster["side_counts"], which is
    seeded once per raw tracklet in _consolidate and kept up to date
    incrementally by _merge_into - not recomputed from entries here, since
    this is called repeatedly inside _pair_cost's O(n^2) candidate scan and a
    cluster's entries can run to thousands of frames."""
    counts = cluster["side_counts"]
    total = counts["A"] + counts["B"]

    if total == 0:
        return None, 0.0

    side = "A" if counts["A"] >= counts["B"] else "B"

    return side, counts[side] / total


def _cluster_duration(cluster):
    """Total frames a cluster was actually tracked across all its spans -
    used by _enforce_roster_cap to judge which identities on an over-capacity
    side have the strongest evidence of being a real, primary player."""
    return sum(end - start + 1 for start, end in cluster["spans"])


def cosine_distance(a, b):
    """
    Distance in [0, 1] between two L2-normalised embeddings (0 = identical).

    Clamped with plain min/max rather than np.clip - np.clip is built for
    array inputs and its dispatch overhead (type checks, ufunc machinery)
    dwarfs the cost of clamping a single float. Profiling an offline
    consolidation run (tens of millions of calls, comparing every candidate
    pair's embedding galleries) found this one substitution was ~92% of that
    run's total time - np.clip on a scalar, not any algorithmic cost.
    """

    if a is None or b is None:
        return 1.0

    similarity = max(-1.0, min(1.0, float(np.dot(a, b))))

    return max(0.0, min(1.0, (1.0 - similarity) / 2.0))


def extract_crop(frame, box, exclude_boxes=None):
    """
    Crop a player's box, muting any pixels claimed by a neighbouring detection
    so a crowded frame doesn't bleed one player's appearance into another.
    Muted pixels are replaced with the patch's own average color rather than
    black: a solid black block creates a sharp artificial edge the CNN
    responds strongly to, which made occluded players look more like each
    other (they'd share the same black-block feature) instead of less. Trims
    a thin sliver off the top/bottom to cut down on ball motion blur and
    floor bleed.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_h = max(1.0, y2 - y1)

    top = int(max(0, min(height - 1, y1 + box_h * 0.03)))
    bottom = int(max(0, min(height, y2 - box_h * 0.02)))
    left = int(max(0, min(width - 1, x1)))
    right = int(max(0, min(width, x2)))

    if bottom - top < 8 or right - left < 8:
        return None

    patch = frame[top:bottom, left:right].copy()

    exclude_mask = np.zeros(patch.shape[:2], dtype=bool)

    for ex_box in (exclude_boxes or []):
        ex_x1, ex_y1, ex_x2, ex_y2 = ex_box

        ox1 = int(max(left, ex_x1)) - left
        oy1 = int(max(top, ex_y1)) - top
        ox2 = int(min(right, ex_x2)) - left
        oy2 = int(min(bottom, ex_y2)) - top

        if ox2 > ox1 and oy2 > oy1:
            exclude_mask[oy1:oy2, ox1:ox2] = True

    if exclude_mask.any() and not exclude_mask.all():
        fill_colour = patch[~exclude_mask].mean(axis=0)
        patch[exclude_mask] = fill_colour

    return patch


class AppearanceEncoder:
    """
    Batched appearance feature extractor used to tell visually similar players
    apart.

    Wraps Ultralytics' own pretrained person-ReID encoder
    (ultralytics.trackers.utils.reid.ReID, the same encoder BoT-SORT can use
    internally - see TRACKER_CONFIG's own note on why that internal path
    (`with_reid`) stays off in favour of this external one). This
    replaced an ImageNet-pretrained ResNet18 used purely as a fixed feature
    extractor: that model was trained to recognise object CATEGORIES, never to
    tell two individual people apart, so plain classifier features regularly
    failed to distinguish teammates wearing identical kits - the biggest
    source of identity swaps in this pipeline. REID_MODEL_NAME's encoder is
    instead trained via metric learning specifically for "is this the same
    person", which is a much closer match to what this pipeline actually
    needs.

    Falls back to CPU automatically (Ultralytics' own AutoBackend prints a
    warning and retries) if no working CUDA execution provider is found for
    onnxruntime. This used to be the normal path on this project's RTX 5090 -
    not because the GPU itself couldn't run it, but because the venv had both
    `onnxruntime` (CPU-only) and `onnxruntime-gpu` pip-installed at once; the
    two packages install into the exact same `onnxruntime/` module path, so
    whichever was installed second silently won regardless of what
    requirements.txt asked for, and CUDAExecutionProvider wasn't even offered.
    `pip uninstall onnxruntime onnxruntime-gpu` followed by a clean
    `pip install onnxruntime-gpu` (matching requirements.txt, which only ever
    listed the GPU package) fixes it - confirmed via
    onnxruntime.get_available_providers() actually listing
    CUDAExecutionProvider afterward, and this encoder measurably running on
    cuda:0. CPU inference for a whole frame's worth of player crops still
    comfortably fits this pipeline's offline-batch budget in a pinch (see
    REID_MODEL_NAME's own benchmark note) if that conflict ever reappears -
    nowhere near the tighter budget a live/real-time tracker would need - but
    GPU cuts a meaningful chunk off this module's own track_and_chain runtime
    (~39% less wall-clock time on a real test clip), so it's worth checking
    `onnxruntime.get_available_providers()` if player tracking ever feels
    unusually slow again.
    """

    def __init__(self, device):
        from ultralytics.trackers.utils.reid import ReID

        self.device = device
        # ReID wants an int GPU index (or "cpu"), not a torch.device - this
        # pipeline's own device values are either a torch.device("cuda:0")/
        # ("cpu") (track_and_chain below) or the bare string "cpu"
        # (API/players.py, API/player_gallery.py), so normalise both to what
        # it expects.
        if isinstance(device, torch.device):
            reid_device = device.index if device.type == "cuda" else "cpu"
        else:
            reid_device = 0 if str(device).startswith("cuda") else "cpu"

        self._reid = ReID(REID_MODEL_NAME, device=reid_device)

    def _prepare(self, crop):
        size = self._reid.imgsz
        # A distant player's crop is usually far smaller than the model's
        # input, so this is normally an upsample - INTER_CUBIC preserves more
        # of what little detail is there than INTER_LINEAR's flatter blur.
        interpolation = cv2.INTER_CUBIC if crop.shape[0] < size or crop.shape[1] < size else cv2.INTER_AREA
        resized = cv2.resize(crop, (size, size), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return np.transpose(rgb, (2, 0, 1))

    @torch.no_grad()
    def encode(self, crops):
        """Batch-encode a list of BGR crops (any entry may be None). Returns a
        list of L2-normalised embeddings aligned with the input, with None
        wherever the crop was missing or too small to embed."""

        valid = [i for i, c in enumerate(crops) if c is not None and c.size > 0]

        results = [None] * len(crops)

        if not valid:
            return results

        batch = np.stack([self._prepare(crops[i]) for i in valid])
        tensor = torch.from_numpy(batch).to(self._reid.device)

        # Most of these ReID exports use a dynamic batch dimension (no fixed
        # size to respect), but a handful of static ONNX exports don't - see
        # ultralytics.trackers.utils.reid.ReID.__call__, whose chunk-and-pad
        # handling for that case is mirrored here.
        bs, n = self._reid.batch_size, tensor.shape[0]
        if bs is None or n == bs:
            features = self._reid.model(tensor)
        else:
            outs = []
            for s in range(0, n, bs):
                chunk = tensor[s:s + bs]
                if chunk.shape[0] < bs:
                    chunk = torch.cat([chunk, chunk[-1:].expand(bs - chunk.shape[0], *chunk.shape[1:])], 0)
                outs.append(self._reid.model(chunk))
            features = torch.cat(outs, 0)[:n]

        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(np.asarray(features))
        features = F.normalize(features, dim=1)
        features = features.cpu().numpy()

        for slot, i in enumerate(valid):
            results[i] = features[slot]

        return results


def identity_colour(stable_id):

    rng = np.random.default_rng(stable_id * 9781)

    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def detection_isolation(box, other_boxes):
    """How much room a detection has to itself: the distance to the nearest
    other detection's box centre, in multiples of this box's own width (so
    it means the same thing for a near player and a distant one). Higher is
    cleaner - see GALLERY_ISOLATION_WHEN_ALONE for why the gallery cares."""
    centre_x = (box[0] + box[2]) / 2.0
    width = max(1.0, box[2] - box[0])

    gaps = [
        abs((other[0] + other[2]) / 2.0 - centre_x) / width
        for other in other_boxes
        if other is not box
    ]

    return min(gaps) if gaps else GALLERY_ISOLATION_WHEN_ALONE


def _store_embedding(tracklet, embedding, box, isolation):
    """Adds one (embedding, isolation) snapshot to the tracklet's gallery for
    this body-shape bucket, evicting the MOST crowded snapshot once the
    bucket is full rather than the oldest - see GALLERY_ISOLATION_WHEN_ALONE.
    A tracklet that only ever saw crowded crops still keeps a full gallery of
    its own least-bad ones; nothing is ever dropped for being crowded in
    absolute terms."""
    if embedding is None:
        return

    bucket = shape_bin(aspect_ratio(box))
    gallery = tracklet["embeddings"].setdefault(bucket, [])
    gallery.append((embedding, float(isolation)))

    if len(gallery) > APPEARANCE_GALLERY_SIZE:
        # By index, not list.remove: an entry is (numpy embedding, float),
        # and remove() compares earlier entries with == before it reaches the
        # one to drop, which on a numpy array raises rather than returning a
        # bool.
        most_crowded = min(range(len(gallery)), key=lambda i: gallery[i][1])
        del gallery[most_crowded]


def _new_tracklet(box, track_id, embedding, frame_idx, isolation):
    tracklet = {"track_id": track_id, "entries": [(frame_idx, box)], "embeddings": {}}
    _store_embedding(tracklet, embedding, box, isolation)

    return tracklet


def _extend_tracklet(tracklet, box, track_id, embedding, frame_idx, isolation):
    tracklet["track_id"] = track_id
    tracklet["entries"].append((frame_idx, box))
    _store_embedding(tracklet, embedding, box, isolation)


def _unpack_tracks(tracks):
    """
    BoT-SORT's raw update() return: an (N, 8) array of
    [x1, y1, x2, y2, track_id, score, cls, idx] rows (idx isn't used here -
    it's the source detection's row, meaningful only to the caller that
    still holds that detection array). Converts to the (boxes, ids) shape
    the rest of this module works with.
    """
    if tracks is None or len(tracks) == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=int)

    return np.asarray(tracks[:, :4], dtype=float), np.asarray(tracks[:, 4], dtype=int)


def _advance_tracklets(open_tracklets, detections, embeddings, frame_idx):
    """
    One frame of pure local chaining: a tracklet only continues when a
    detection shares BoT-SORT's own track_id with it AND (once it has enough
    of a gallery to judge) still looks like the same person - see
    _appearance_breaks_continuity. A raw track_id is unique to at most one
    active track at a time, so this is a direct lookup rather than the
    overlap-scoring/greedy-assignment a multi-tracker ensemble used to need.
    Every open tracklet not extended this frame closes, and every detection
    that didn't extend one starts a new tracklet - no identity decision
    happens here, that's entirely consolidate_tracklets's job once the whole
    video has been seen.
    """
    by_track_id = {t["track_id"]: oi for oi, t in enumerate(open_tracklets)}

    matched_open = {}
    used_detections = set()

    for di, (box, track_id) in enumerate(detections):
        oi = by_track_id.get(track_id)
        if oi is None:
            continue
        if _appearance_breaks_continuity(open_tracklets[oi], embeddings[di], box):
            continue
        matched_open[oi] = di
        used_detections.add(di)

    closed = [t for oi, t in enumerate(open_tracklets) if oi not in matched_open]

    # How much room each detection had to itself this frame, so the gallery
    # can keep the cleanest crops rather than the newest ones.
    frame_boxes = [box for box, _track_id in detections]
    isolations = [detection_isolation(box, frame_boxes) for box in frame_boxes]

    next_open = []
    for oi, di in matched_open.items():
        box, track_id = detections[di]
        tracklet = open_tracklets[oi]
        _extend_tracklet(tracklet, box, track_id, embeddings[di], frame_idx, isolations[di])
        next_open.append(tracklet)

    for di, (box, track_id) in enumerate(detections):
        if di not in used_detections:
            next_open.append(_new_tracklet(box, track_id, embeddings[di], frame_idx, isolations[di]))

    return next_open, closed


def _gallery_distance(gallery_a, gallery_b):
    """
    MEAN cosine distance across two tracklets' embedding galleries -
    preferring same-bucket comparisons but allowing cross-bucket ones with a
    mismatch penalty rather than refusing to compare at all.

    This used to take the nearest-neighbour (minimum) distance instead,
    which quietly destroyed the entire appearance signal. Measured on a real
    match, using labels no human had to supply (two embeddings from one
    unbroken tracklet are the same person; two tracklets overlapping in time
    are provably different people): individual embedding pairs separate
    same-from-different with AUC 0.843, but taking the MINIMUM over a
    10x10 gallery pair means the one closest pair decides, and across
    different people that minimum dipped under the accept bar 100% of the
    time. The gate was accepting every pair put to it - correct merges and
    wrong ones alike - so identities were really being decided by the
    no-overlap rule in _bridge and a weak spatial term, not by appearance at
    all. That is what let unrelated people chain together (A~B, B~C) into
    one identity: 12 of that match's 21 identities ended up containing
    members further apart than the merge bar itself.

    Averaging instead asks "do these two look alike overall", which is the
    actual question, and recovers AUC 0.940 on realistic re-linking pairs.
    Distances on this scale are much tighter than the old minimum produced,
    which is why APPEARANCE_MAX_DISTANCE and the other appearance bars below
    are calibrated to it directly rather than carried over.
    """
    # One matrix product per bucket pair rather than a Python loop over every
    # embedding pair. This is the pipeline's hottest arithmetic by a wide
    # margin - consolidation calls it for every candidate pair of clusters,
    # and each call compares up to a 10x10 gallery - so the per-element
    # interpreter overhead was most of the cost of consolidating a long
    # video. The arithmetic is cosine_distance's, unchanged: dot product of
    # two already-L2-normalised vectors, clamped, mapped to [0, 1].
    total = 0.0
    count = 0

    for bucket_a, embeddings_a in gallery_a.items():
        if not embeddings_a:
            continue

        left = np.asarray([embedding for embedding, _isolation in embeddings_a], dtype=np.float32)

        for bucket_b, embeddings_b in gallery_b.items():
            if not embeddings_b:
                continue

            right = np.asarray([embedding for embedding, _isolation in embeddings_b], dtype=np.float32)
            penalty = bin_penalty(bucket_a, bucket_b)

            similarity = np.clip(left @ right.T, -1.0, 1.0)
            distance = np.clip((1.0 - similarity) / 2.0, 0.0, 1.0)

            total += float(np.minimum(1.0, distance + penalty).sum())
            count += distance.size

    return total / count if count else None


def _appearance_breaks_continuity(tracklet, embedding, box):
    """True if `embedding` (this frame's candidate detection, from `box`)
    looks like a clearly different person from `tracklet`'s own gallery so
    far - see APPEARANCE_CONTINUITY_MAX_DISTANCE. False (never blocks
    continuation) until the tracklet has gathered enough of a gallery to
    judge by, or if no embedding could be extracted this frame."""
    if embedding is None:
        return False

    gallery_size = sum(len(embeddings) for embeddings in tracklet["embeddings"].values())
    if gallery_size < MIN_GALLERY_FOR_CONTINUITY_CHECK:
        return False

    bucket = shape_bin(aspect_ratio(box))
    distance = _gallery_distance(
        tracklet["embeddings"], {bucket: [(embedding, GALLERY_ISOLATION_WHEN_ALONE)]}
    )

    return distance is not None and distance > APPEARANCE_CONTINUITY_MAX_DISTANCE


def _tracklet_to_jsonable(tracklet):
    """track_and_chain's in-memory tracklet (numpy boxes) -> a plain-JSON
    dict for TRACKLETS_RAW_NAME. See _tracklet_from_jsonable for the
    inverse.

    The embeddings are NOT included: _save_tracklets stacks every tracklet's
    into one float32 array of its own and replaces this key with the row
    range that belongs to each (see TRACKLET_EMBEDDINGS_NAME). Building them
    here as decimal text only for the caller to discard was itself
    expensive at the file sizes that motivated the split."""
    return {
        "track_id": int(tracklet["track_id"]),
        "entries": [[int(frame_idx), [float(v) for v in box]] for frame_idx, box in tracklet["entries"]],
    }


def _gallery_entry_from_jsonable(entry):
    """One saved gallery snapshot -> (embedding, isolation).

    Tolerates the older format, where a snapshot was a bare list of floats
    with no isolation recorded (see GALLERY_ISOLATION_WHEN_ALONE), so a
    TRACKLETS_RAW_NAME written before that existed can still be re-
    consolidated rather than having to repeat the expensive detect+track+
    embed pass. Such a snapshot is treated as maximally crowded: nothing is
    known about it, so it should be the first to lose its gallery slot to a
    snapshot that has a real measurement behind it."""
    if entry and isinstance(entry[0], (list, tuple)):
        embedding, isolation = entry
        return np.array(embedding, dtype=np.float32), float(isolation)

    return np.array(entry, dtype=np.float32), 0.0


def _tracklet_from_jsonable(data):
    return {
        "track_id": data["track_id"],
        "entries": [(frame_idx, np.array(box, dtype=float)) for frame_idx, box in data["entries"]],
        "embeddings": {
            int(bucket): [_gallery_entry_from_jsonable(entry) for entry in embeddings]
            for bucket, embeddings in data["embeddings"].items()
        },
    }


def _usable_tracklet(tracklet):
    """Whether a tracklet could ever become an identity - see
    _consolidate, which drops anything shorter than MIN_TRACKLET_FRAMES
    before it builds a single cluster. Checked at save time too so the raw
    file does not carry tracklets that every reader is going to throw away:
    on an uncalibrated 4K job the single-frame flicker of a whole crowd was
    the bulk of an 80,000-tracklet file."""
    return len(tracklet["entries"]) >= MIN_TRACKLET_FRAMES


def _save_tracklets(tracklets, meta, output_path):
    """Persists track_and_chain's raw output (every tracklet, plus the frame
    size/fps consolidate_tracklets needs) so consolidation can be re-run
    later without repeating the expensive detect+track+embed pass - same
    idea as BallDetection.ballDetection's RAW_CANDIDATES_LOG_NAME /
    reselect_ball. Not indented (unlike this module's other JSON logs): a
    real video's tracklets carry thousands of appearance embeddings between
    them, and pretty-printing that would meaningfully bloat the file for a
    human who was never going to read raw embedding floats anyway."""
    path = Path(output_path) / TRACKLETS_RAW_NAME
    embeddings_path = Path(output_path) / TRACKLET_EMBEDDINGS_NAME

    keep = [t for t in tracklets if _usable_tracklet(t)]
    dropped = len(tracklets) - len(keep)

    # Every gallery snapshot in the job, stacked into one array; each
    # tracklet's JSON entry just records which rows are its own.
    vectors, isolations = [], []
    jsonable = []
    for tracklet in keep:
        record = _tracklet_to_jsonable(tracklet)
        buckets = {}

        for bucket, entries in tracklet["embeddings"].items():
            start = len(vectors)
            for embedding, isolation in entries:
                vectors.append(embedding)
                isolations.append(isolation)
            buckets[str(bucket)] = [start, len(vectors)]
        record["embeddings"] = buckets
        jsonable.append(record)

    with open(path, "w") as f:
        json.dump({"meta": meta, "tracklets": jsonable, "embeddings_file": TRACKLET_EMBEDDINGS_NAME}, f)

    np.savez(
        embeddings_path,
        vectors=np.asarray(vectors, dtype=np.float32).reshape(len(vectors), -1),
        isolations=np.asarray(isolations, dtype=np.float32),
    )

    if dropped:
        print(f"Raw tracklets: dropped {dropped} under {MIN_TRACKLET_FRAMES} frames "
              f"(consolidation discards them anyway)")
    print(f"Raw tracklets saved: {path} ({path.stat().st_size / 1e6:.0f} MB) "
          f"+ {embeddings_path.name} ({embeddings_path.stat().st_size / 1e6:.0f} MB)")


def _load_tracklets(output_path):
    path = Path(output_path) / TRACKLETS_RAW_NAME

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run track_and_chain(video_path, output_path) first "
            f"(or call trackplayers_offline() to run both stages together)."
        )

    with open(path) as f:
        data = json.load(f)

    embeddings_file = data.get("embeddings_file")
    if embeddings_file is None:  # noqa: SIM108 - see the comment below
        # Older file with the embeddings inline as JSON numbers - still
        # readable, so a job whose expensive detect pass already ran does
        # not have to repeat it just because the format moved on.
        tracklets = [_tracklet_from_jsonable(t) for t in data["tracklets"]]
        return tracklets, data["meta"]

    stored = np.load(Path(output_path) / embeddings_file)
    vectors, isolations = stored["vectors"], stored["isolations"]

    tracklets = []
    for record in data["tracklets"]:
        tracklet = _tracklet_from_jsonable({**record, "embeddings": {}})
        tracklet["embeddings"] = {
            int(bucket): [(vectors[i], float(isolations[i])) for i in range(start, end)]
            for bucket, (start, end) in record["embeddings"].items()
        }
        tracklets.append(tracklet)

    return tracklets, data["meta"]


def _bridge(cluster_a, cluster_b):
    """
    None if the two clusters' time spans overlap anywhere - they were on
    screen simultaneously as distinct detections, so they can never be the
    same person. Otherwise (gap_frames, box_a, box_b) for the closest pair of
    fragment boundaries across the two clusters, since a cluster that has
    already absorbed earlier merges may itself span several disjoint spans.

    The two boundary BOXES are returned rather than their centres because
    _pair_cost needs both derivations: the centre for its pixel-space
    spatial term, and the foot point for the court-metre physical-continuity
    check (see foot_point on why a ground-plane homography needs the foot,
    not the centre).

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
        return gap, cluster_a["end_pos"], cluster_b["start_pos"]

    if cluster_b["end_frame"] < cluster_a["start_frame"]:
        gap = cluster_a["start_frame"] - cluster_b["end_frame"]
        return gap, cluster_b["end_pos"], cluster_a["start_pos"]

    # Before treating any overlap as proof of two different people, check
    # whether the frames they actually share are one body detected twice.
    # A real tracker does this: a player mid-jump, or half-occluded at the
    # net, briefly yields two boxes on the same person, and each box feeds a
    # different tracklet. Those few frames used to veto the merge outright -
    # on a real match that split one player (1514 frames) from the rest of
    # himself (2561 frames) over a 5-frame overlap, one frame of which had
    # the two boxes at IoU 0.85 with their centres 9px apart. The overlap
    # has to be both negligible and box-coincident to qualify, so two
    # genuinely co-present players (who share many frames, far apart) are
    # still disqualified exactly as before.
    shared = cluster_a["by_frame"].keys() & cluster_b["by_frame"].keys()
    if shared:
        if not _is_duplicate_detection(cluster_a, cluster_b, shared):
            return None

        frame = next(iter(shared))
        return 0, cluster_a["by_frame"][frame], cluster_b["by_frame"][frame]

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

    return best


def _is_duplicate_detection(cluster_a, cluster_b, shared):
    """Whether the frames two clusters share look like one person detected
    twice rather than two people genuinely on screen together - see the
    caller in _bridge. Requires the shared frames to be a negligible slice
    of both clusters AND the two boxes to sit essentially on top of each
    other throughout them; two different players who happen to stand close
    score far below the IoU bar (measured at 0.06 and 0.00 on the real
    match's other small-overlap pairs, against 0.48 for the genuine
    double-detection)."""
    if len(shared) > DUPLICATE_OVERLAP_MAX_FRAMES:
        return False

    smaller = min(len(cluster_a["by_frame"]), len(cluster_b["by_frame"]))
    if not smaller or len(shared) / smaller > DUPLICATE_OVERLAP_MAX_FRACTION:
        return False

    total = 0.0
    for frame in shared:
        total += _box_overlap(cluster_a["by_frame"][frame], cluster_b["by_frame"][frame])

    return total / len(shared) >= DUPLICATE_OVERLAP_MIN_IOU


def _pair_cost(cluster_a, cluster_b, diagonal, homography, max_cost=MERGE_MAX_COST):
    """Merge cost for two clusters, or None if they can't/shouldn't merge -
    i.e. the weight of the graph edge between them (see _consolidate), with
    None meaning "no edge". Lower is a stronger match.

    max_cost defaults to MERGE_MAX_COST for the main consolidation pass, but
    _enforce_roster_cap's relaxed re-merge attempt passes a looser bar
    (RELAXED_MERGE_MAX_COST) to recover a real player who was over-fragmented
    past the strict threshold, without loosening first-pass matching itself.

    Two routes to a merge, in order of how strong the evidence is:

    1. Physical continuity - a short enough gap and a small enough move
       across the court that no second person could be involved. This skips
       the appearance gate entirely (see PHYSICAL_CONTINUITY_MAX_COURT_M for
       why, and for the measurements) and needs a homography, so it is only
       available once the court is calibrated.
    2. Appearance - the original route, an absolute gate on how alike the
       two galleries look, then a blended appearance/spatial cost to order
       the pairs that passed it.
    """

    # Real players physically cannot cross the net to the other team's side -
    # this is a rule of the game, not a soft prior, so two clusters
    # confidently on opposite sides are disqualified outright, the same way
    # _bridge disqualifies a genuine time overlap. A cost-based penalty was
    # tried first and rejected: two tracklets in identical-looking kits (the
    # exact ambiguity this exists to resolve) can have a near-zero
    # appearance+spatial cost on their own, so no penalty short of
    # effectively re-implementing a hard block reliably stopped that pair
    # from merging for the wrong reason. A genuine side switch (teams swap
    # ends between sets) still gets a fragmented identity here, but that's
    # a fail-safe outcome a human fixes in seconds via the existing
    # candidate-match/rename tooling (_find_candidate_matches,
    # API/services/players.build_canonical_mapping) - a false merge would
    # instead have silently corrupted stats with no such recovery path.
    side_a, confidence_a = _cluster_side(cluster_a)
    side_b, confidence_b = _cluster_side(cluster_b)
    side_mismatch = side_a is not None and side_b is not None and side_a != side_b

    if side_mismatch and confidence_a >= SIDE_MIN_CONFIDENCE and confidence_b >= SIDE_MIN_CONFIDENCE:
        return None

    # Below that confidence the sides still disagree, which is still
    # evidence - just not proof. See SIDE_MISMATCH_PENALTY.
    side_penalty = (SIDE_MISMATCH_PENALTY * min(confidence_a, confidence_b)) if side_mismatch else 0.0

    bridge = _bridge(cluster_a, cluster_b)
    if bridge is None or bridge[0] > MAX_MERGE_GAP_FRAMES:
        return None

    gap, box_a, box_b = bridge

    # Route 1: physical certainty. Deliberately checked before the
    # appearance gate rather than as a discount afterwards - the whole point
    # is that a pair this physically constrained does not need the crops to
    # agree, and gating it on appearance first is exactly what was splitting
    # single players into a fragment every few seconds.
    if homography is not None and gap <= PHYSICAL_CONTINUITY_MAX_GAP_FRAMES:
        court_a = np.array(pixel_to_court(*foot_point(box_a), homography))
        court_b = np.array(pixel_to_court(*foot_point(box_b), homography))
        moved_m = float(np.linalg.norm(court_a - court_b))

        if moved_m <= PHYSICAL_CONTINUITY_MAX_COURT_M:
            # Deliberately not carrying side_penalty or the zone gate below.
            # Two fragments a metre apart half a second apart are the same
            # person; if their dominant sides disagree at all it is because
            # the player is standing at the net, where the side split is a
            # coin toss, not because they crossed it.
            return PHYSICAL_CONTINUITY_MAX_COST * (moved_m / PHYSICAL_CONTINUITY_MAX_COURT_M)

    # Route 2: appearance, now led by where on the court the two clusters
    # actually were.
    #
    # The zone gate is absolute, like the appearance one below it: two
    # people who never occupied any of the same court are not the same
    # player, however alike two crops of them happen to look. This is what
    # stops a left-back fragment being offered as the same person as a
    # right-front one - measured on this pipeline's own candidate pairs,
    # 26% of them shared no court zone at all before this existed.
    zone = _zone_distance(cluster_a, cluster_b)
    if zone is not None and zone > ZONE_MAX_DISTANCE:
        return None

    pos_a, pos_b = centre_of(box_a), centre_of(box_b)

    appearance = _gallery_distance(cluster_a["embeddings"], cluster_b["embeddings"])
    if appearance is None:
        return None

    # The appearance gate is absolute - the blended cost below only orders
    # the pairs that already passed it. Folding appearance into a weighted
    # sum instead (as the only check) let a good spatial term buy a merge
    # between two people who plainly don't look alike, which is precisely
    # how unrelated players ended up chained into one identity.
    if appearance > APPEARANCE_MAX_DISTANCE:
        return None

    raw_distance = np.linalg.norm(pos_a - pos_b) / diagonal
    spatial = min(1.0, raw_distance / (1.0 + gap / POSITION_RELAXATION_FRAMES))

    # With no calibration there is no zone opinion to weigh, so appearance
    # and the pixel-space spatial term split the whole cost between them
    # exactly as they did before zones existed.
    if zone is None:
        cost = (APPEARANCE_WEIGHT_WITHOUT_ZONES * appearance
                + (1.0 - APPEARANCE_WEIGHT_WITHOUT_ZONES) * spatial)
    else:
        cost = (APPEARANCE_WEIGHT * appearance
                + ZONE_WEIGHT * zone
                + max(0.0, 1.0 - APPEARANCE_WEIGHT - ZONE_WEIGHT) * spatial)

    cost += side_penalty

    return cost if cost <= max_cost else None


def _merge_into(cluster_a, cluster_b):
    # One identity gets at most one box per frame. The two clusters can
    # genuinely share a frame or two - that is the duplicate-detection case
    # _bridge deliberately lets through (see _is_duplicate_detection), one
    # body the detector boxed twice - and keeping both copies would put the
    # same person on screen twice at once in player_positions.json, which
    # everything downstream (the rendered overlay, thumbnail conflict
    # detection, stats) reads as two people. The larger box wins those
    # frames, on the same reasoning as players._ranked_candidates_per_
    # player: the tighter of two boxes on one person is usually the one
    # clipped by the occlusion that caused the double detection.
    merged = dict(cluster_a["by_frame"])
    for frame_idx, box in cluster_b["by_frame"].items():
        current = merged.get(frame_idx)
        if current is None or _box_area(box) > _box_area(current):
            merged[frame_idx] = box

    cluster_a["by_frame"] = merged
    cluster_a["entries"] = sorted(merged.items())
    # _bridge's sweep over spans depends on both lists being sorted by start
    # frame - each side is already sorted coming in, so this is a cheap
    # one-time sort, not the repeated O(n*m) scan that used to dominate.
    cluster_a["spans"] = sorted(cluster_a["spans"] + cluster_b["spans"])
    cluster_a["track_ids"] |= cluster_b["track_ids"]
    cluster_a["side_counts"]["A"] += cluster_b["side_counts"]["A"]
    cluster_a["side_counts"]["B"] += cluster_b["side_counts"]["B"]
    cluster_a["zone_counts"] = [a + b for a, b in
                                zip(cluster_a["zone_counts"], cluster_b["zone_counts"])]

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
            # Keep the least crowded snapshots out of both fragments, the
            # same rule _store_embedding applies within one. This replaced an
            # even subsample over the merged span, which was measured as the
            # WORST of the selection strategies tried (AUC 0.777 against
            # 0.841 for least-crowded) - spreading the samples out evenly
            # says nothing about whether any of them are usable.
            #
            # Unless there is nothing to choose between them, which is the
            # case for a TRACKLETS_RAW_NAME written before isolation was
            # recorded (see _gallery_entry_from_jsonable - every snapshot in
            # such a file reads back as equally crowded). Truncating a tied
            # list would silently become "keep whichever happened to be
            # first", an arbitrary rule nothing measured, so those files keep
            # the even subsample they were always consolidated with until
            # they are re-tracked.
            isolations = {isolation for _embedding, isolation in gallery}
            if len(isolations) > 1:
                gallery.sort(key=lambda entry: -entry[1])
                del gallery[APPEARANCE_GALLERY_SIZE:]
            else:
                keep = sorted(set(np.linspace(0, len(gallery) - 1, APPEARANCE_GALLERY_SIZE).round().astype(int)))
                gallery[:] = [gallery[i] for i in keep]


def _concurrently_excess(clusters, side_indices, max_per_side):
    """
    Sweep-line over every span of every cluster on one side, returning
    {cluster index: how many frames it spent as the "extra" identity} for
    every cluster that was ever part of a moment where more than
    max_per_side of them were simultaneously active. A side simply having
    more than max_per_side distinct identities across the WHOLE video is not
    by itself a violation - heavy fragmentation (many short pieces of the
    same few real players) or ordinary substitutions (a sub's tracklet only
    starts once the player they replaced is off court) can easily produce
    that without anyone ever sharing the screen with 6 others at once. Only
    an identity simultaneously on screen alongside max_per_side
    stronger (longer-tracked) ones is excess, and only for as long as that
    actually holds.

    Counting frames rather than just returning a set matters: on real
    footage the over-capacity moments are mostly brief detection artefacts
    (one player momentarily detected as two overlapping boxes), so a plain
    "was it ever excess" test condemns an identity with 1400 good frames
    over a 5-frame glitch. _enforce_roster_cap uses the ratio of these
    frames to the cluster's own length to tell a real interloper (extra for
    most of its life) from a real player caught in a blip.

    Spans are treated as closed intervals [start, end] (matching _bridge's
    own inclusive overlap test), modelled here as half-open [start, end+1)
    so a sweep can find real simultaneity without a naive O(n^2) pairwise
    scan: two adjacent, non-overlapping spans (one ending at frame 100, the
    next starting at 101) must never register as overlapping, which is why
    each span's "leaves" event is processed before any "enters" event at the
    same coordinate.
    """
    events = []
    for i in side_indices:
        for start, end in clusters[i]["spans"]:
            events.append((start, 1, i))
            events.append((end + 1, -1, i))
    events.sort(key=lambda e: (e[0], e[1]))  # leaves (-1) before enters (+1) at the same frame

    active = set()
    excess_frames = {}
    position = None

    for frame, delta, i in events:
        # Charge the interval just ended to whoever was extra throughout it,
        # before this event changes the active set.
        if position is not None and frame > position and len(active) > max_per_side:
            weakest = sorted(active, key=lambda k: _cluster_duration(clusters[k]))
            for k in weakest[:len(active) - max_per_side]:
                excess_frames[k] = excess_frames.get(k, 0) + (frame - position)

        position = frame

        if delta == 1:
            active.add(i)
        else:
            active.discard(i)

    return excess_frames


def _enforce_roster_cap(clusters, diagonal, homography, max_per_side=MAX_PLAYERS_PER_SIDE):
    """
    Post-consolidation pass: indoor volleyball fields at most max_per_side
    players simultaneously on one side of the net, so a side with more than
    that many identities genuinely on screen together at some point (see
    _concurrently_excess) is either over-fragmented (the strict
    MERGE_MAX_COST bar split one real player into extra pieces that
    happened to overlap a stronger fragment of the same person - a
    fragmentation artefact, not two real people) or contaminated by a
    non-player who wandered into the service-area buffer (coach, ref, ball
    kid). For each such excess identity, tries a relaxed re-merge into one
    of that moment's stronger (longer-tracked) identities first
    (RELAXED_MERGE_MAX_COST, still respecting _bridge's no-time-overlap
    rule). Whatever still doesn't merge is flagged likely_non_player - but
    only if it was over capacity for a real share of its own tracked life
    (NON_PLAYER_MIN_EXCESS_FRACTION), since most over-capacity moments on
    real footage are momentary detection artefacts rather than an actual
    extra person. Flagged identities are left in place rather than deleted,
    so a human can still review and override.

    Clusters with no confident side (see _cluster_side - no calibration, or
    too much time spent near the net to judge) are left out of this
    entirely: capping a side we're not confident about would risk wrongly
    flagging a real player.

    Returns (clusters with excess identities merged away, the set of
    surviving clusters' `id()`s that were flagged likely_non_player).
    """
    non_player_ids = set()

    for side in ("A", "B"):
        side_indices = []
        for i, cluster in enumerate(clusters):
            if cluster is None:
                continue
            cluster_side, confidence = _cluster_side(cluster)
            if cluster_side == side and confidence >= SIDE_MIN_CONFIDENCE:
                side_indices.append(i)

        if len(side_indices) <= max_per_side:
            continue

        excess_frames = _concurrently_excess(clusters, side_indices, max_per_side)
        if not excess_frames:
            continue

        primary = [i for i in side_indices if i not in excess_frames]

        for j in excess_frames:
            best_target, best_cost = None, RELAXED_MERGE_MAX_COST
            for i in primary:
                cost = _pair_cost(clusters[i], clusters[j], diagonal, homography,
                                  max_cost=RELAXED_MERGE_MAX_COST)
                if cost is not None and cost < best_cost:
                    best_target, best_cost = i, cost

            if best_target is not None:
                print(f"roster cap: re-merging excess side-{side} identity "
                      f"(frames {clusters[j]['spans']}) into primary (cost {best_cost:.3f})")
                _merge_into(clusters[best_target], clusters[j])
                clusters[j] = None
                continue

            # Nothing to merge into - decide whether this is a real
            # interloper or a real player caught in a brief detection
            # artefact, by how much of its own life it spent as the extra
            # one (see NON_PLAYER_MIN_EXCESS_FRACTION).
            duration = _cluster_duration(clusters[j])
            fraction = excess_frames[j] / duration if duration else 0.0

            if fraction >= NON_PLAYER_MIN_EXCESS_FRACTION:
                non_player_ids.add(id(clusters[j]))
                print(f"roster cap: side {side} identity over capacity for "
                      f"{excess_frames[j]}/{duration} frames ({fraction:.0%} of its life) "
                      f"- flagging likely_non_player")
            else:
                print(f"roster cap: side {side} identity over capacity for only "
                      f"{excess_frames[j]}/{duration} frames ({fraction:.0%} of its life) "
                      f"- treating as a real player caught in a brief overlap")

    return [c for c in clusters if c is not None], non_player_ids


def _consolidate(tracklets, diagonal, homography):
    """
    The graph-matching core of consolidate_tracklets. Builds the tracklet-
    similarity graph - one node per raw tracklet (that survives
    MIN_TRACKLET_FRAMES and MIN_IN_PLAY_FRACTION, i.e. is long enough to be
    a real person and spent its time on the court rather than in the crowd),
    one edge per pair judged compatible by _pair_cost (no time overlap,
    appearance/spatial/side cost under MERGE_MAX_COST) - then greedily
    contracts the globally cheapest edge first, repeating until no
    compatible pair is left. Finally runs _enforce_roster_cap so no side
    ends up with more simultaneously-active identities than volleyball
    actually allows.

    This is deliberately NOT "for each tracklet, pick its single best match
    independently": a track only gets matched by first looking at every
    compatible edge in the WHOLE graph and taking the strongest one anywhere
    (ties elsewhere never get the chance to steal a merge from a clearly
    better match), and merging updates that cluster's gallery/spans before
    any further edges from it are considered - so a fragment can bridge a
    gap via a third tracklet's evidence, not just a direct pairwise
    comparison. Global-cheapest-first (rather than a single left-to-right
    pass) is what lets a fragment pick its best match anywhere in the video,
    not just the nearest one in time.

    Returns (final clusters, the set of surviving clusters' `id()`s flagged
    likely_non_player by _enforce_roster_cap, the graph's nodes, its initial
    edges, and the ordered list of merges actually performed) - the nodes/
    edges/merge-log exist purely for IDENTITY_GRAPH_NAME's inspection log
    (see consolidate_tracklets/_save_identity_graph) and don't feed back into
    the matching itself.
    """
    clusters = []
    off_court = 0

    for t in tracklets:
        if len(t["entries"]) < MIN_TRACKLET_FRAMES:
            continue

        # Whoever this is, they spent their time off the court - the crowd,
        # the bench, someone walking behind the baseline - so they never
        # become an identity to match, name or review in the first place.
        if in_play_fraction(t["entries"], homography) < MIN_IN_PLAY_FRACTION:
            off_court += 1
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
            "track_ids": {t["track_id"]},
            "side_counts": _count_sides(t["entries"], homography),
            "zone_counts": _count_zones(t["entries"], homography),
        })

    if off_court:
        print(f"consolidate: dropped {off_court} off-court tracklet(s) - under "
              f"{MIN_IN_PLAY_FRACTION:.0%} of their frames were inside the court and its "
              f"service area, so they are not players")

    nodes = [
        {
            "index": index,
            "track_id": next(iter(cluster["track_ids"])),
            "start_frame": cluster["start_frame"],
            "end_frame": cluster["end_frame"],
            "frame_count": len(cluster["entries"]),
        }
        for index, cluster in enumerate(clusters)
    ]

    # Without a court there is nothing worth deciding here yet, so the
    # graph is not built at all - every surviving tracklet stays its own
    # provisional identity and the real matching waits for calibration.
    #
    # This is the "do everything now, cut it down afterwards" split. None of
    # the evidence that actually resolves identity survives an uncalibrated
    # pass: the court gate that separates players from the crowd, which side
    # of the net someone is on, their rotation zone, and physical continuity
    # in metres are all court-derived, leaving only an appearance model that
    # measurably cannot carry the decision on its own. So an uncalibrated
    # consolidation is not a worse answer, it is a throwaway one - and
    # stage_runner._recalibrate redoes it from these same saved tracklets
    # the moment a court exists, for free.
    #
    # Paying for it anyway is what made a 51-minute 4K job unusable: with
    # every detection in the building kept, its 80,000 tracklets put three
    # billion pairs through the graph, and it was still grinding through
    # them three hours after detection had finished.
    if homography is None:
        print(f"consolidate: no court calibration yet - keeping {len(clusters)} tracklet(s) as "
              f"provisional identities and leaving the matching until one is saved")
        return clusters, set(), nodes, [], []

    alive = list(range(len(clusters)))
    cost_cache = {}

    # Only pairs close enough in time to be the same person are costed at
    # all. _pair_cost already refuses anything further apart than
    # MAX_MERGE_GAP_FRAMES, so skipping those here changes nothing about the
    # result - it just stops computing an answer that is always None.
    #
    # What that is worth grows with the video. Costing every pair is
    # quadratic in tracklets AND the expensive part of each one is
    # _gallery_distance, so a long recording spent hours comparing the
    # appearance of people who were on screen forty minutes apart: a
    # 51-minute 4K job sat in this loop for close to three hours after
    # detection had already finished. Clusters are ordered by start frame
    # here, so once one starts later than the current one ends plus the max
    # gap, so does every cluster after it, and the scan can stop.
    order = sorted(alive, key=lambda i: clusters[i]["start_frame"])

    for position, a in enumerate(order):
        reach = clusters[a]["end_frame"] + MAX_MERGE_GAP_FRAMES
        for b in order[position + 1:]:
            if clusters[b]["start_frame"] > reach:
                break
            cost = _pair_cost(clusters[a], clusters[b], diagonal, homography)
            if cost is not None:
                cost_cache[(min(a, b), max(a, b))] = cost

    # A snapshot of the full graph as first built - every pair judged
    # compatible at all, not just the ones that end up merged. The merge
    # loop below is free to mutate cost_cache itself from here on.
    graph_edges = [{"a": i, "b": j, "cost": round(float(cost), 4)} for (i, j), cost in cost_cache.items()]
    merge_log = []

    while cost_cache:
        (i, j), cost = min(cost_cache.items(), key=lambda kv: kv[1])

        # Printed so a merge can be checked against the annotated video by
        # frame number - a cost near MERGE_MAX_COST was accepted but wasn't a
        # confident match, and is the first place to look if two different
        # players ever end up sharing one id.
        borderline = bool(cost > MERGE_MAX_COST * 0.7)
        flag = " <- borderline, worth checking" if borderline else ""
        print(f"consolidate: merging frames {clusters[j]['spans']} into "
              f"{clusters[i]['spans']} (cost {cost:.3f}/{MERGE_MAX_COST}){flag}")

        merge_log.append({
            "order": len(merge_log) + 1,
            "into_track_ids": sorted(clusters[i]["track_ids"]),
            "merged_track_ids": sorted(clusters[j]["track_ids"]),
            "cost": round(float(cost), 4),
            "borderline": borderline,
        })

        _merge_into(clusters[i], clusters[j])
        clusters[j] = None
        alive.remove(j)

        for key in [k for k in cost_cache if i in k or j in k]:
            del cost_cache[key]

        for k in alive:
            if k == i:
                continue
            lo, hi = (i, k) if i < k else (k, i)
            cost = _pair_cost(clusters[lo], clusters[hi], diagonal, homography)
            if cost is not None:
                cost_cache[(lo, hi)] = cost

    survivors = [clusters[k] for k in alive]
    survivors, non_player_ids = _enforce_roster_cap(survivors, diagonal, homography)

    return survivors, non_player_ids, nodes, graph_edges, merge_log


def _save_identity_graph(nodes, graph_edges, merge_log, clusters, non_player_ids, output_path, provisional):
    """Saves the tracklet-similarity graph _consolidate matched identities
    from - see IDENTITY_GRAPH_NAME."""
    stable_ids = [
        {
            "stable_id": stable_id,
            "source_track_ids": sorted(cluster["track_ids"]),
            "start_frame": cluster["start_frame"],
            "end_frame": cluster["end_frame"],
            "team": _cluster_side(cluster)[0],
            "likely_non_player": id(cluster) in non_player_ids,
        }
        for stable_id, cluster in enumerate(clusters, start=1)
    ]

    graph_file = Path(output_path) / IDENTITY_GRAPH_NAME
    with open(graph_file, "w") as f:
        json.dump(
            {
                # True when these identities are one-per-tracklet placeholders
                # from an uncalibrated pass rather than a real matching - see
                # _consolidate. stage_runner._recalibrate reads this to know
                # the job still owes itself a real consolidation.
                "provisional": provisional,
                "nodes": nodes,
                "edges": graph_edges,
                "merges": merge_log,
                "stable_ids": stable_ids,
            },
            f, indent=2,
        )

    print(f"Player identity graph saved: {graph_file}")
    for entry in stable_ids:
        if len(entry["source_track_ids"]) > 1:
            print(f"  P{entry['stable_id']}: merged from raw tracks {entry['source_track_ids']}")


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

            # The same court-position evidence _pair_cost weighs, applied
            # here too. A hint is cheap to be wrong about, but a hint
            # between two fragments that were never on the same part of the
            # court is not a weak hint, it is a wrong one - and the review
            # UI groups these hints transitively (see API/services/players.
            # build_candidate_groups), so one bad pair pulls a whole group
            # off. Measured before this existed: 26% of the pairs offered
            # shared no court zone at all.
            zone = _zone_distance(clusters[i], clusters[j])
            if zone is not None and zone > CANDIDATE_MAX_ZONE_DISTANCE:
                continue

            side_i, confidence_i = _cluster_side(clusters[i])
            side_j, confidence_j = _cluster_side(clusters[j])
            if (
                side_i is not None and side_j is not None and side_i != side_j
                and confidence_i >= SIDE_MIN_CONFIDENCE and confidence_j >= SIDE_MIN_CONFIDENCE
            ):
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


def _run_pipeline(cap, detect_fn, windows=None):
    """
    Drives track_and_chain's 3-stage read -> detect -> track+embed+chain
    pipeline: a reader thread decodes frames, a detector thread runs
    detect_fn (YOLO) on them, and this generator yields
    (frame_idx, frame, det) tuples for the caller to run BoT-SORT's
    tracker.update() and everything after it on - single-threaded and in
    strict frame order, since BoT-SORT's Kalman state depends on it.

    The actual speedup: while the caller is busy tracking/cropping/embedding
    frame N, the reader and detector threads are already decoding and
    detecting frame N+1, instead of that work waiting for frame N's tracking
    step to finish first. cv2's frame decode and the detectors' CUDA calls
    both release the GIL for the bulk of their work, so this overlap is real
    wall-clock time saved, not just concurrency on paper.

    windows, if given, is GameStatusDetection.rallyWindows.compute_track_
    windows's own [start_frame, end_frame] ranges (inclusive, sorted,
    non-overlapping) - everywhere outside them is skipped entirely rather
    than decoded and discarded, which is where the actual time is saved
    (decode is real cost too, not just the detector call - see
    MAX_INFERENCE_LONG_SIDE's own profiling notes on how much of this
    stage's time is per-frame work). None (the default) reads every frame,
    unchanged from before this existed - track_and_chain is the only caller
    that ever passes windows.

    Frame indices emitted after a skip are NOT contiguous with the ones
    before it - the caller (track_and_chain) is what notices that jump and
    resets BoT-SORT's own tracker state at exactly that point, since a
    generator has no way to signal "and also, forget everything you knew"
    to code several stack frames up other than the data it yields.

    Bounded queues (FRAME_QUEUE_SIZE/DETECTION_QUEUE_SIZE) give backpressure
    so a slow downstream stage can't let an upstream one race arbitrarily far
    ahead and pile up whole BGR frames in memory. A sentinel object (not
    None - a real detection result could plausibly look falsy) marks
    end-of-stream on both queues; an exception in either thread is captured
    and re-raised here once the pipeline has fully drained, rather than left
    to hang the other thread waiting on a producer that already died.
    """
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    detection_queue = queue.Queue(maxsize=DETECTION_QUEUE_SIZE)
    stop = object()
    errors = []

    def read_frames():
        try:
            if windows is None:
                frame_idx = 0
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame_queue.put((frame_idx, frame))
                    frame_idx += 1
            else:
                for start, end in windows:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                    frame_idx = start
                    while frame_idx <= end:
                        success, frame = cap.read()
                        if not success:
                            # Ran out of real video mid-window (a window's
                            # own end estimate can run slightly past the
                            # true last frame) - nothing left for this or
                            # any later window either.
                            return
                        frame_queue.put((frame_idx, frame))
                        frame_idx += 1
        except Exception as exc:  # noqa: BLE001 - captured, re-raised on the calling thread below
            errors.append(exc)
        finally:
            frame_queue.put(stop)

    def detect_frames():
        errored = False
        try:
            while True:
                item = frame_queue.get()
                if item is stop:
                    break
                frame_idx, frame = item
                detection_queue.put((frame_idx, frame, detect_fn(frame)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            errored = True

        if errored:
            # read_frames only ever blocks trying to push into a full queue,
            # never waiting on a consumer - draining whatever it still
            # pushes guarantees it can always finish instead of blocking
            # forever on a stage that has already given up.
            while frame_queue.get() is not stop:
                pass

        detection_queue.put(stop)

    reader = threading.Thread(target=read_frames, daemon=True)
    detector = threading.Thread(target=detect_frames, daemon=True)
    reader.start()
    detector.start()

    while True:
        item = detection_queue.get()
        if item is stop:
            break
        yield item

    reader.join()
    detector.join()

    if errors:
        raise errors[0]


def track_and_chain(video_path, output_path):
    """
    Stage 1: detect (YOLO26x) + track (BoT-SORT) every frame, chaining
    detections into local tracklets using the tracker's own frame-to-frame
    ids plus the appearance-continuity guard (_appearance_breaks_continuity).
    No identity decision is made here - a tracklet simply ends the instant
    BoT-SORT loses it or a clear appearance mismatch is caught; fragmentation
    is expected and is entirely consolidate_tracklets's job to resolve.

    Has nothing stable to preview - identity isn't decided until
    consolidate_tracklets has seen the whole video - so unlike that stage,
    this one has no show_preview/save_video of its own. Saves the raw
    tracklets to TRACKLETS_RAW_NAME and also returns them (with the frame
    metadata consolidate_tracklets needs) so trackplayers_offline can hand
    them straight to consolidate_tracklets in the same process without a
    redundant round-trip through disk.

    Runs as a 3-stage pipeline (read -> detect -> track+embed+chain)
    connected by bounded queues, each its own thread, rather than one
    sequential per-frame loop - see _run_pipeline for why this is safe and
    what it actually buys.
    """
    detector = YOLO(MODEL_PATH)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cuda_available = torch.cuda.is_available()
    track_device = 0 if cuda_available else "cpu"
    torch_device = torch.device("cuda:0" if cuda_available else "cpu")
    encoder = AppearanceEncoder(torch_device)

    # Built once (matching persist=True's old effect: state carries across
    # frames) using the exact same config resolution model.track() uses
    # internally, so behaviour matches what TRACKER_CONFIG's own yaml
    # specifies.
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKER_CONFIG)))
    cfg.device = torch_device
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)

    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", cuda_available)

    # Loaded up front so detections outside the court and its service area
    # never even become a tracklet - a bench player or spectator that's
    # never tracked can't later cost a consolidation match or show up in the
    # review UI.
    homography = load_homography(output_path)

    # The frames actually worth running detection over - see
    # rallyWindows.compute_track_windows's own docstring. This is what
    # makes game_status running FIRST in the pipeline (see
    # API/jobs.PHASE_ONE_STAGES) pay off: without it, this was the single
    # most expensive stage in the whole pipeline (697s on a real 5-minute
    # match, measured), most of it spent tracking dead time between
    # rallies nobody ever asked about.
    windows = compute_track_windows(output_path, fps)
    if windows is None:
        print("track_and_chain: no game_status.json yet - tracking the whole video")
    else:
        kept_frames = sum(end - start + 1 for start, end in windows)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"track_and_chain: tracking {len(windows)} rally window(s), "
              f"{kept_frames} of {total_frames} frame(s) ({kept_frames / max(1, total_frames):.0%}) - "
              f"see rallyWindows.PRE_ROLL_BUFFER_S for the pre-roll each one carries")

    started = time.monotonic()
    open_tracklets = []
    finished_tracklets = []
    frame_idx = -1
    previous_frame_idx = None

    # The frame's own size, but never above MAX_INFERENCE_LONG_SIDE - well
    # above Ultralytics' default 640 downscale, so distant players still
    # survive, without paying 4K's penalty for nothing. Aspect ratio is
    # preserved; boxes come back in the source frame's coordinates either
    # way, so nothing downstream needs to know this happened.
    inference_scale = min(1.0, MAX_INFERENCE_LONG_SIDE / max(frame_width, frame_height))
    inference_size = (int(frame_height * inference_scale), int(frame_width * inference_scale))
    if inference_scale < 1.0:
        print(f"detector running at {inference_size[1]}x{inference_size[0]} "
              f"(source is {frame_width}x{frame_height}) - see MAX_INFERENCE_LONG_SIDE")

    def detect(frame):
        # Returns the same Boxes object BoT-SORT has always been handed; the
        # fusion layer that briefly sat here is gone (see MODEL_PATH).
        return detector.predict(
            source=frame, classes=[0], conf=0.25, iou=0.45,
            imgsz=inference_size, device=track_device, verbose=False,
        )[0].boxes.cpu().numpy()

    for frame_idx, frame, det in _run_pipeline(cap, detect, windows=windows):
        # A non-contiguous frame_idx means the reader just jumped across a
        # skipped dead-time gap (see _run_pipeline's own doc comment).
        # BoT-SORT's tracker object has no idea real video time passed at
        # all in that case - update() is simply never called for the
        # skipped frames, so its own Kalman motion prediction and
        # track-expiry timers would otherwise carry on as if zero time had
        # elapsed, and could wrongly keep a pre-gap track "open" long enough
        # to snap onto whichever detection happens to appear right after
        # the jump. A fresh tracker instance has no tracks to make that
        # mistake with, and forcing every still-open tracklet closed first
        # stops a coincidentally-reused small track_id from silently
        # extending a tracklet that really ended before the gap - the same
        # thing that already happens at true end-of-video below, just
        # triggered mid-stream instead of once at the end.
        if previous_frame_idx is not None and frame_idx != previous_frame_idx + 1:
            finished_tracklets.extend(open_tracklets)
            open_tracklets = []
            tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
        previous_frame_idx = frame_idx

        tracks = tracker.update(det, frame, feats=None)
        boxes, ids = _unpack_tracks(tracks)

        detections = [
            (box, int(track_id)) for box, track_id in zip(boxes, ids) if is_in_play_area(box, homography)
        ]

        boxes_kept = [box for box, _ in detections]
        crops = [
            extract_crop(frame, box, exclude_boxes=[boxes_kept[j] for j in range(len(boxes_kept)) if j != i])
            for i, box in enumerate(boxes_kept)
        ]
        embeddings = encoder.encode(crops)

        open_tracklets, closed = _advance_tracklets(open_tracklets, detections, embeddings, frame_idx)
        finished_tracklets.extend(closed)

        if frame_idx % 200 == 0:
            print(f"track_and_chain - frame {frame_idx}: {len(detections)} detections, "
                  f"{len(open_tracklets)} open tracklets, {len(finished_tracklets)} finished")

    finished_tracklets.extend(open_tracklets)
    cap.release()

    elapsed = time.monotonic() - started
    print(f"track_and_chain done in {elapsed:.1f}s: {len(finished_tracklets)} raw tracklets over {frame_idx + 1} frames")

    meta = {"fps": fps, "frame_width": frame_width, "frame_height": frame_height}
    _save_tracklets(finished_tracklets, meta, output_path)

    return finished_tracklets, meta


def _replay_and_export(video_path, output_path, clusters, non_player_ids, fps, frame_width, frame_height,
                        show_preview, save_video):
    """
    Stage 2's replay: walks the source video once more, writing the
    annotated output video and player-position log with clusters' final
    stable ids, and - when show_preview is on - showing that same replay
    live (the "player identification preview": see consolidate_tracklets).
    """
    frame_lookup = {}
    team_by_stable_id = {}
    non_player_by_stable_id = {}
    for stable_id, cluster in enumerate(clusters, start=1):
        for f_idx, box in cluster["entries"]:
            frame_lookup.setdefault(f_idx, []).append((stable_id, box))
        team_by_stable_id[stable_id] = _cluster_side(cluster)[0]
        non_player_by_stable_id[stable_id] = id(cluster) in non_player_ids

    homography = load_homography(output_path)
    court_polygon = load_court_polygon(output_path)

    writer = None
    need_annotation = save_video or show_preview
    if save_video:
        writer = cv2.VideoWriter(
            str(Path(output_path) / OUTPUT_VIDEO_NAME),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height),
        )

    window_name = "Player Identification Preview"
    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def log_frame(frame_idx, entries):
        frame_players = []
        for stable_id, box in entries:
            cx, cy = centre_of(box)
            court = pixel_to_court(cx, cy, homography) if homography is not None else None
            # court_ground is where the player actually STANDS: the
            # homography only maps the ground plane, and `court` above
            # projects the box centre (roughly waist height), which the
            # transform therefore reads as a ground point further from the
            # camera than the player really is (see foot_point). On a
            # side-on camera that error runs across the court's width and
            # is mostly harmless, but on a camera looking down the court's
            # length it runs along the very axis that decides which half of
            # the net someone is on - measured on a real back-view match it
            # put players at 27.7m and -11.8m on an 18m court and flipped
            # the team of 15 of 41 identities. Anything comparing a player
            # against fixed court geometry (which side they're on, how far
            # from the net) wants this; `court` is kept as-is for
            # comparisons against the BALL, whose own court position is an
            # equally airborne projection (see
            # BallDetection.ballDetection.pixel_to_court) that the box
            # centre's similar elevation partly cancels against.
            ground = pixel_to_court(*foot_point(box), homography) if homography is not None else None
            frame_players.append({
                "stable_id": int(stable_id),
                "box": [float(v) for v in box],
                "pixel": [float(cx), float(cy)],
                "court": list(court) if court is not None else None,
                "court_ground": list(ground) if ground is not None else None,
                "team": team_by_stable_id.get(stable_id),
                "likely_non_player": non_player_by_stable_id.get(stable_id, False),
            })

        positions_log.append({
            "frame_idx": frame_idx,
            "timestamp_s": frame_idx / fps,
            "players": frame_players,
        })

    positions_log = []

    if not need_annotation:
        # Nothing is being drawn, so the source video's pixels are never
        # read - every box this logs already lives in the clusters. The
        # decode loop below exists only to drive the annotated output, and
        # walking a full match's frames just to count indices costs minutes
        # of pure video I/O for nothing. Skipping it is what makes
        # re-running consolidation after a calibration change cheap (see
        # API/services/stage_runner._recalibrate).
        for frame_idx in sorted(frame_lookup):
            log_frame(frame_idx, frame_lookup[frame_idx])

        positions_log_file = Path(output_path) / POSITIONS_LOG_NAME
        with open(positions_log_file, "w") as f:
            json.dump(positions_log, f, indent=2)

        print(f"Player positions saved: {positions_log_file}")
        return

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        entries = frame_lookup.get(frame_idx, [])

        if court_polygon is not None:
            cv2.polylines(frame, [court_polygon.astype(np.int32)],
                          isClosed=True, color=(255, 0, 255), thickness=2)

        _annotate(frame, entries)

        cv2.putText(frame, f"frame {frame_idx}   players {len(entries)}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if entries:
            log_frame(frame_idx, entries)

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

    print(f"Player positions saved: {positions_log_file}")


def consolidate_tracklets(video_path, output_path, tracklets=None, meta=None, show_preview=False, save_video=False):
    """
    Stage 2 ("post processing"): consolidate track_and_chain's raw tracklets
    into final player identities via the graph matching in _consolidate,
    save that graph for inspection (IDENTITY_GRAPH_NAME - see
    _save_identity_graph), then replay the source video writing the
    annotated output/position log with those final ids.

    show_preview shows that replay live - the "player identification
    preview" - which is meaningful here in a way it isn't in
    track_and_chain: every id is already decided by this point, so what's on
    screen is the actual, final call this run made for every player, not a
    raw, not-yet-consolidated BoT-SORT id.

    tracklets/meta let a caller in the same process (trackplayers_offline)
    hand off track_and_chain's output directly; leave them as None to load
    TRACKLETS_RAW_NAME from disk instead, which is what makes this stage
    independently re-runnable - re-tune MERGE_MAX_COST or similar, or just
    turn the preview on, and replay as many times as needed without
    repeating track_and_chain's expensive detect+track+embed pass.
    """
    if tracklets is None:
        tracklets, meta = _load_tracklets(output_path)

    fps = meta["fps"]
    frame_width = meta["frame_width"]
    frame_height = meta["frame_height"]
    diagonal = float(np.hypot(frame_width, frame_height))

    # Which side of the net each tracklet dominantly occupies - used by
    # _consolidate's side-mismatch disqualification and by
    # _enforce_roster_cap's 6-per-side check (see NET_X_M/MAX_PLAYERS_PER_SIDE).
    # None (no disqualification, no cap) when there's no calibration yet -
    # see is_in_play_area's own docstring for the same "no calibration ->
    # don't filter" tradeoff.
    homography = load_homography(output_path)

    started = time.monotonic()

    clusters, non_player_ids, nodes, graph_edges, merge_log = _consolidate(tracklets, diagonal, homography)
    clusters.sort(key=lambda c: c["entries"][0][0])

    # clusters[i] becomes stable_id i+1 below and in _save_identity_graph, so
    # compute candidates against this exact order - each pair's (a, b) is
    # final stable_ids directly.
    #
    # Skipped entirely on a provisional pass, for the same reason the graph
    # above is: these clusters are raw tracklets, not identities, so a
    # "might be the same person" hint between two of them means nothing yet,
    # and recalibration recomputes the lot against real identities anyway.
    # It also costs the same quadratic sweep the graph does - on a 51-minute
    # 4K job's 56,120 provisional tracklets that is 1.6 billion pairs, which
    # is where a run that had already finished detecting sat for four hours.
    candidate_matches = [] if homography is None else _find_candidate_matches(clusters)
    candidate_matches_file = Path(output_path) / CANDIDATE_MATCHES_LOG_NAME
    with open(candidate_matches_file, "w") as f:
        json.dump(
            [{"a": i + 1, "b": j + 1, "confidence": round(1.0 - distance, 3)} for i, j, distance in candidate_matches],
            f, indent=2,
        )

    _save_identity_graph(nodes, graph_edges, merge_log, clusters, non_player_ids, output_path,
                         provisional=homography is None)

    elapsed = time.monotonic() - started
    print(f"consolidate_tracklets done in {elapsed:.1f}s: consolidated into {len(clusters)} player identities "
          f"({len(non_player_ids)} flagged likely_non_player), {len(candidate_matches)} candidate same-person hint(s)")

    _replay_and_export(video_path, output_path, clusters, non_player_ids, fps, frame_width, frame_height,
                        show_preview, save_video)

    return clusters


def trackplayers_offline(video_path, output_path, show_preview=False, save_video=False):
    """
    Single-process convenience wrapper running track_and_chain and
    consolidate_tracklets back to back - this is what the production job
    pipeline calls (API/stage_runner.py's "player_tracking" stage). See the
    module docstring for the two-stage design, and consolidate_tracklets for
    what show_preview/save_video actually control (there's nothing stable to
    show until consolidation has run).

    For iterating on matching/consolidation - retuning a constant, replaying
    the "player identification preview", or both - call track_and_chain once
    and consolidate_tracklets as many times as needed afterward (it reloads
    from TRACKLETS_RAW_NAME on its own) rather than this wrapper, which
    always repeats both stages. RunEverything.py does exactly that.
    """
    run_started = time.monotonic()

    tracklets, meta = track_and_chain(video_path, output_path)
    consolidate_tracklets(
        video_path, output_path, tracklets=tracklets, meta=meta,
        show_preview=show_preview, save_video=save_video,
    )

    print(f"Total: {time.monotonic() - run_started:.1f}s")
