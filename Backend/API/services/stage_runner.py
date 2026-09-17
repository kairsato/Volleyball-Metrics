"""Runs a single pipeline stage as a standalone process.

Invoked as `python -m API.services.stage_runner <stage> <video_path>
<output_path>` by API/services/pipeline.py, one stage at a time. Running
each stage as its own process (rather than a thread inside the API server)
is what makes cancellation actually work: a YOLO inference loop has no
cooperative "should I stop?" checks anywhere in it, so the only reliable way
to stop one mid-frame is to kill the OS process running it.
"""

import sys
import traceback
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from BallDetection import ballDetection as ball_detection_module  # noqa: E402
from BallDetection.ballDetection import detectBall, reselect_ball  # noqa: E402
from PlayerDetection import tracker as tracker_module  # noqa: E402
from PlayerDetection.tracker import consolidate_tracklets, trackplayers_offline  # noqa: E402
from GameStatusDetection import gameStatusDetection as game_status_module  # noqa: E402
from GameStatusDetection.gameStatusDetection import detectGameStatus  # noqa: E402
from ActionDetection import actionDetection as action_detection_module  # noqa: E402
from ActionDetection.actionDetection import detectActions  # noqa: E402
from PostProcessing.consolidate import consolidateStats  # noqa: E402
from PostProcessing import renderVideo as render_video_module  # noqa: E402
from PostProcessing.renderVideo import renderAnnotatedVideo  # noqa: E402
from PostProcessing.generate_dashboard import generateDashboard  # noqa: E402
from PostProcessing import transcode as transcode_module  # noqa: E402
from PostProcessing.transcode import generate_renditions  # noqa: E402
from API import config  # noqa: E402
from API.services import heuristics  # noqa: E402
from API.services import players as players_service  # noqa: E402
from API.services import score  # noqa: E402
from API.services.players import (  # noqa: E402
    identification_is_provisional,
    load_names,
    recalibrate_players,
    write_grouped_actions,
)

# Every module a Configuration-page heuristic profile can override, keyed
# by the same stage name STAGE_FUNCS below uses - see
# heuristics.apply_overrides. "recalibrate" touches both ball_detection
# (reselect_ball) and action_detection (detectActions), so overrides are
# applied for every stage here unconditionally on each run rather than only
# for the one stage actually requested (harmless: a module with none of a
# stage's constants defined is simply skipped).
HEURISTIC_STAGE_MODULES = {
    "player_tracking": tracker_module,
    "ball_detection": ball_detection_module,
    "game_status": game_status_module,
    "action_detection": action_detection_module,
    "rendering": render_video_module,
    "transcoding": transcode_module,
}


def _consolidate_with_groups(output: str):
    grouped_file = write_grouped_actions(Path(output))
    consolidateStats(output, actions_filename=grouped_file.name)


def _recalibrate(video: str, output: str):
    """Cheap re-run for a calibration change on an already-processed job -
    see pipeline.start_recalibration. Re-picks the ball from its saved raw
    candidates, re-consolidates player identities, and re-derives player
    court coordinates/auto-ignores, then re-runs action_detection (the only
    downstream stage whose output actually depends on those court
    coordinates - see actionDetection.py) against the refreshed logs. Never
    touches player_tracking's or ball_detection's own (expensive) detection
    passes.

    Re-consolidating matters because player_tracking runs BEFORE the court
    is ever calibrated (court_calibration is a Setup-tab step after phase
    one - see API/jobs.py's PHASE_ONE_STAGES), so that first pass had no
    homography and therefore could not apply any of the court-aware
    identity heuristics: which side of the net a player is on, and the
    6-per-side roster cap (see PlayerDetection.tracker's _pair_cost and
    _enforce_roster_cap). Without this re-run those heuristics would never
    fire on a real job at all. It's affordable precisely because tracker.py
    splits detection from consolidation: this reloads the saved raw
    tracklets and skips the video entirely (seconds), rather than repeating
    the detect+track+embed pass (minutes).

    It is skipped, though, once a name has been pinned to the existing
    stable_ids - typed by a human, or auto-filled from the cross-video
    gallery right after player_tracking. Re-consolidation renumbers
    identities from scratch, so running it then would silently repoint every
    saved name at a different player, which is far worse than not applying
    the heuristics. Calibrating before the job is processed avoids the
    tradeoff entirely: player_tracking then has the homography from the
    start.

    Only *names* pin, and only when there is actually one saved - the file
    merely existing does not. Both player_names.json and player_ignored.json
    are written even when they end up empty (see players.save_names /
    save_ignored), and player_ignored.json in particular is written by
    recalibrate_players' own auto-ignore pass at the end of this very
    function, so an existence check meant the first recalibration of a job
    permanently blocked every later one - including the one right after a
    "redo player identification" that just cleared the names. The ignore
    flags themselves are not worth preserving across a re-consolidation:
    their stable_ids are about to refer to different people, so carrying
    them over would ignore the wrong players. They are cleared here instead
    and re-derived from the new identities by recalibrate_players below."""
    reselect_ball(output)

    output_path = Path(output)
    names = load_names(output_path)

    # A provisional pass (see players.identification_is_provisional) has not
    # matched anything yet - it is exactly the job this re-run exists to
    # finish, so it is never skipped. Nothing can be pinned to those
    # placeholder ids either, because the API refuses to name them until a
    # court is saved.
    provisional = identification_is_provisional(output_path)

    if names and not provisional:
        print(f"recalibrate: skipping re-consolidation - {config.PLAYER_NAMES_NAME} pins "
              f"{len(names)} name(s) to the current stable_ids, which re-consolidation would "
              f"renumber. Calibrate before processing to get court-aware identity matching.")
    else:
        (output_path / config.PLAYER_IGNORED_NAME).unlink(missing_ok=True)
        # Both of these are keyed on stable_id, which consolidation is about
        # to reassign from scratch. The ignore flags would land on the wrong
        # people; a cached thumbnail would only be reused where the new
        # identity's best frame happens to match the old one's, but that is a
        # real collision across a hundred-odd identities and it shows the
        # reviewer a photo of someone else entirely.
        (output_path / players_service.THUMBNAIL_CACHE_NAME).unlink(missing_ok=True)
        consolidate_tracklets(video, output)

    recalibrate_players(output_path)
    detectActions(video, output)

    # Best-effort, same spirit as player_gallery auto-naming in
    # pipeline._phase_one: court calibration is what infer_rally_winners
    # needs was missing until just now (ball_speed.json only got real court
    # coordinates from reselect_ball above), so this is the first point in
    # a job's life an automatic score attempt can do anything at all.
    #
    # Deliberately only when method is still "none", the untouched default
    # - not just for setting the method, but for calling compute_automatic
    # at all. A job whose method already reads "automatic" got there either
    # from an earlier run of this exact check, or a human explicitly chose
    # it - either way score_result.json may already carry hand corrections
    # (see score.py's own module docstring: every automatic method is a
    # starting point a human is expected to review/correct, "confirmed"
    # or not), and compute_automatic with no rally_range recomputes EVERY
    # rally's winner, silently overwriting those. Only running it the one
    # time there is provably nothing yet to overwrite is what keeps a
    # later recalibration (a corrected court point, say) from quietly
    # discarding scoring work already done. A human who wants a fresh
    # automatic pass after that still has the Setup tab's own "Automatic"
    # button for it - this is only ever the unattended head start, never a
    # standing override. Never fatal either way - not worth failing
    # recalibration over.
    try:
        cfg = score.load_config(output_path)
        if cfg.get("method", "none") == "none":
            score.save_config(output_path, {"method": "automatic"})
            score.compute_automatic(output_path)
    except Exception:  # noqa: BLE001 - genuinely best-effort, never blocks recalibration
        traceback.print_exc()


STAGE_FUNCS = {
    "player_tracking": lambda video, output: trackplayers_offline(video, output, show_preview=False, save_video=False),
    "ball_detection": lambda video, output: detectBall(video, output, show_preview=False, save_video=False),
    "game_status": lambda video, output: detectGameStatus(video, output),
    "action_detection": lambda video, output: detectActions(video, output),
    "recalibrate": _recalibrate,
    "consolidating": lambda video, output: _consolidate_with_groups(output),
    "dashboard": lambda video, output: generateDashboard(output),
    "rendering": lambda video, output: renderAnnotatedVideo(video, output),
    # Best-effort by design (see transcode.py's module docstring) - never
    # raises, so a transcode failure can't fail the job the way every other
    # stage here still can.
    "transcoding": lambda video, output: generate_renditions(video, output),
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

    for stage_key, module in HEURISTIC_STAGE_MODULES.items():
        heuristics.apply_overrides(stage_key, module)

    func(video_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
