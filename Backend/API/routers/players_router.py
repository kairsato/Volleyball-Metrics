from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from starlette.concurrency import run_in_threadpool

from .. import config
from ..services import player_profiles, players, roster
from ..jobs import store
from ..schemas import (
    CandidateGroupOut,
    CandidateMatchOut,
    NamesUpdateIn,
    NamesUpdateOut,
    PlayerConfirmIn,
    PlayerOut,
    PlayersListOut,
)

router = APIRouter(prefix="/api/jobs/{job_id}/players", tags=["players"])


def _require_job(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")


def _get_players_sync(job_id: str) -> PlayersListOut:
    video_path = config.find_input_video(job_id)
    output_path = config.output_dir(job_id)

    if video_path is None or not (output_path / players.PLAYER_POSITIONS_NAME).exists():
        raise HTTPException(
            status_code=409,
            detail="Player tracking hasn't finished yet - process the job first.",
        )

    # Nothing has been matched yet, so there is nothing worth decoding a
    # thumbnail for - and on an uncalibrated job that list is every
    # detection in the building, which is exactly the page that used to be
    # unusable.
    if players.identification_is_provisional(output_path):
        return PlayersListOut(job_id=job_id, players=[], provisional=True,
                              confirmed=players.load_player_confirmed(output_path))

    player_list = players.list_players(video_path, output_path)
    candidate_matches = players.load_candidate_matches(output_path)
    candidate_groups = players.build_candidate_groups(output_path)
    return PlayersListOut(
        job_id=job_id,
        players=[PlayerOut(**p) for p in player_list],
        candidate_matches=[CandidateMatchOut(**m) for m in candidate_matches],
        candidate_groups=[CandidateGroupOut(**g) for g in candidate_groups],
        confirmed=players.load_player_confirmed(output_path),
    )


@router.get("", response_model=PlayersListOut)
async def get_players(job_id: str):
    _require_job(job_id)
    # list_players does synchronous disk I/O and, on an uncached thumbnail,
    # a video seek + OpenCV encode - run it off the event loop thread so N
    # concurrent calls (Players/Teams pages fetch every completed job's
    # players at once) can actually overlap instead of queueing behind each
    # other one at a time, blocking the whole server meanwhile.
    return await run_in_threadpool(_get_players_sync, job_id)


def _remember_in_gallery(video_path, output_path, stable_id_str: str, name: str) -> None:
    """The embedding lookup this wraps re-reads player_positions.json (can be
    hundreds of MB for a long match) and does a video-frame crop per player -
    slow enough that doing it inline used to make every single "Assign" click
    feel sluggish, even though it's purely a best-effort side effect (a
    *future* video's still-unnamed detections get auto-suggested against
    this name - see player_gallery.py). Running it as a background task lets
    the response - and the name actually being saved - return immediately."""
    from ..services import player_gallery

    embedding = players.embed_player(video_path, output_path, int(stable_id_str))
    if embedding is not None:
        player_gallery.remember(name.strip(), embedding)


@router.get("/{stable_id}/frame")
async def get_player_frame(job_id: str, stable_id: int):
    """The whole video frame a player's thumbnail was cut from, boxed - what
    the magnifier on an unidentified tile opens (see
    players.extract_player_frame).

    Served as an image rather than base64 inside the players payload, so it
    costs a video seek only for the one player a reviewer actually opened.
    Folding it into GET /players would mean decoding a full frame for every
    detection in the job on every page load.
    """
    _require_job(job_id)

    video_path = config.find_input_video(job_id)
    output_path = config.output_dir(job_id)
    if video_path is None or not output_path.exists():
        raise HTTPException(status_code=409, detail="Job hasn't produced any output yet.")

    jpeg = await run_in_threadpool(players.extract_player_frame, video_path, output_path, stable_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail=f"No usable frame for player #{stable_id}.")

    # Immutable for a given (job, stable_id): the chosen frame only changes
    # when tracking is re-run, which renumbers stable_ids anyway.
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400"})


@router.put("/names", response_model=NamesUpdateOut)
async def update_names(job_id: str, body: NamesUpdateIn, background_tasks: BackgroundTasks):
    _require_job(job_id)

    output_path = config.output_dir(job_id)
    if not output_path.exists():
        raise HTTPException(status_code=409, detail="Job hasn't produced any output yet.")

    # Refuse to persist a naming that puts one player in two places at once.
    # Naming two identities the same is how a reviewer says "these are one
    # person" (build_canonical_mapping then merges them), which is only
    # coherent while the two never share a frame - so this is checked before
    # anything is written rather than left to surface later as a player
    # whose name appears twice on court and whose stats are two people's
    # added together.
    #
    # Checked against the names this request would LEAVE SAVED, not against
    # the request on its own. The UI sends only the tiles a reviewer just
    # acted on, so checking the request alone caught a clash only when both
    # halves of it were named in the same click - and naming one fragment
    # now and its twin a minute later, which is the normal way a review
    # actually goes, sailed straight past it. One real job finished review
    # with thirty such pairs saved, including two Kais on court together for
    # 69 frames.
    if players.identification_is_provisional(output_path):
        raise HTTPException(
            status_code=409,
            detail=(
                "This video's players haven't been worked out yet - calibrate the court first, "
                "then recalibrate the job. Naming now would pin names to placeholder ids that "
                "re-matching is about to renumber."
            ),
        )

    merged_names = {**players.load_names(output_path), **body.names}
    conflicts = players.find_simultaneous_name_conflicts(output_path, merged_names, set(body.ignored))

    # Only the ones this request is responsible for. Conflicts that were
    # already saved are not this save's fault, and blocking on them would
    # trap a reviewer who has inherited a bad state - unable to fix it,
    # because every fix is itself a save.
    touched = {int(stable_id) for stable_id in body.names}
    conflicts = [c for c in conflicts if touched & set(c["stable_ids"])]
    if conflicts:
        detail = "; ".join(
            f"{c['name']} is on two players at once (#{c['stable_ids'][0]} and #{c['stable_ids'][1]}) "
            f"for {c['frames']} frames, first at {c['first_timestamp_s']}s"
            for c in conflicts
        )
        raise HTTPException(
            status_code=409,
            detail=(
                f"{detail}. One player cannot be in two places at once - rename or ignore one of "
                f"each pair. If both really are that player, the tracking has mixed two people "
                f"into one of them."
            ),
        )

    updated_names = players.save_names(output_path, body.names)
    updated_ignored = players.save_ignored(output_path, set(body.ignored))

    # A name typed here for the first time joins the shared roster too, so
    # it's a dropdown pick rather than a retype on every future job.
    if updated_names:
        roster.save_roster([*roster.load_roster(), *updated_names.values()])

    # See _remember_in_gallery's docstring for why this is deferred rather
    # than awaited - a video with no input file left just skips the gallery
    # update entirely rather than failing the save.
    video_path = config.find_input_video(job_id)
    if video_path is not None:
        for stable_id_str, name in body.names.items():
            if not name.strip():
                continue
            background_tasks.add_task(_remember_in_gallery, video_path, output_path, stable_id_str, name)

    return NamesUpdateOut(job_id=job_id, names=updated_names, ignored=updated_ignored)


@router.put("/confirm", response_model=PlayersListOut)
async def set_confirmed(job_id: str, body: PlayerConfirmIn, background_tasks: BackgroundTasks):
    """Separate from PUT /names above (which persists the actual naming
    work) so confirming/redoing can never accidentally touch a name or
    ignored flag, and vice versa - saving a name doesn't touch this. Same
    split score_router.set_confirmed uses for scoring."""
    _require_job(job_id)
    output_path = config.output_dir(job_id)
    if not output_path.exists():
        raise HTTPException(status_code=409, detail="Job hasn't produced any output yet.")

    players.save_player_confirmed(output_path, body.confirmed)
    # Confirming is what actually bakes the current names into
    # player_stats.json (via finalize, called right before this - see
    # UnidentifiedPlayersSection.handleConfirm) - warm the cross-game
    # player-profiles store now so the Players list/profile pages are
    # already fast on the very next visit instead of paying the rebuild
    # cost then (see player_profiles.py).
    background_tasks.add_task(player_profiles.warm)
    return await get_players(job_id)
