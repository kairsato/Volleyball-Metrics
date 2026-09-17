# Demo assets

The root `README.md` links to the GIFs in this folder. Replace any of them by dropping in a new file with the same name, and it picks up automatically, no other change needed.

## Analysis pipeline

One GIF per stage, each isolated to just that stage's overlay (no minimap, no other detections bleeding in):

- `demo-actions.gif` — action classifications (serve/set/spike/dig/block) called out live, bounding box + label only. Owned/recorded separately from the rest of this set — if it still shows the old combined minimap+player-tracking style, that's a pending re-record, not a docs bug.
- `demo-ball-detection.gif` — the ball tracked across the court, trail + current position only
- `demo-court-detection.gif` — the detected court boundary/net line overlaid on a live frame
- `demo-player-detection.gif` — bounding boxes around every tracked player, nothing else
- `demo-game-status-detection.gif` — the PLAY / NO PLAY / SERVE state banner over a rally that includes a serve

Generated (except `demo-actions.gif`) by replaying an already-completed job's JSON logs over its source video via `docs/assets/tools/render_stage_demos.py` (currently recorded against job `b44b4bd3ab2d`) — see that script's own docstring for usage. It does not touch the production rendering pipeline (`PostProcessing/renderVideo.py`).

## Frontend

- `frontend-games.gif` — the Games page (renamed from Videos): the job grid, fully loaded
- `frontend-teams.gif` — the Teams page: roster list
- `frontend-players.gif` — the Players page: full player grid loaded
- `frontend-game-results.gif` — a game's results view (job `399152820483`, the main reference job for this flow - large, 3 sets, every rendition tier present), recorded at a real 2K desktop size (2560x1440) rather than the smaller viewport the other flows use, cycling every tab (Stats / Analytics / Rallies / Actions / Setup) and scrolling each one's own content

Games/Teams/Players/Game Results all got much faster to record (and to actually use) after two backend changes worth knowing about if you're re-recording:
- `Backend/API/resultcache.py`'s `ttl_cache` now wraps every expensive cross-job computation (`team_stats.compute_team_stats`/`compute_player_radar`, `teams.build_matchup`, `action_quality.compute_action_quality`, `players.list_players`) - a 60s TTL, not just `players.load_player_positions`'s own mtime-keyed parse cache from before. Warm calls on all of these are now low-single-digit milliseconds instead of multiple seconds.
- The main results page's video preview no longer defaults to streaming the untouched original upload (which can be 1GB+) - `ResultsView.tsx` now auto-picks the lowest-resolution rendition actually generated for the job (480p when present) instead of just the smallest-*file-size* one, and `results_router.get_thumbnail` reads job-grid thumbnails from that same low-res rendition instead of the original.

## Calibration

From the Setup tab, into each sub-page:

- `frontend-court-calibration.gif` — dragging a calibration point handle (with the magnifier loupe)
- `frontend-player-id.gif` — selecting unidentified player tiles and opening the name picker
- `frontend-scoring.gif` — Method dropdown → Computer Vision, picking Team 1, and clicking timeline cells in the score track editor to cycle a rally's winner

None of these three ever click a final "Set/Confirm/Redo" button — they demonstrate the interaction on a real job without persisting a save, since the job they're recorded against (`09735a8b2de0`) is real user data, not a synthetic fixture.

## Privacy

Only `frontend-players.gif` redacts anything: real player first names on the Players page are blurred via a scoped CSS `filter` on the name caption (`{ names: true }` in `record-frontend-demos.mjs`'s `blurSensitiveMedia`) — not DOM text mutation, which fights React's own reconciliation and can hang the heavier pages. Every other GIF here shows real faces/footage/photos as recorded, by request.

If that scope ever needs to widen again, note for next time: CSS `filter: blur()` does not actually render on this app's `blob:`-URL-sourced `<img>`/`<video>` (the calibration frame image, unidentified-player thumbnails) in this environment — `getComputedStyle` reports the right value but the painted pixels come back sharp even with GPU compositing disabled. A pixelation-overlay-canvas workaround for that case previously lived in `record-calibration-demos.mjs`; it's gone now, but the approach (and the reason a one-shot version gets wiped by this app's frequent re-renders) is preserved in that file's git history if needed again.

Keep any replacement under a few MB (crop tightly, ~5-10s loop, reduce fps/colors) so the README stays fast to load.
