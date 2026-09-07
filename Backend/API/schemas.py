from typing import Optional

from pydantic import BaseModel


class VideoSegmentOut(BaseModel):
    id: str
    order: int
    original_filename: str
    duration_s: Optional[float] = None
    # User-provided at upload time, defaulting to the video's own recording-
    # date metadata when readable, else the upload date - see
    # video_metadata.py and jobs_router.upload_video.
    date_played: Optional[str] = None


class VideoDateIn(BaseModel):
    # ISO date string (YYYY-MM-DD) - lets a user correct an auto-detected or
    # defaulted date_played after upload. See jobs_router.set_video_date.
    date_played: str


class JobOut(BaseModel):
    id: str
    original_filename: str
    created_at: str
    updated_at: str
    status: str
    stage: Optional[str] = None
    completed_stages: list[str] = []
    stage_durations_s: dict[str, float] = {}
    # When the pipeline last actually finished running - see Job.processed_at
    # for why this is deliberately not the same thing as updated_at (which
    # bumps on essentially any change to the job, not just a completed run).
    processed_at: Optional[str] = None
    error: Optional[str] = None
    # The video's own length - None until it's been computed (see
    # jobs_router._ensure_duration). For a multi-video job this mirrors
    # segment 0's own duration_s - see `videos` below for every segment.
    duration_s: Optional[float] = None
    # Segment 0's own date_played, mirrored here for convenience - same
    # reasoning as duration_s above.
    date_played: Optional[str] = None
    # Every video segment making up this job, in order - length 1 for an
    # ordinary single-video job. See Backend/API/jobs.py's job_videos().
    videos: list[VideoSegmentOut] = []
    # Only meaningful when len(videos) > 1 - see calibration_router.py /
    # CalibrationPanel.tsx's "segmented court selections" toggle.
    segmented_calibration: bool = False
    # Video-list summary flags, computed fresh on every response rather than
    # stored on Job itself - cheap (see score.compute_summary), and this way
    # they can never drift out of sync with the player names/score data they
    # reflect. Both are always False/None for a job that isn't "complete"
    # yet, since neither player identification nor scoring is available
    # before then.
    needs_player_id: bool = False
    needs_scoring_review: bool = False
    winner_team_name: Optional[str] = None
    # Whether a warmup period has been confirmed for this video (see
    # warmup.py) - when True, warmup_start_s/warmup_end_s below are the
    # resolved absolute-video-time bounds every viewing surface (the player,
    # thumbnails, rallies/stats) restricts itself to. Both null when no
    # warmup period is confirmed, meaning the whole video is in play as
    # before this feature existed.
    warmup_confirmed: bool = False
    warmup_start_s: Optional[float] = None
    warmup_end_s: Optional[float] = None
    # 1-based position in the pipeline queue while this job is still
    # waiting for a worker to actually pick it up (see
    # pipeline.queue_position) - None once it's running, done, or never
    # queued at all.
    queue_position: Optional[int] = None


class PlayerOut(BaseModel):
    stable_id: int
    name: Optional[str] = None
    appearances: int
    thumbnail_base64: Optional[str] = None
    # Only set when the chosen thumbnail frame has another player's box
    # crowding into the crop (see players._has_conflict) - the same crop
    # with a white highlight box around the actual subject, so the
    # player-identification page (the only place this is ever shown) can
    # tell a human which of two people in frame is being named.
    identification_thumbnail_base64: Optional[str] = None
    thumbnail_frame_idx: Optional[int] = None
    thumbnail_timestamp_s: Optional[float] = None
    ignored: bool = False


class CandidateMatchOut(BaseModel):
    a: int
    b: int
    confidence: float


class PlayersListOut(BaseModel):
    job_id: str
    players: list[PlayerOut]
    candidate_matches: list[CandidateMatchOut] = []
    # Whether the user has explicitly signed off on this video's player
    # identification - see players.load_player_confirmed.
    confirmed: bool = False


class PlayerConfirmIn(BaseModel):
    confirmed: bool


class NamesUpdateIn(BaseModel):
    # stable_id (as string, since JSON object keys are strings) -> name.
    # An empty/blank name clears that player's name.
    names: dict[str, str]
    # stable_ids to exclude entirely from stats (referee, coach, a false
    # detection, etc.) - int keys here since these never need a display name.
    ignored: list[int] = []


class NamesUpdateOut(BaseModel):
    job_id: str
    names: dict[str, str]
    ignored: list[int] = []


class RosterOut(BaseModel):
    players: list[str]


class RosterAddIn(BaseModel):
    name: str


# A user-authored named group of roster players (e.g. "Varsity") - distinct
# from TeamPlayerOut/MatchupOut above, which are a per-video geometric
# Team A/B split inferred from tracked court position, not a persistent
# named group.
class TeamEntryOut(BaseModel):
    id: str
    name: str
    players: list[str] = []


class TeamRosterOut(BaseModel):
    teams: list[TeamEntryOut]


class TeamSaveIn(BaseModel):
    name: str
    players: list[str] = []


class Point(BaseModel):
    x: float
    y: float


class CalibrationPointsOut(BaseModel):
    job_id: str
    middle_left: Point
    middle_right: Point
    far_left: Point
    far_right: Point
    # The net's top edge above middle_left/middle_right - optional (a
    # preset guess is always returned, see calibration.default_points), but
    # only counted as a real calibration once dragged onto the actual net -
    # see net_top_calibrated. Together with the 4 ground points, these give
    # enough known 3D reference points (2 heights) to solve the camera's
    # full pose (see calibration.py/CourtDefinition.court.
    # estimate_camera_pose), which is what makes ball-height estimation
    # possible at all.
    net_top_left: Point
    net_top_right: Point
    net_height_m: float
    calibrated: bool
    # Whether the saved calibration is locked - distinct from `calibrated`
    # (which just means points exist): "Redo Court Identification" flips
    # this to False without clearing the points, same confirmed/locked/redo
    # pattern as scoring and player identification (see
    # ScoreConfigOut.confirmed / PlayersListOut.confirmed).
    confirmed: bool = False
    # Whether net_top_left/net_top_right were actually dragged onto the net
    # (vs. left at their untouched preset guess) - camera-pose/ball-height
    # estimation is skipped entirely when this is False, same spirit as the
    # desktop tool's own net_calibrated flag (CourtDefinition.court.
    # save_calibration).
    net_top_calibrated: bool = False
    # Whether a camera pose was actually solved from the 6 points (requires
    # net_top_calibrated) and is usable for ball-height estimation - None
    # when net_top_calibrated is False, otherwise reflects whether
    # estimate_camera_pose actually succeeded.
    camera_pose_available: Optional[bool] = None
    # Mean reprojection error (pixels) of the solved camera pose against the
    # 6 marked points - a rough, visible confidence signal for how much to
    # trust height estimates from this calibration (single-view pose
    # recovery has no external ground truth to validate against otherwise).
    camera_pose_reprojection_error_px: Optional[float] = None
    # The near baseline's two corners and both attack lines, derived from
    # the 4 points above (see court.predict_court_geometry) - only present
    # once calibrated, and purely a preview: never something sent back in
    # CalibrationIn.
    predicted: Optional[dict[str, Point]] = None


class CalibrationConfirmIn(BaseModel):
    confirmed: bool


class WarmupConfigOut(BaseModel):
    job_id: str
    start_s: float
    # None means "the end of the video" - still resolves to a concrete
    # value via duration_s below for anything that needs one (e.g. a range
    # slider's max), without baking the video's length into the saved
    # config itself.
    end_s: Optional[float] = None
    confirmed: bool
    duration_s: Optional[float] = None


class WarmupConfigIn(BaseModel):
    start_s: float
    end_s: Optional[float] = None


class WarmupConfirmIn(BaseModel):
    confirmed: bool


class CalibrationIn(BaseModel):
    middle_left: Point
    middle_right: Point
    far_left: Point
    far_right: Point
    net_top_left: Point
    net_top_right: Point
    net_height_m: float


class PlayerEvent(BaseModel):
    frame_idx: int
    timestamp_s: float
    action_type: str
    rally_index: Optional[int] = None
    # Ball speed immediately before/after this touch, in real-world m/s when
    # real_units is True, otherwise px/s (see actionDetection.py/
    # ball_speed.json's own identical real_units pattern) - None when the
    # underlying ball_speed.json reading itself wasn't available at this
    # frame.
    speed_in_m_per_s: Optional[float] = None
    speed_out_m_per_s: Optional[float] = None
    real_units: bool = False
    # Estimated height (metres) of the ball at this touch - only ever
    # populated when this job's calibration includes a solved camera pose
    # (see CalibrationPointsOut.camera_pose_available); None otherwise.
    ball_height_m: Optional[float] = None
    # Gap since the previous touch in the same rally (seconds) - None for a
    # rally's first recorded touch.
    time_since_prev_touch_s: Optional[float] = None


class PlayerStat(BaseModel):
    total_hits: int
    hits_by_type: dict[str, int]
    rallies_participated: int
    rally_ending_touches: int
    rally_ending_touch_rate: Optional[float] = None
    name: Optional[str] = None
    events: list[PlayerEvent] = []


class RallyOut(BaseModel):
    rally_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float


class QualitiesOut(BaseModel):
    # "original" (the untouched uploaded file) is always first, followed by
    # whichever downscaled renditions actually got generated (see
    # transcode.py's TRANSCODE_TIERS/generate_renditions) - never includes a
    # tier that wasn't actually produced (source too small, job predates
    # this feature, transcoding failed).
    qualities: list[str]


class ResultsOut(BaseModel):
    job_id: str
    players: dict[str, PlayerStat]
    rallies: list[RallyOut]
    caveats: list[str]
    dashboard_available: bool
    video_available: bool


class BallTrajectoryPointOut(BaseModel):
    # Seconds, relative to the warmup period's start (or the raw video's
    # start if no warmup period is confirmed) - same basis as every other
    # timestamp the frontend player deals in, see warmup.py.
    t: float
    # Real-world court coordinates in metres - same coordinate system as
    # ActionQualityCategory instances' ball_court field, origin/axes per
    # CourtDefinition.court's own convention. Can fall outside the
    # court_length_m x court_width_m rectangle below (e.g. a serve from
    # behind the baseline) - the frontend clamps/clips for display. Used
    # for the small corner minimap (VideoPlayer's Minimap annotation).
    # None when this job has no court calibration yet - the on-video Ball
    # tracking annotation (px/py below) doesn't need it, only the minimap
    # does, so a point isn't dropped entirely just for lacking this.
    x: Optional[float] = None
    y: Optional[float] = None
    # Raw video-frame pixel coordinates of the same detection, when the
    # frame it came from had one (None for a court-projected-only point) -
    # used for VideoPlayer's Ball tracking annotation, which draws directly
    # on the video at the ball's actual on-screen position rather than a
    # corner diagram. Unlike x/y this is in the SOURCE video's own pixel
    # space, not metres - the frontend maps it to displayed coordinates
    # itself (see BallTrackingOverlay.tsx).
    px: Optional[float] = None
    py: Optional[float] = None
    # Estimated height (metres) above the court's ground plane - same
    # estimate_ball_height reading ball_speed.json itself carries, only
    # ever populated when this job's calibration includes a solved camera
    # pose. Used by BallTrajectoryOverlay's arc annotation to clip the
    # drawn curve once the ball descends back below net height.
    height_m: Optional[float] = None


class BallTrajectoryOut(BaseModel):
    job_id: str
    points: list[BallTrajectoryPointOut] = []
    # Metres - CalibrationPointsOut's own net_height_m when this job has
    # been calibrated, CourtDefinition.court.NET_HEIGHT_M (the standard
    # men's height) otherwise, same fallback calibration.py itself uses
    # before a net height is explicitly set.
    net_height_m: float
    court_length_m: float
    court_width_m: float
    # The ORIGINAL uploaded video's own pixel resolution - what every
    # px/py above is actually measured in (see BallTrajectoryPointOut).
    # Deliberately NOT the currently-selected playback quality's resolution:
    # a viewer can switch to a lower-bitrate rendition (see
    # transcode.TRANSCODE_TIERS) while watching, which changes the <video>
    # element's own decoded videoWidth/videoHeight but never touches these
    # already-detected coordinates. An overlay that sized its SVG viewBox
    # from the playing element instead of this field would silently
    # misplace every annotation the moment someone picked a lower quality -
    # the two only ever agreed by coincidence, when "original" happened to
    # still be selected. None for a job whose ball_trajectory.json hasn't
    # been written yet (ball_detection hasn't completed).
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None


class PlayerBoxOut(BaseModel):
    stable_id: int
    # None for a still-unnamed detection - the overlay falls back to
    # "#<stable_id>", same label the server-rendered analysis.mp4 uses.
    name: Optional[str] = None
    # [x1, y1, x2, y2] in the source video's own pixel space - same mapping
    # concern as BallTrajectoryPointOut.px/py above.
    box: list[float]
    # Real-world court position in metres, same coordinate system as
    # BallTrajectoryPointOut.x/y - projected from the box's BOTTOM-CENTRE
    # (the player's feet), not its centre: the homography maps the court's
    # ground plane, so only a point actually on that plane projects to where
    # the player is really standing. player_positions.json's own stored
    # "court" field is centre-derived and lands well off-court for exactly
    # that reason, so it deliberately isn't reused here. None when this job
    # has no court calibration yet.
    court_x: Optional[float] = None
    court_y: Optional[float] = None


class PlayerTrajectoryFrameOut(BaseModel):
    # Same relative-to-warmup-start basis as BallTrajectoryPointOut.t.
    t: float
    players: list[PlayerBoxOut] = []


class PlayerTrajectoryOut(BaseModel):
    job_id: str
    frames: list[PlayerTrajectoryFrameOut] = []
    # Same meaning, and same reason it exists, as BallTrajectoryOut.frame_w/
    # frame_h - the resolution every box above is measured in, independent
    # of whatever playback quality is currently selected.
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None


class TeamPlayerOut(BaseModel):
    stable_id: int
    name: Optional[str] = None
    avg_court_x: float


class RallyOutcomeOut(BaseModel):
    rally_index: int
    start_time_s: float
    end_time_s: float
    winning_team: Optional[str] = None


class RadarPointOut(BaseModel):
    action_type: str
    team_a_win_rate: Optional[float] = None
    team_b_win_rate: Optional[float] = None
    sample_size_a: int = 0
    sample_size_b: int = 0


class MatchupOut(BaseModel):
    job_id: str
    available: bool
    reason: Optional[str] = None
    team_a: list[TeamPlayerOut] = []
    team_b: list[TeamPlayerOut] = []
    rallies: list[RallyOutcomeOut] = []
    wins_a: int = 0
    wins_b: int = 0
    radar: list[RadarPointOut] = []
    caveats: list[str] = []


# Which named roster team (team_roster.py) won each rally, and how rallies
# group into games/sets - distinct from MatchupOut's anonymous, whole-video
# "Team A/B" geometric split above. See score.py's module docstring.
class ScoreConfigOut(BaseModel):
    method: str = "none"
    team_x_id: Optional[str] = None
    team_y_id: Optional[str] = None
    ocr_region: Optional[dict] = None
    # Only meaningful for method "ocr": whether the left-to-right digit
    # order read from the marked region should be flipped before being
    # treated as (geometric side A, geometric side B) - see score_cv.py's
    # module docstring for the "sideline view" assumption this corrects
    # for when the camera/region has the two sides mirrored.
    cv_reverse_direction: bool = False
    # Minimum easyocr confidence (0-1) a digit-run detection needs before
    # score_cv counts it - see score.DEFAULT_CONFIG's docstring.
    ocr_min_confidence: float = 0.0
    compute_status: str = "idle"
    compute_error: Optional[str] = None
    # Whether the user has explicitly signed off on the scoring shown in
    # the track editor - see score.compute_summary's docstring.
    confirmed: bool = False


class ScoreConfigIn(BaseModel):
    method: str
    team_x_id: Optional[str] = None
    team_y_id: Optional[str] = None
    ocr_region: Optional[dict] = None
    cv_reverse_direction: bool = False
    ocr_min_confidence: float = 0.0


class ScoreConfirmIn(BaseModel):
    confirmed: bool


class OcrRegionIn(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ScoreRegionTestIn(BaseModel):
    t: float
    region: OcrRegionIn
    min_confidence: float = 0.0


class OcrDetectionOut(BaseModel):
    text: str
    confidence: float


class ScoreRegionTestOut(BaseModel):
    left: Optional[int] = None
    left_confidence: Optional[float] = None
    right: Optional[int] = None
    right_confidence: Optional[float] = None
    # Every raw OCR detection in the crop, not just the two digit runs
    # picked out as left/right - lets a human see what the model actually
    # saw (a misread digit, a stray label, nothing at all) even when
    # left/right come back None.
    detections: list[OcrDetectionOut] = []


class GameOut(BaseModel):
    game_index: int
    start_rally_index: int
    end_rally_index: int


class RallyWinnerOut(BaseModel):
    rally_index: int
    game_index: int
    winner: Optional[str] = None
    confidence: str = "manual"
    # The raw digits the Computer Vision method actually read for this
    # rally, if any - saved as a reference only (see score_cv.py's module
    # docstring); winner/confidence above never depend on these being
    # present. Always None for manual/automatic-method rallies.
    cv_left: Optional[int] = None
    cv_right: Optional[int] = None


class RallyWinnerIn(BaseModel):
    winner: Optional[str] = None


# range_start/range_end are inclusive, 0-based, and mean game_index for
# "games" or rally_index for "rallies" - unused (and ignored) for "match".
class ScoreComputeIn(BaseModel):
    range_type: str = "match"
    range_start: Optional[int] = None
    range_end: Optional[int] = None


class GameBoundaryIn(BaseModel):
    split: bool


class ScoreResultOut(BaseModel):
    games: list[GameOut] = []
    rallies: list[RallyWinnerOut] = []


class ScoreOut(BaseModel):
    job_id: str
    config: ScoreConfigOut
    result: Optional[ScoreResultOut] = None


class ActionQualityInstanceOut(BaseModel):
    frame_idx: int
    timestamp_s: float
    rally_index: Optional[int] = None
    player_stable_id: Optional[int] = None
    # "A"/"B" (teams.assign_teams's anonymous geometric split) - None
    # whenever team splitting itself is unavailable for this job (see
    # ActionQualityOut.teams_available) or couldn't be determined for this
    # specific player.
    team: Optional[str] = None
    # Named 0-1 sub-scores (e.g. "speed"/"placement"/"trajectory_height" for
    # a serve) - a factor's value is None when it couldn't be computed for
    # this instance (missing team/height data, no next touch to measure
    # placement against, etc.); see action_quality.py's per-action-type
    # weights for which keys exist per action type.
    factors: dict[str, Optional[float]]
    # Weighted average of `factors` (see action_quality._weighted_score) -
    # None only when every single factor was unavailable.
    overall_score: Optional[float] = None
    # Raw values a couple of the factors above are derived from - (x, y)
    # metres, same convention as CalibrationPointsOut's court geometry, net
    # at x=9.0 (action_quality.NET_X_M). None under the same conditions the
    # corresponding factor is None (no ball_court reading, no camera pose,
    # etc.) - a UI wanting an actual position/number (e.g. a court map)
    # reads these instead of the normalized 0-1 factor.
    ball_court: Optional[list[float]] = None
    time_since_prev_touch_s: Optional[float] = None
    ball_height_m: Optional[float] = None


class ActionQualityPlayerOut(BaseModel):
    stable_id: int
    average_score: Optional[float] = None
    count: int


class ActionQualityCategoryOut(BaseModel):
    # Fixed weights this category's overall_score values were computed
    # with (see action_quality.py's SERVE_WEIGHTS/RECEIVE_WEIGHTS/
    # SET_WEIGHTS/SPIKE_WEIGHTS) - surfaced so a UI can label which factors
    # contributed and how much, not just show a bare percentage.
    weights: dict[str, float]
    # Fixed "ideal" values a UI can compare a raw instance value against
    # (e.g. Set's {"height_ideal_m": 3.0, "time_reference_s": 1.5}) -
    # mirrors action_quality.py's own reference constants so the frontend
    # never has to hardcode a number that could drift from them. Empty for
    # every category without a meaningful single "ideal" to show.
    reference: dict[str, float] = {}
    count: int
    average_score: Optional[float] = None
    players: list[ActionQualityPlayerOut] = []
    instances: list[ActionQualityInstanceOut] = []


class ActionQualityOut(BaseModel):
    job_id: str
    # Whether teams.assign_teams found two well-tracked sides for this job -
    # when False, every team-relative factor (placement, blockers) is None
    # throughout and each category's overall_score is renormalized across
    # whatever's left, same "available" pattern as MatchupOut.
    teams_available: bool
    # Whether this job's calibration has a solved camera pose (net-top
    # points marked and pose-solving succeeded) - when False, every
    # trajectory-height factor is None throughout.
    height_available: bool
    serve: ActionQualityCategoryOut
    receive: ActionQualityCategoryOut
    set: ActionQualityCategoryOut
    spike: ActionQualityCategoryOut
    caveats: list[str] = []


class TeamPlayerSummaryOut(BaseModel):
    name: str
    total_hits: int
    hits_by_type: dict[str, int] = {}


class TeamVideoSummaryOut(BaseModel):
    job_id: str
    original_filename: str
    game_wins: int
    game_losses: int


class TeamStatsOut(BaseModel):
    team_id: str
    team_name: str
    # Every complete job with at least one roster member named in its
    # player_stats.json contributes to total_hits/hits_by_type/players
    # below, regardless of whether Scoring was ever configured for it.
    videos_total: int
    # Only jobs where Scoring was configured with this team as team_x/
    # team_y contribute to match/game records and the radar - see
    # team_stats.py's module docstring for why (team_roster.py has no
    # inherent link to any per-video geometric side without it).
    videos_with_scoring: int
    match_wins: int
    match_losses: int
    game_wins: int
    game_losses: int
    radar: list[RadarPointOut] = []
    total_hits: int
    hits_by_type: dict[str, int] = {}
    players: list[TeamPlayerSummaryOut] = []
    videos: list[TeamVideoSummaryOut] = []


class PlayerRadarOut(BaseModel):
    name: str
    videos_with_data: int
    radar: list[RadarPointOut] = []


class AuthStatusOut(BaseModel):
    # Whether login is turned on globally - drives the Share dialog's own
    # UI (the switch, password form, etc.), NOT whether any particular
    # caller needs to authenticate - see login_required below for that.
    enabled: bool
    password_set: bool
    # ISO timestamp of when login will auto-disable itself (see
    # auth.check_share_expired) - None while login is off, or for a config
    # saved before this field existed.
    expires_at: Optional[str] = None
    # Whether THIS caller specifically needs to log in - true only when
    # `enabled` is true AND this request isn't from the LAN (see
    # network.is_lan_request/main.py's AuthMiddleware, which has the same
    # LAN bypass). LoginGate.tsx reads this, not `enabled`, to decide
    # whether to show the login screen at all - a LAN visitor should never
    # see one, even while Share is on for remote visitors.
    login_required: bool = False
    # Whether THIS caller is on the LAN - independent of whether login is
    # even turned on (unlike login_required above, which is always false
    # while `enabled` is false). SettingsMenu reads this to hide the Share
    # menu item entirely for a remote visitor, not just block what it does -
    # every /api/share/* call a remote visitor could reach from it is
    # already blocked server-side (see main.py's LAN_ONLY_PATH_PREFIXES),
    # this is purely about not showing a menu item that would just error.
    is_lan: bool = True


class CaptchaOut(BaseModel):
    captcha_id: str
    image_base64: str


class LoginIn(BaseModel):
    password: str
    captcha_id: str
    captcha_answer: str


class LoginOut(BaseModel):
    token: str


class SetPasswordIn(BaseModel):
    password: str


class SetEnabledIn(BaseModel):
    enabled: bool
    # Only meaningful when enabled=True: how many days until Share
    # auto-disables itself, or None for "Forever" (never auto-expires) - see
    # the Share dialog's duration dropdown. Ignored when disabling.
    duration_days: Optional[int] = 7


class SuggestedPasswordOut(BaseModel):
    # A fresh strong password to preview - purely a suggestion, nothing is
    # saved server-side until the user actually submits it via /set-password.
    password: str


class PortStatusOut(BaseModel):
    port: int
    # "open" (UPnP mapped this port), "error" (UPnP is on but this specific
    # port failed to map), or "not_forwarded" (UPnP is off - may still be
    # reachable if forwarded manually, we just didn't do it).
    status: str


class ShareStatusOut(BaseModel):
    upnp_enabled: bool
    external_ip: Optional[str] = None
    local_ip: str
    hostname: Optional[str] = None
    ports: list[PortStatusOut]
    share_url: str
    last_error: Optional[str] = None


class HostnameIn(BaseModel):
    # None/blank clears it - see share.set_hostname.
    hostname: Optional[str] = None
