from typing import Optional

from pydantic import BaseModel


class JobOut(BaseModel):
    id: str
    original_filename: str
    created_at: str
    updated_at: str
    status: str
    stage: Optional[str] = None
    completed_stages: list[str] = []
    stage_durations_s: dict[str, float] = {}
    error: Optional[str] = None
    # The video's own length - None until it's been computed (see
    # jobs_router._ensure_duration).
    duration_s: Optional[float] = None
    # Video-list summary flags, computed fresh on every response rather than
    # stored on Job itself - cheap (see score.compute_summary), and this way
    # they can never drift out of sync with the player names/score data they
    # reflect. Both are always False/None for a job that isn't "complete"
    # yet, since neither player identification nor scoring is available
    # before then.
    needs_player_id: bool = False
    needs_scoring_review: bool = False
    winner_team_name: Optional[str] = None
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


class ResultsOut(BaseModel):
    job_id: str
    players: dict[str, PlayerStat]
    rallies: list[RallyOut]
    caveats: list[str]
    dashboard_available: bool
    video_available: bool


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


class ScoreConfirmIn(BaseModel):
    confirmed: bool


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
