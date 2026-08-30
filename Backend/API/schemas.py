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
    corners: list[Point]
    net_points: list[Point]
    calibrated: bool


class CalibrationIn(BaseModel):
    corners: list[Point]
    net_points: list[Point]


class PlayerEvent(BaseModel):
    frame_idx: int
    timestamp_s: float
    action_type: str
    rally_index: Optional[int] = None


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
