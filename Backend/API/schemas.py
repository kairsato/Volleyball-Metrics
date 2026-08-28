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
