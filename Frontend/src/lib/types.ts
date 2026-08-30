export type JobStatus =
  | "uploaded"
  | "processing"
  | "awaiting_player_review"
  | "finalizing"
  | "complete"
  | "error"
  | "cancelled";

export interface Job {
  id: string;
  original_filename: string;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  stage: string | null;
  completed_stages: string[];
  stage_durations_s: Record<string, number>;
  error: string | null;
  // The uploaded video's own length in seconds - null until the backend has
  // computed it (see Backend/API/routers/jobs_router.py's _ensure_duration).
  duration_s: number | null;
  // Video-list summary flags - always false/null for a job that isn't
  // "complete" yet, since neither player identification nor scoring exists
  // before then.
  needs_player_id: boolean;
  needs_scoring_review: boolean;
  winner_team_name: string | null;
}

export interface Player {
  stable_id: number;
  name: string | null;
  appearances: number;
  thumbnail_base64: string | null;
  thumbnail_frame_idx: number | null;
  thumbnail_timestamp_s: number | null;
  ignored: boolean;
}

export interface CandidateMatch {
  a: number;
  b: number;
  confidence: number;
}

export interface PlayersListOut {
  job_id: string;
  players: Player[];
  candidate_matches: CandidateMatch[];
}

export interface NamesUpdateOut {
  job_id: string;
  names: Record<string, string>;
  ignored: number[];
}

export interface PlayerEvent {
  frame_idx: number;
  timestamp_s: number;
  action_type: string;
  rally_index: number | null;
}

export interface PlayerStat {
  total_hits: number;
  hits_by_type: Record<string, number>;
  rallies_participated: number;
  rally_ending_touches: number;
  rally_ending_touch_rate: number | null;
  name: string | null;
  events: PlayerEvent[];
}

export interface Point {
  x: number;
  y: number;
}

export interface CalibrationPointsOut {
  job_id: string;
  corners: Point[];
  net_points: Point[];
  calibrated: boolean;
}

export interface Rally {
  rally_index: number;
  start_time_s: number;
  end_time_s: number;
  duration_s: number;
}

export interface ResultsOut {
  job_id: string;
  players: Record<string, PlayerStat>;
  rallies: Rally[];
  caveats: string[];
  dashboard_available: boolean;
  video_available: boolean;
}

export interface TeamPlayer {
  stable_id: number;
  name: string | null;
  avg_court_x: number;
}

export interface RallyOutcome {
  rally_index: number;
  start_time_s: number;
  end_time_s: number;
  winning_team: "A" | "B" | null;
}

export interface RadarPoint {
  action_type: string;
  team_a_win_rate: number | null;
  team_b_win_rate: number | null;
  sample_size_a: number;
  sample_size_b: number;
}

export interface RosterOut {
  players: string[];
}

// A user-authored named group of roster players - distinct from
// MatchupOut's TeamPlayer below, which is a per-video geometric Team A/B
// split inferred from tracked court position, not a persistent named group.
export interface TeamEntry {
  id: string;
  name: string;
  players: string[];
}

export interface TeamRosterOut {
  teams: TeamEntry[];
}

export interface MatchupOut {
  job_id: string;
  available: boolean;
  reason: string | null;
  team_a: TeamPlayer[];
  team_b: TeamPlayer[];
  rallies: RallyOutcome[];
  wins_a: number;
  wins_b: number;
  radar: RadarPoint[];
  caveats: string[];
}

export type ScoreMethod = "none" | "manual" | "automatic" | "ocr";

export interface OcrRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ScoreConfig {
  method: ScoreMethod;
  team_x_id: string | null;
  team_y_id: string | null;
  ocr_region: OcrRegion | null;
  // Only meaningful for method "ocr" - flips which geometric side the
  // left-read digit is treated as belonging to.
  cv_reverse_direction: boolean;
  compute_status: "idle" | "computing" | "done" | "error";
  compute_error: string | null;
  // Whether the user has explicitly signed off on the scoring shown in
  // the track editor - persisted so it survives a reload, and drives the
  // Setup tab's "Done"/"Needs attention" chip for Scoring Determination.
  confirmed: boolean;
}

export interface Game {
  game_index: number;
  start_rally_index: number;
  end_rally_index: number;
}

// "x"/"y" refer to the two roster teams identified in ScoreConfig
// (team_x_id/team_y_id) - unrelated to MatchupOut's anonymous, geometric
// "A"/"B" split above.
export interface RallyWinner {
  rally_index: number;
  game_index: number;
  winner: "x" | "y" | null;
  confidence: "auto" | "manual" | "uncertain";
}

export interface ScoreResult {
  games: Game[];
  rallies: RallyWinner[];
}

export interface ScoreOut {
  job_id: string;
  config: ScoreConfig;
  result: ScoreResult | null;
}
