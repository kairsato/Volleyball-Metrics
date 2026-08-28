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
