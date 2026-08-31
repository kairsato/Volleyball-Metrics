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
  // 1-based position in the pipeline queue while still waiting for a
  // worker to pick this job up - null once it's actually running, done,
  // or was never queued at all.
  queue_position: number | null;
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
  // Whether the user has explicitly signed off on this video's player
  // identification - same pattern as ScoreConfig.confirmed.
  confirmed: boolean;
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
  // Ball speed immediately before/after this touch - m/s when real_units is
  // true, otherwise px/s (mirrors ball_speed.json's own real_units field).
  // null when the underlying ball_speed.json reading wasn't available.
  speed_in_m_per_s: number | null;
  speed_out_m_per_s: number | null;
  real_units: boolean;
  // Estimated ball height (metres) at this touch - only ever populated when
  // this job's calibration has a solved camera pose (see
  // CalibrationPointsOut.camera_pose_available); null otherwise.
  ball_height_m: number | null;
  // Gap since the previous touch in the same rally (seconds) - null for a
  // rally's first recorded touch.
  time_since_prev_touch_s: number | null;
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

// The near baseline corners + both attack lines, derived from the 4
// points below (see Backend's court.predict_court_geometry) - only
// present once calibrated, and purely a preview for drawing: never sent
// back when saving.
export interface PredictedCourtGeometry {
  near_left: Point;
  near_right: Point;
  attack_far_left: Point;
  attack_far_right: Point;
  attack_near_left: Point;
  attack_near_right: Point;
}

export interface CalibrationPointsOut {
  job_id: string;
  middle_left: Point;
  middle_right: Point;
  far_left: Point;
  far_right: Point;
  // The net's top edge above middle_left/middle_right - together with the
  // 4 ground points, gives enough known-height reference points to solve
  // the camera's full 3D pose (see net_top_calibrated/camera_pose_available
  // below), which is what makes ball-height estimation possible at all.
  net_top_left: Point;
  net_top_right: Point;
  net_height_m: number;
  calibrated: boolean;
  confirmed: boolean;
  // Whether net_top_left/net_top_right were actually dragged onto the net
  // (vs. left at their untouched preset guess) - camera pose/ball-height
  // estimation only ever runs once this is true.
  net_top_calibrated: boolean;
  // Whether a camera pose was actually solved from the 6 points - null
  // when net_top_calibrated is false, otherwise reflects whether pose
  // solving itself succeeded.
  camera_pose_available: boolean | null;
  // Mean reprojection error (pixels) of the solved camera pose - a rough
  // confidence signal, since single-view pose recovery has no external
  // ground truth to validate against otherwise.
  camera_pose_reprojection_error_px: number | null;
  predicted: PredictedCourtGeometry | null;
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
  // The raw digits the Computer Vision method actually read for this
  // rally, if any - a reference only, never required for winner/
  // confidence above. Always null for manual/automatic-method rallies.
  cv_left: number | null;
  cv_right: number | null;
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

export interface ActionQualityInstance {
  frame_idx: number;
  timestamp_s: number;
  rally_index: number | null;
  player_stable_id: number | null;
  // "A"/"B" (MatchupOut's anonymous geometric split) - null whenever team
  // splitting is unavailable for this job, or couldn't be determined for
  // this specific player.
  team: "A" | "B" | null;
  // Named 0-1 sub-scores - which keys exist depends on the action type
  // (e.g. serve has speed/placement/trajectory_height). A factor is null
  // when it couldn't be computed for this instance.
  factors: Record<string, number | null>;
  // Weighted average of `factors` - null only when every factor was
  // unavailable.
  overall_score: number | null;
  // Raw values a couple of the factors above are derived from - (x, y)
  // metres, net at x=9.0. Null under the same conditions the corresponding
  // factor is null.
  ball_court: [number, number] | null;
  time_since_prev_touch_s: number | null;
  ball_height_m: number | null;
}

export interface ActionQualityPlayer {
  stable_id: number;
  average_score: number | null;
  count: number;
}

export interface ActionQualityCategory {
  // Fixed weights this category's overall_score values were computed with -
  // surfaced so the UI can label which factors contributed and how much.
  weights: Record<string, number>;
  // Fixed "ideal" values a UI can compare a raw instance value against
  // (e.g. Set's height_ideal_m/time_reference_s) - empty for a category
  // with no single meaningful "ideal" to show.
  reference: Record<string, number>;
  count: number;
  average_score: number | null;
  players: ActionQualityPlayer[];
  instances: ActionQualityInstance[];
}

export interface ActionQualityOut {
  job_id: string;
  // Whether team splitting found two well-tracked sides for this job -
  // when false, every team-relative factor (placement, blockers) is null
  // throughout, same "available" pattern as MatchupOut.
  teams_available: boolean;
  // Whether this job's calibration has a solved camera pose (net-top
  // points marked) - when false, every trajectory-height factor is null.
  height_available: boolean;
  serve: ActionQualityCategory;
  receive: ActionQualityCategory;
  set: ActionQualityCategory;
  spike: ActionQualityCategory;
  caveats: string[];
}

export interface QualitiesOut {
  // "original" (the untouched uploaded file) is always first, followed by
  // whichever downscaled renditions actually got generated for this job.
  qualities: string[];
}

export interface TeamPlayerSummary {
  name: string;
  total_hits: number;
  hits_by_type: Record<string, number>;
}

export interface TeamVideoSummary {
  job_id: string;
  original_filename: string;
  game_wins: number;
  game_losses: number;
}

export interface TeamStatsOut {
  team_id: string;
  team_name: string;
  // Every complete video with a roster member named in it contributes to
  // total_hits/hits_by_type/players below, scored or not.
  videos_total: number;
  // Only videos where Scoring was configured with this team as team_x/
  // team_y contribute to the win/loss records and radar below - see
  // TeamStatsPage's own caveat text for why.
  videos_with_scoring: number;
  match_wins: number;
  match_losses: number;
  game_wins: number;
  game_losses: number;
  radar: RadarPoint[];
  total_hits: number;
  hits_by_type: Record<string, number>;
  players: TeamPlayerSummary[];
  videos: TeamVideoSummary[];
}

export interface PlayerRadarOut {
  name: string;
  videos_with_data: number;
  radar: RadarPoint[];
}

export interface AuthStatus {
  enabled: boolean;
  password_set: boolean;
  // ISO timestamp of when login will auto-disable itself - null while
  // login is off, or for a pre-existing config saved before this existed.
  expires_at: string | null;
}

export interface Captcha {
  captcha_id: string;
  image_base64: string;
}

// A fresh strong password to preview - purely a suggestion, nothing is
// saved server-side until it's actually submitted via authSetPassword.
export interface SuggestedPassword {
  password: string;
}

export interface PortStatus {
  port: number;
  status: "open" | "error" | "not_forwarded";
}

export interface ShareStatus {
  upnp_enabled: boolean;
  external_ip: string | null;
  local_ip: string;
  ports: PortStatus[];
  share_url: string;
  last_error: string | null;
}
