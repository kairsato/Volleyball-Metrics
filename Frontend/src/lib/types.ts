export type JobStatus =
  | "uploaded"
  | "processing"
  | "awaiting_player_review"
  | "finalizing"
  | "complete"
  | "error"
  | "cancelled";

// One uploaded video within a job's `videos` list - see Job.videos below.
export interface VideoSegment {
  id: string;
  order: number;
  original_filename: string;
  duration_s: number | null;
  // User-provided at upload time, defaulting to the video's own recording-
  // date metadata when readable, else the upload date - see
  // Backend/API/video_metadata.py.
  date_played: string | null;
}

export interface Job {
  id: string;
  original_filename: string;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  stage: string | null;
  completed_stages: string[];
  stage_durations_s: Record<string, number>;
  // When the pipeline last actually finished running - deliberately NOT
  // the same thing as updated_at, which bumps on essentially any change to
  // the job (a calibration save, a player rename), not just a completed
  // run. Null for a job that's never completed one, or one processed
  // before this field existed.
  processed_at: string | null;
  error: string | null;
  // The uploaded video's own length in seconds - null until the backend has
  // computed it (see Backend/API/routers/jobs_router.py's _ensure_duration).
  // For a multi-video job this mirrors segment 0's own duration_s - see
  // `videos` below for every segment.
  duration_s: number | null;
  // Segment 0's own date_played, mirrored here for convenience - same
  // reasoning as duration_s above.
  date_played: string | null;
  // Every video segment making up this job, in order - length 1 for an
  // ordinary single-video job.
  videos: VideoSegment[];
  // Only meaningful when videos.length > 1 - see CalibrationPanel.tsx's
  // "segmented court selections" toggle.
  segmented_calibration: boolean;
  // Video-list summary flags - always false/null for a job that isn't
  // "complete" yet, since neither player identification nor scoring exists
  // before then.
  needs_player_id: boolean;
  needs_scoring_review: boolean;
  winner_team_name: string | null;
  // Whether a warmup period is confirmed for this video - when true,
  // warmup_start_s/warmup_end_s are the resolved absolute-video-time
  // bounds the player, thumbnails, and every rally/stat below already
  // restrict themselves to server-side (see Backend/API/warmup.py). Both
  // null when no warmup period is confirmed, meaning the whole video is in
  // play as before this feature existed.
  warmup_confirmed: boolean;
  warmup_start_s: number | null;
  warmup_end_s: number | null;
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
  // Only set when the thumbnail frame has another player's box crowding
  // into the crop - a copy of thumbnail_base64 with a white highlight box
  // around the actual subject. Only ever used on the player-identification
  // page's still-unnamed tiles; everywhere else shows thumbnail_base64.
  identification_thumbnail_base64: string | null;
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
  // Confidence (0-1) of the trained action classifier's call on this hit -
  // null when action_type instead came from the "serve" timing rule or the
  // geometric-heuristic fallback (see ActionDetection.actionDetection's
  // NOTE ON ACCURACY / consolidate.py's CAVEATS).
  action_type_confidence: number | null;
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

export interface WarmupConfig {
  job_id: string;
  start_s: number;
  // null means "the end of the video" - duration_s below always resolves
  // it to a concrete value for anything that needs one (e.g. a range
  // slider's max).
  end_s: number | null;
  confirmed: boolean;
  duration_s: number | null;
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

export interface BallTrajectoryPoint {
  // Seconds, relative to the warmup period's start (same basis as every
  // other timestamp the player deals in) - see warmup.py.
  t: number;
  // Real-world court coordinates in metres, same system as
  // ActionQualityCategory instances' ball_court - can fall outside
  // [0, court_length_m] x [0, court_width_m] (a serve from behind the
  // baseline, tracking noise). Used for the corner Minimap annotation.
  // Null when this job has no court calibration yet - the point still
  // carries px/py in that case, just nothing Minimap can plot.
  x: number | null;
  y: number | null;
  // Raw source-video pixel coordinates of the same detection, when
  // available - used for the on-video Ball tracking annotation (see
  // BallTrackingOverlay.tsx). Null for a point with no pixel reading.
  px: number | null;
  py: number | null;
  // Estimated height (metres) above the court's ground plane - only
  // populated when this job's calibration includes a solved camera pose.
  // Used by BallTrajectoryOverlay to clip its arc once the ball descends
  // back below net height.
  height_m: number | null;
}

export interface BallTrajectory {
  job_id: string;
  points: BallTrajectoryPoint[];
  court_length_m: number;
  court_width_m: number;
  net_height_m: number;
  // The ORIGINAL uploaded video's own pixel resolution - what every px/py
  // above is measured in. NOT the currently-playing quality rendition's
  // resolution (see api.ts's sourceVideoUrl/VideoPlayer's quality prop):
  // those can differ once a viewer picks a lower-bitrate quality, and an
  // overlay sized from the <video> element's own decoded videoWidth/
  // videoHeight instead of this would misplace every annotation the moment
  // that happens. Null while ball_detection hasn't completed yet.
  frame_w: number | null;
  frame_h: number | null;
}

export interface PlayerBox {
  stable_id: number;
  // null for a still-unnamed detection - falls back to "#<stable_id>".
  name: string | null;
  // [x1, y1, x2, y2] in the source video's own pixel space.
  box: [number, number, number, number];
  // Real-world court metres under this player's feet, same coordinate
  // system as BallTrajectoryPoint.x/y - used by BallMinimap to place them
  // on the court diagram. Null for a job with no court calibration.
  court_x: number | null;
  court_y: number | null;
}

export interface PlayerTrajectoryFrame {
  // Same relative-to-warmup-start basis as BallTrajectoryPoint.t.
  t: number;
  players: PlayerBox[];
}

export interface PlayerTrajectory {
  job_id: string;
  frames: PlayerTrajectoryFrame[];
  // Same meaning as BallTrajectory.frame_w/frame_h - see there.
  frame_w: number | null;
  frame_h: number | null;
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

export interface OcrDetection {
  text: string;
  // 0-1, easyocr's own confidence for this one detection - not tied to
  // whether it ended up being picked as the left/right digit at all (see
  // ScoreRegionTestResult), just how sure the model was it read `text`
  // correctly.
  confidence: number;
}

// One-off "Test frame" read of a single exact frame/region - see
// Backend/API/score_cv.py's test_region and the Setup tab's Scoreboard
// Identification Region dialog, which calls this before a region is even
// saved so a human can sanity-check it against a real frame first.
export interface ScoreRegionTestResult {
  left: number | null;
  left_confidence: number | null;
  right: number | null;
  right_confidence: number | null;
  // Every raw detection in the crop, not just the two picked out as
  // left/right - shows what the model actually saw (a misread digit, a
  // stray label, nothing at all) even when left/right come back null.
  detections: OcrDetection[];
}

export interface ScoreConfig {
  method: ScoreMethod;
  team_x_id: string | null;
  team_y_id: string | null;
  ocr_region: OcrRegion | null;
  // Only meaningful for method "ocr" - flips which geometric side the
  // left-read digit is treated as belonging to.
  cv_reverse_direction: boolean;
  // Minimum OCR confidence (0-1) a digit-run detection needs before it
  // counts at all - a detection below this is treated the same as that
  // side not having been read, rather than trusting a low-confidence
  // guess. Also what the Scoreboard Identification Region dialog's own
  // "Test frame" button uses, so a test read previews real behavior.
  ocr_min_confidence: number;
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
  // Whether login is turned on globally - drives the Share dialog's own UI
  // (the switch, password form, etc.), NOT whether this specific browser
  // needs to authenticate - see login_required below for that.
  enabled: boolean;
  password_set: boolean;
  // ISO timestamp of when login will auto-disable itself - null while
  // login is off, or for a pre-existing config saved before this existed.
  expires_at: string | null;
  // Whether THIS browser specifically needs to log in - false for a LAN
  // visitor even while `enabled` is true for remote ones. LoginGate reads
  // this, not `enabled`, to decide whether to show the login screen.
  login_required: boolean;
  // Whether THIS browser is on the LAN - independent of whether login is
  // even on (unlike login_required above). SettingsMenu reads this to hide
  // the Share menu item entirely for a remote visitor, since every
  // /api/share/* call it could make is already blocked server-side anyway.
  is_lan: boolean;
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
  hostname: string | null;
  ports: PortStatus[];
  share_url: string;
  last_error: string | null;
}
