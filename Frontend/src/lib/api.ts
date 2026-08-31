import type {
  ActionQualityOut,
  AuthStatus,
  CalibrationPointsOut,
  Captcha,
  Job,
  MatchupOut,
  NamesUpdateOut,
  OcrRegion,
  Point,
  PlayerRadarOut,
  PlayersListOut,
  QualitiesOut,
  ResultsOut,
  RosterOut,
  ScoreConfig,
  ScoreMethod,
  ScoreOut,
  ScoreResult,
  ShareStatus,
  SuggestedPassword,
  TeamRosterOut,
  TeamStatsOut,
} from "./types";

// Derived from wherever this page was itself loaded from, rather than a
// hardcoded "http://127.0.0.1:8000" - a device on the LAN loading the
// frontend via the host machine's own IP (e.g. http://192.168.1.27:5173,
// see vite.config.ts's server.host) would otherwise have its browser call
// 127.0.0.1:8000, which resolves to THAT device's own loopback, not the
// host serving the app. VITE_API_BASE still overrides this when the API
// genuinely lives somewhere else (a separate host/port from the frontend).
const API_BASE = import.meta.env.VITE_API_BASE ?? `http://${window.location.hostname}:8000`;

// Only ever meaningful once login has actually been turned on (see
// auth.py's module docstring) - reading/writing this is always safe even
// then, since AuthMiddleware just ignores a missing/stale token when
// login is off.
const AUTH_TOKEN_KEY = "vva_auth_token";

function getAuthToken(): string | null {
  try {
    return localStorage.getItem(AUTH_TOKEN_KEY);
  } catch {
    return null;
  }
}

function setAuthToken(token: string | null) {
  try {
    if (token) localStorage.setItem(AUTH_TOKEN_KEY, token);
    else localStorage.removeItem(AUTH_TOKEN_KEY);
  } catch {
    // localStorage unavailable (private browsing, etc.) - login just won't
    // persist across a reload, but the current session still works.
  }
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// For the handful of endpoints loaded as a plain <img>/<video> src rather
// than through fetch() - those can't carry a custom Authorization header,
// so the same token travels as a query param instead (AuthMiddleware
// accepts either - see main.py).
function withAuthQuery(url: string): string {
  const token = getAuthToken();
  if (!token) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(token)}`;
}

async function handleErrorResponse(res: Response): Promise<never> {
  if (res.status === 401) {
    // The stored token is gone or expired - drop it and let whoever's
    // listening (LoginGate) fall back to the login screen, rather than
    // every call site having to check for this individually.
    setAuthToken(null);
    window.dispatchEvent(new Event("auth:unauthorized"));
  }

  let detail = res.statusText;
  try {
    const body = await res.json();
    detail = body.detail ?? detail;
  } catch {
    // response wasn't JSON - fall back to the status text
  }
  throw new Error(detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers ?? {}) },
  });

  if (!res.ok) return handleErrorResponse(res);

  return res.json() as Promise<T>;
}

export const api = {
  uploadVideo(file: File): Promise<Job> {
    const form = new FormData();
    form.append("file", file);
    return request<Job>("/api/jobs", { method: "POST", body: form });
  },

  listJobs(): Promise<Job[]> {
    return request<Job[]>("/api/jobs");
  },

  getJob(jobId: string): Promise<Job> {
    return request<Job>(`/api/jobs/${jobId}`);
  },

  processJob(jobId: string): Promise<Job> {
    return request<Job>(`/api/jobs/${jobId}/process`, { method: "POST" });
  },

  finalizeJob(jobId: string): Promise<Job> {
    return request<Job>(`/api/jobs/${jobId}/finalize`, { method: "POST" });
  },

  redoJob(jobId: string): Promise<Job> {
    return request<Job>(`/api/jobs/${jobId}/redo`, { method: "POST" });
  },

  // Cheap path for a calibration change on an already-complete job - see
  // jobs_router.recalibrate_job. Re-picks the ball from its saved raw
  // candidates and re-derives player court coordinates/auto-ignores instead
  // of re-running tracking/detection from scratch (falls back to a full
  // reprocess server-side for a job old enough not to have that raw data).
  recalibrateJob(jobId: string): Promise<Job> {
    return request<Job>(`/api/jobs/${jobId}/recalibrate`, { method: "POST" });
  },

  cancelJob(jobId: string): Promise<Job> {
    return request<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  },

  async deleteJob(jobId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: "DELETE", headers: authHeaders() });
    if (!res.ok) await handleErrorResponse(res);
  },

  async getCalibrationFrame(jobId: string): Promise<string> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/calibration/frame`, { headers: authHeaders() });
    if (!res.ok) await handleErrorResponse(res);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },

  getCalibrationPoints(jobId: string): Promise<CalibrationPointsOut> {
    return request<CalibrationPointsOut>(`/api/jobs/${jobId}/calibration`);
  },

  setCalibrationPoints(
    jobId: string,
    middleLeft: Point,
    middleRight: Point,
    farLeft: Point,
    farRight: Point,
    netTopLeft: Point,
    netTopRight: Point,
    netHeightM: number,
  ): Promise<CalibrationPointsOut> {
    return request<CalibrationPointsOut>(`/api/jobs/${jobId}/calibration`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        middle_left: middleLeft,
        middle_right: middleRight,
        far_left: farLeft,
        far_right: farRight,
        net_top_left: netTopLeft,
        net_top_right: netTopRight,
        net_height_m: netHeightM,
      }),
    });
  },

  setCalibrationConfirmed(jobId: string, confirmed: boolean): Promise<CalibrationPointsOut> {
    return request<CalibrationPointsOut>(`/api/jobs/${jobId}/calibration/confirm`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmed }),
    });
  },

  getPlayers(jobId: string): Promise<PlayersListOut> {
    return request<PlayersListOut>(`/api/jobs/${jobId}/players`);
  },

  updateNames(jobId: string, names: Record<string, string>, ignored: number[] = []): Promise<NamesUpdateOut> {
    return request<NamesUpdateOut>(`/api/jobs/${jobId}/players/names`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names, ignored }),
    });
  },

  setPlayersConfirmed(jobId: string, confirmed: boolean): Promise<PlayersListOut> {
    return request<PlayersListOut>(`/api/jobs/${jobId}/players/confirm`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmed }),
    });
  },

  getResults(jobId: string): Promise<ResultsOut> {
    return request<ResultsOut>(`/api/jobs/${jobId}/results`);
  },

  getMatchup(jobId: string): Promise<MatchupOut> {
    return request<MatchupOut>(`/api/jobs/${jobId}/matchup`);
  },

  getActionQuality(jobId: string): Promise<ActionQualityOut> {
    return request<ActionQualityOut>(`/api/jobs/${jobId}/action-quality`);
  },

  getRoster(): Promise<RosterOut> {
    return request<RosterOut>("/api/roster");
  },

  addToRoster(name: string): Promise<RosterOut> {
    return request<RosterOut>("/api/roster", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  },

  removeFromRoster(name: string): Promise<RosterOut> {
    return request<RosterOut>(`/api/roster/${encodeURIComponent(name)}`, { method: "DELETE" });
  },

  getTeams(): Promise<TeamRosterOut> {
    return request<TeamRosterOut>("/api/teams");
  },

  createTeam(name: string, players: string[]): Promise<TeamRosterOut> {
    return request<TeamRosterOut>("/api/teams", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, players }),
    });
  },

  updateTeam(teamId: string, name: string, players: string[]): Promise<TeamRosterOut> {
    return request<TeamRosterOut>(`/api/teams/${teamId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, players }),
    });
  },

  deleteTeam(teamId: string): Promise<TeamRosterOut> {
    return request<TeamRosterOut>(`/api/teams/${teamId}`, { method: "DELETE" });
  },

  getTeamStats(teamId: string): Promise<TeamStatsOut> {
    return request<TeamStatsOut>(`/api/teams/${teamId}/stats`);
  },

  getPlayerRadar(name: string): Promise<PlayerRadarOut> {
    return request<PlayerRadarOut>(`/api/players/${encodeURIComponent(name)}/radar`);
  },

  async getScoreFrame(jobId: string, timestampS?: number): Promise<string> {
    const query = timestampS !== undefined ? `?t=${timestampS}` : "";
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/score/frame${query}`, { headers: authHeaders() });
    if (!res.ok) await handleErrorResponse(res);
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },

  getScore(jobId: string): Promise<ScoreOut> {
    return request<ScoreOut>(`/api/jobs/${jobId}/score`);
  },

  saveScoreConfig(
    jobId: string,
    method: ScoreMethod,
    teamXId: string | null,
    teamYId: string | null,
    ocrRegion: OcrRegion | null,
    cvReverseDirection: boolean,
  ): Promise<ScoreConfig> {
    return request<ScoreConfig>(`/api/jobs/${jobId}/score/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        method,
        team_x_id: teamXId,
        team_y_id: teamYId,
        ocr_region: ocrRegion,
        cv_reverse_direction: cvReverseDirection,
      }),
    });
  },

  setScoreConfirmed(jobId: string, confirmed: boolean): Promise<ScoreConfig> {
    return request<ScoreConfig>(`/api/jobs/${jobId}/score/confirm`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmed }),
    });
  },

  computeScore(
    jobId: string,
    rangeType: "match" | "games" | "rallies" = "match",
    rangeStart?: number,
    rangeEnd?: number,
  ): Promise<ScoreConfig> {
    return request<ScoreConfig>(`/api/jobs/${jobId}/score/compute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ range_type: rangeType, range_start: rangeStart ?? null, range_end: rangeEnd ?? null }),
    });
  },

  setGameBoundary(jobId: string, rallyIndex: number, split: boolean): Promise<ScoreResult> {
    return request<ScoreResult>(`/api/jobs/${jobId}/score/games/${rallyIndex}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ split }),
    });
  },

  setRallyWinner(jobId: string, rallyIndex: number, winner: "x" | "y" | null): Promise<ScoreResult> {
    return request<ScoreResult>(`/api/jobs/${jobId}/score/rallies/${rallyIndex}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ winner }),
    });
  },

  resetScores(jobId: string): Promise<ScoreResult> {
    return request<ScoreResult>(`/api/jobs/${jobId}/score/reset`, { method: "POST" });
  },

  dashboardUrl(jobId: string): string {
    return withAuthQuery(`${API_BASE}/api/jobs/${jobId}/dashboard`);
  },

  videoUrl(jobId: string): string {
    return withAuthQuery(`${API_BASE}/api/jobs/${jobId}/video`);
  },

  // `quality` is one of QualitiesOut.qualities ("original" or a generated
  // tier like "720p") - omitted or unrecognized falls back to the original
  // file server-side, so this is safe to call exactly as before too.
  sourceVideoUrl(jobId: string, quality?: string): string {
    const query = quality && quality !== "original" ? `?quality=${encodeURIComponent(quality)}` : "";
    return withAuthQuery(`${API_BASE}/api/jobs/${jobId}/source${query}`);
  },

  getQualities(jobId: string): Promise<QualitiesOut> {
    return request<QualitiesOut>(`/api/jobs/${jobId}/qualities`);
  },

  thumbnailUrl(jobId: string): string {
    return withAuthQuery(`${API_BASE}/api/jobs/${jobId}/thumbnail`);
  },

  // --- Auth ---

  authStatus(): Promise<AuthStatus> {
    return request<AuthStatus>("/api/auth/status");
  },

  authGetCaptcha(): Promise<Captcha> {
    return request<Captcha>("/api/auth/captcha");
  },

  async authLogin(password: string, captchaId: string, captchaAnswer: string): Promise<void> {
    const { token } = await request<{ token: string }>("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password, captcha_id: captchaId, captcha_answer: captchaAnswer }),
    });
    setAuthToken(token);
  },

  async authLogout(): Promise<void> {
    await request("/api/auth/logout", { method: "POST" }).catch(() => undefined);
    setAuthToken(null);
    // Distinct from "auth:unauthorized" (fired only on a real 401, which can
    // only happen while login is actually enabled) - this fires on every
    // logout click regardless, so the listener has to re-check status
    // itself rather than assuming a login screen is warranted.
    window.dispatchEvent(new Event("auth:logout"));
  },

  authSetPassword(password: string): Promise<AuthStatus> {
    return request<AuthStatus>("/api/auth/set-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
  },

  authSetEnabled(enabled: boolean): Promise<AuthStatus> {
    return request<AuthStatus>("/api/auth/enable", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
  },

  // Pure preview - generates a fresh strong password without saving
  // anything; authSetPassword is what actually commits it.
  authSuggestPassword(): Promise<SuggestedPassword> {
    return request<SuggestedPassword>("/api/auth/generate-password");
  },

  // --- Share ---

  shareStatus(): Promise<ShareStatus> {
    return request<ShareStatus>("/api/share/status");
  },

  enableUpnp(): Promise<ShareStatus> {
    return request<ShareStatus>("/api/share/upnp/enable", { method: "POST" });
  },

  disableUpnp(): Promise<ShareStatus> {
    return request<ShareStatus>("/api/share/upnp/disable", { method: "POST" });
  },
};
