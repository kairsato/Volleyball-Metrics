import type {
  CalibrationPointsOut,
  Job,
  MatchupOut,
  NamesUpdateOut,
  OcrRegion,
  Point,
  PlayersListOut,
  ResultsOut,
  RosterOut,
  ScoreConfig,
  ScoreMethod,
  ScoreOut,
  ScoreResult,
  TeamRosterOut,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON - fall back to the status text
    }
    throw new Error(detail);
  }

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

  cancelJob(jobId: string): Promise<Job> {
    return request<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  },

  async deleteJob(jobId: string): Promise<void> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: "DELETE" });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail ?? detail;
      } catch {
        // response wasn't JSON - fall back to the status text
      }
      throw new Error(detail);
    }
  },

  async getCalibrationFrame(jobId: string): Promise<string> {
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/calibration/frame`);
    if (!res.ok) throw new Error("Failed to load calibration frame");
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },

  getCalibrationPoints(jobId: string): Promise<CalibrationPointsOut> {
    return request<CalibrationPointsOut>(`/api/jobs/${jobId}/calibration`);
  },

  setCalibrationPoints(jobId: string, corners: Point[], netPoints: Point[]): Promise<CalibrationPointsOut> {
    return request<CalibrationPointsOut>(`/api/jobs/${jobId}/calibration`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corners, net_points: netPoints }),
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

  getResults(jobId: string): Promise<ResultsOut> {
    return request<ResultsOut>(`/api/jobs/${jobId}/results`);
  },

  getMatchup(jobId: string): Promise<MatchupOut> {
    return request<MatchupOut>(`/api/jobs/${jobId}/matchup`);
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

  async getScoreFrame(jobId: string, timestampS?: number): Promise<string> {
    const query = timestampS !== undefined ? `?t=${timestampS}` : "";
    const res = await fetch(`${API_BASE}/api/jobs/${jobId}/score/frame${query}`);
    if (!res.ok) throw new Error("Failed to load frame");
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
    return `${API_BASE}/api/jobs/${jobId}/dashboard`;
  },

  videoUrl(jobId: string): string {
    return `${API_BASE}/api/jobs/${jobId}/video`;
  },

  sourceVideoUrl(jobId: string): string {
    return `${API_BASE}/api/jobs/${jobId}/source`;
  },

  thumbnailUrl(jobId: string): string {
    return `${API_BASE}/api/jobs/${jobId}/thumbnail`;
  },
};
