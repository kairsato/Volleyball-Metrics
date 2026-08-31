import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { Link as RouterLink, Navigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { TeamStatsOut } from "../lib/types";
import { LoadingSpinner } from "../components/LoadingSpinner";
import { RadarChart } from "../components/results/RadarChart";

const ACTION_COLORS: Record<string, string> = {
  serve: "#3b82f6",
  spike: "#f97316",
  set: "#22c55e",
  dig: "#eab308",
  block: "#ec4899",
  hit: "#10b981",
};

function StatTile({ value, label }: { value: string | number; label: string }) {
  return (
    <Grid size={{ xs: 6, sm: 3 }}>
      <Card variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          {value}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
      </Card>
    </Grid>
  );
}

// A named team's stats rolled up across every video it's appeared in -
// reached by clicking a team card on the Teams page. Modeled directly on
// PlayerStatsPage.tsx (stat tiles, action-mix bar, a "by video" list) with
// two additions: a match/game win-loss record and a radar chart, both only
// ever populated from videos where Scoring was actually configured with
// this team - team_roster.json's own docstring is explicit that a named
// team otherwise has no link to any per-video geometric side at all (see
// the caveat banner below).
export function TeamStatsPage() {
  const [searchParams] = useSearchParams();
  const teamId = searchParams.get("id");
  const [stats, setStats] = useState<TeamStatsOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    document.title = stats ? stats.team_name : "Volleyball Metrics";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [stats]);

  useEffect(() => {
    if (!teamId) return;
    let cancelled = false;
    setStats(null);
    setError(null);
    api
      .getTeamStats(teamId)
      .then((res) => !cancelled && setStats(res))
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, [teamId]);

  if (!teamId) return <Navigate to="/teams" replace />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!stats) return <LoadingSpinner minHeight={200} />;

  const hitsByTypeTotal = Object.values(stats.hits_by_type).reduce((a, b) => a + b, 0);
  const hasRecord = stats.videos_with_scoring > 0;

  return (
    <Box sx={{ maxWidth: 900 }}>
      <Button component={RouterLink} to="/teams" startIcon={<ArrowBackIcon />} sx={{ mb: 2 }}>
        Teams
      </Button>

      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        {stats.team_name}
      </Typography>

      {hasRecord ? (
        <>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <StatTile value={`${stats.match_wins}-${stats.match_losses}`} label="Match record" />
            <StatTile value={`${stats.game_wins}-${stats.game_losses}`} label="Game record" />
            <StatTile value={stats.videos_with_scoring} label="Scored videos" />
            <StatTile value={stats.videos_total} label="Total videos" />
          </Grid>

          {stats.radar.some((p) => p.sample_size_a > 0 || p.sample_size_b > 0) && (
            <Box sx={{ mb: 4 }}>
              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                Win % by action
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                Solid = {stats.team_name}, dashed = opponents (combined across every scored video).
              </Typography>
              <RadarChart radar={stats.radar} />
            </Box>
          )}
        </>
      ) : (
        <Alert severity="info" sx={{ mb: 3 }}>
          No scored videos yet for {stats.team_name} - configure Scoring (Setup tab → Scoring Determination) with
          this team on at least one video to see a win/loss record and win-rate radar here.
        </Alert>
      )}

      {hitsByTypeTotal > 0 && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Action mix
          </Typography>
          <Stack direction="row" sx={{ height: 10, borderRadius: 999, overflow: "hidden" }}>
            {Object.entries(stats.hits_by_type).map(([type, count]) => (
              <Box
                key={type}
                sx={{ width: `${(count / hitsByTypeTotal) * 100}%`, bgcolor: ACTION_COLORS[type] ?? "grey.500" }}
              />
            ))}
          </Stack>
          <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }}>
            {Object.entries(stats.hits_by_type).map(([type, count]) => (
              <Stack key={type} direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
                <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: ACTION_COLORS[type] ?? "grey.500" }} />
                <Typography variant="caption" color="text.secondary">
                  {type} ({count})
                </Typography>
              </Stack>
            ))}
          </Stack>
        </Box>
      )}

      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Players
      </Typography>
      {stats.players.length === 0 ? (
        <Typography color="text.secondary" sx={{ mb: 4 }}>
          No roster players on this team yet.
        </Typography>
      ) : (
        <Stack spacing={1} sx={{ mb: 4 }}>
          {stats.players.map((p) => (
            <Card
              key={p.name}
              variant="outlined"
              sx={{ p: 1.5, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}
            >
              <Typography sx={{ fontWeight: 600 }}>{p.name}</Typography>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <Typography variant="caption" color="text.secondary">
                  {p.total_hits} hits
                </Typography>
                <Button size="small" component={RouterLink} to={`/player?name=${encodeURIComponent(p.name)}`}>
                  View
                </Button>
              </Stack>
            </Card>
          ))}
        </Stack>
      )}

      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Videos
      </Typography>
      {stats.videos.length === 0 ? (
        <Typography color="text.secondary">Not featured in any finished video yet.</Typography>
      ) : (
        <Stack spacing={1}>
          {stats.videos.map((v) => (
            <Card
              key={v.job_id}
              variant="outlined"
              sx={{ p: 1.5, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}
            >
              <Box sx={{ minWidth: 0 }}>
                <Typography noWrap sx={{ fontWeight: 600 }}>
                  {v.original_filename}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {v.game_wins}-{v.game_losses} games
                </Typography>
              </Box>
              <Button size="small" component={RouterLink} to={`/video?job=${v.job_id}`} sx={{ flexShrink: 0 }}>
                Open
              </Button>
            </Card>
          ))}
        </Stack>
      )}
    </Box>
  );
}
