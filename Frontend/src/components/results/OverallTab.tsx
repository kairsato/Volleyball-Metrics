import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import { api } from "../../lib/api";
import type { MatchupOut, ResultsOut } from "../../lib/types";
import { LoadingSpinner } from "../LoadingSpinner";
import { RadarChart } from "./RadarChart";
import type { FlatEvent } from "./types";
import { WinLossTrend } from "./WinLossTrend";

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

interface OverallTabProps {
  results: ResultsOut;
  flatEvents: FlatEvent[];
  onSeek: (timeS: number) => void;
  onJumpToAction: (playerId: string, actionType: string) => void;
}

export function OverallTab({ results, flatEvents, onSeek, onJumpToAction }: OverallTabProps) {
  const [matchup, setMatchup] = useState<MatchupOut | null>(null);
  const [matchupLoading, setMatchupLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api
      .getMatchup(results.job_id)
      .then((res) => !cancelled && setMatchup(res))
      .catch(() => undefined)
      .finally(() => !cancelled && setMatchupLoading(false));
    return () => {
      cancelled = true;
    };
  }, [results.job_id]);

  const playerEntries = Object.entries(results.players);

  const mvp = playerEntries.slice().sort((a, b) => b[1].total_hits - a[1].total_hits)[0];

  const totalHits = playerEntries.reduce((sum, [, p]) => sum + p.total_hits, 0);
  const avgRallyDuration =
    results.rallies.length > 0
      ? results.rallies.reduce((sum, r) => sum + r.duration_s, 0) / results.rallies.length
      : 0;

  const actionMix: Record<string, number> = {};
  for (const [, p] of playerEntries) {
    for (const [type, count] of Object.entries(p.hits_by_type)) {
      actionMix[type] = (actionMix[type] ?? 0) + count;
    }
  }
  const actionMixTotal = Object.values(actionMix).reduce((a, b) => a + b, 0);

  const leaderboard = playerEntries.slice().sort((a, b) => b[1].total_hits - a[1].total_hits).slice(0, 6);

  // For each action category, whoever recorded the most of it - "most sets",
  // "most spikes", etc, rather than just overall hit count.
  const categoryLeaders = Object.keys(actionMix)
    .map((type) => {
      let leader: { playerId: string; count: number } | null = null;
      for (const [id, stat] of playerEntries) {
        const count = stat.hits_by_type[type] ?? 0;
        if (count > 0 && (!leader || count > leader.count)) leader = { playerId: id, count };
      }
      return leader ? { type, ...leader } : null;
    })
    .filter((entry): entry is { type: string; playerId: string; count: number } => entry !== null);

  return (
    <Box>
      {mvp && mvp[1].total_hits > 0 && (
        <Card
          variant="outlined"
          sx={{ p: 2.5, mb: 3, display: "flex", alignItems: "center", gap: 2, bgcolor: "action.hover" }}
        >
          <EmojiEventsIcon sx={{ fontSize: 40, color: "warning.main" }} />
          <Box>
            <Typography variant="overline" color="text.secondary">
              Most involved
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>
              {mvp[1].name ?? `Player ${mvp[0]}`}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {mvp[1].total_hits} hits across {mvp[1].rallies_participated} rallies
            </Typography>
          </Box>
        </Card>
      )}

      {matchupLoading && (
        <Card variant="outlined" sx={{ mb: 3 }}>
          <LoadingSpinner minHeight={140} />
        </Card>
      )}

      {!matchupLoading && matchup?.available && (
        <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
            Match outcome
          </Typography>
          <WinLossTrend matchup={matchup} />

          {matchup.radar.length > 0 && <RadarChart radar={matchup.radar} />}
        </Card>
      )}

      {!matchupLoading && matchup && !matchup.available && (
        <Card variant="outlined" sx={{ p: 2, mb: 3, bgcolor: "action.hover" }}>
          <Typography variant="body2" color="text.secondary">
            Match outcome unavailable: {matchup.reason}
          </Typography>
        </Card>
      )}

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <StatTile value={results.rallies.length} label="Rallies" />
        <StatTile value={totalHits} label="Total hits" />
        <StatTile value={playerEntries.length} label="Players" />
        <StatTile value={avgRallyDuration ? `${avgRallyDuration.toFixed(1)}s` : "-"} label="Avg rally length" />
      </Grid>

      {actionMixTotal > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Action mix
          </Typography>
          <Stack direction="row" sx={{ height: 10, borderRadius: 999, overflow: "hidden" }}>
            {Object.entries(actionMix).map(([type, count]) => (
              <Box
                key={type}
                sx={{ width: `${(count / actionMixTotal) * 100}%`, bgcolor: ACTION_COLORS[type] ?? "grey.500" }}
              />
            ))}
          </Stack>
          <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }}>
            {Object.entries(actionMix).map(([type, count]) => (
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

      {categoryLeaders.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Most by category
          </Typography>
          <Grid container spacing={1.5}>
            {categoryLeaders.map(({ type, playerId, count }) => {
              const name = results.players[playerId]?.name ?? `Player ${playerId}`;
              return (
                <Grid key={type} size={{ xs: 6, sm: 4 }}>
                  <Card variant="outlined" sx={{ p: 1.5 }}>
                    <Stack direction="row" spacing={0.75} sx={{ alignItems: "center", mb: 0.5 }}>
                      <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: ACTION_COLORS[type] ?? "grey.500" }} />
                      <Typography variant="caption" color="text.secondary" sx={{ textTransform: "capitalize" }}>
                        Most {type}
                      </Typography>
                    </Stack>
                    <Typography sx={{ fontWeight: 600 }}>{name}</Typography>
                    <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
                      <Typography variant="caption" color="text.secondary">
                        {count}
                      </Typography>
                      <Button size="small" onClick={() => onJumpToAction(playerId, type)}>
                        Jump in
                      </Button>
                    </Stack>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Box>
      )}

      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Leaderboard
      </Typography>
      <Stack spacing={1}>
        {leaderboard.map(([id, stat]) => {
          const firstEvent = flatEvents.find((e) => e.playerId === id);
          return (
            <Card
              key={id}
              variant="outlined"
              sx={{ p: 1.5, display: "flex", alignItems: "center", justifyContent: "space-between" }}
            >
              <Box>
                <Typography sx={{ fontWeight: 600 }}>{stat.name ?? `Player ${id}`}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {stat.total_hits} hits · {stat.rallies_participated} rallies
                </Typography>
              </Box>
              {firstEvent && (
                <Button size="small" onClick={() => onSeek(firstEvent.timestamp_s)}>
                  Jump in
                </Button>
              )}
            </Card>
          );
        })}
        {leaderboard.length === 0 && <Typography color="text.secondary">No hit events recorded.</Typography>}
      </Stack>
    </Box>
  );
}
