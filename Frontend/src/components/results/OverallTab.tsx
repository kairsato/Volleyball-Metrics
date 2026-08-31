import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import { api } from "../../lib/api";
import type { ActionQualityCategory, ActionQualityOut, ResultsOut } from "../../lib/types";
import { LoadingSpinner } from "../LoadingSpinner";
import type { FlatEvent } from "./types";

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

function formatPercent(value: number | null): string {
  return value !== null ? `${Math.round(value * 100)}%` : "-";
}

function factorLabel(key: string): string {
  return key.replace(/_/g, " ");
}

function averageFactor(category: ActionQualityCategory, key: string): number | null {
  const values = category.instances
    .map((instance) => instance.factors[key])
    .filter((value): value is number => value !== null && value !== undefined);
  return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function ScoreBar({ value }: { value: number | null }) {
  return (
    <Stack direction="row" spacing={1} sx={{ alignItems: "center", flex: 1 }}>
      <Box sx={{ flex: 1, height: 6, borderRadius: 999, bgcolor: "action.hover", overflow: "hidden" }}>
        <Box
          sx={{ width: `${value !== null ? value * 100 : 0}%`, height: "100%", bgcolor: "primary.main" }}
        />
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ width: 34, textAlign: "right", flexShrink: 0 }}>
        {formatPercent(value)}
      </Typography>
    </Stack>
  );
}

interface QualitySectionProps {
  title: string;
  category: ActionQualityCategory;
  playerNames: Record<string, string | undefined>;
  onSeek: (timeS: number) => void;
}

// One card per action type (serve/receive/set/spike) - overall weighted
// score up top, a breakdown of the named sub-factors it's built from (see
// Backend/API/action_quality.py for exactly what each factor measures),
// then whoever recorded it ranked by their own average score.
function QualitySection({ title, category, playerNames, onSeek }: QualitySectionProps) {
  if (category.count === 0) {
    return (
      <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          No {title.toLowerCase()} actions detected.
        </Typography>
      </Card>
    );
  }

  const players = category.players.slice().sort((a, b) => (b.average_score ?? -1) - (a.average_score ?? -1));

  return (
    <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
      <Stack direction="row" sx={{ alignItems: "baseline", justifyContent: "space-between", mb: 1.5 }}>
        <Typography variant="subtitle2">{title}</Typography>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          {formatPercent(category.average_score)}
        </Typography>
      </Stack>

      <Stack spacing={0.75} sx={{ mb: 2 }}>
        {Object.entries(category.weights).map(([key, weight]) => (
          <Stack key={key} direction="row" spacing={1} sx={{ alignItems: "center" }}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ width: 150, flexShrink: 0, textTransform: "capitalize" }}
            >
              {factorLabel(key)} ({Math.round(weight * 100)}%)
            </Typography>
            <ScoreBar value={averageFactor(category, key)} />
          </Stack>
        ))}
      </Stack>

      <Stack spacing={0.75}>
        {players.slice(0, 6).map((player) => {
          const firstInstance = category.instances.find((i) => i.player_stable_id === player.stable_id);
          return (
            <Stack
              key={player.stable_id}
              direction="row"
              sx={{ alignItems: "center", justifyContent: "space-between" }}
            >
              <Typography variant="body2">
                {playerNames[String(player.stable_id)] ?? `Player ${player.stable_id}`}
              </Typography>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <Typography variant="caption" color="text.secondary">
                  {formatPercent(player.average_score)} · {player.count}
                </Typography>
                {firstInstance && (
                  <Button size="small" onClick={() => onSeek(firstInstance.timestamp_s)}>
                    Jump in
                  </Button>
                )}
              </Stack>
            </Stack>
          );
        })}
      </Stack>
    </Card>
  );
}

interface OverallTabProps {
  results: ResultsOut;
  flatEvents: FlatEvent[];
  onSeek: (timeS: number) => void;
  onJumpToAction: (playerId: string, actionType: string) => void;
}

export function OverallTab({ results, flatEvents, onSeek, onJumpToAction }: OverallTabProps) {
  const [quality, setQuality] = useState<ActionQualityOut | null>(null);
  const [qualityLoading, setQualityLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api
      .getActionQuality(results.job_id)
      .then((res) => !cancelled && setQuality(res))
      .catch(() => undefined)
      .finally(() => !cancelled && setQualityLoading(false));
    return () => {
      cancelled = true;
    };
  }, [results.job_id]);

  const playerEntries = Object.entries(results.players);
  const playerNames = Object.fromEntries(playerEntries.map(([id, stat]) => [id, stat.name ?? undefined]));

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
      <Stack spacing={1} sx={{ mb: 3 }}>
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

      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Action quality
      </Typography>

      {qualityLoading && (
        <Card variant="outlined" sx={{ mb: 3 }}>
          <LoadingSpinner minHeight={140} />
        </Card>
      )}

      {!qualityLoading && quality && (
        <>
          {(!quality.teams_available || !quality.height_available) && (
            <Alert severity="info" sx={{ mb: 2 }}>
              {!quality.teams_available &&
                "Team splitting isn't available for this video, so placement/blocker factors below are skipped. "}
              {!quality.height_available &&
                "Trajectory-height factors aren't available - mark the net-top points in Court Calibration to enable them."}
            </Alert>
          )}
          <QualitySection title="Serve" category={quality.serve} playerNames={playerNames} onSeek={onSeek} />
          <QualitySection title="Receive" category={quality.receive} playerNames={playerNames} onSeek={onSeek} />
          <QualitySection title="Set" category={quality.set} playerNames={playerNames} onSeek={onSeek} />
          <QualitySection title="Spike" category={quality.spike} playerNames={playerNames} onSeek={onSeek} />
        </>
      )}
    </Box>
  );
}
