import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import { api } from "../../lib/api";
import type { MatchupOut, ResultsOut, ScoreOut, TeamEntry } from "../../lib/types";
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

interface StatsTabProps {
  jobId: string;
  results: ResultsOut;
  flatEvents: FlatEvent[];
  onSeek: (timeS: number) => void;
  onJumpToAction: (playerId: string, actionType: string) => void;
}

function StatTile({ value, label }: { value: string | number; label: string }) {
  return (
    <Grid size={{ xs: 6, sm: 4 }}>
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

interface ScoreSummary {
  teamXName: string;
  teamYName: string;
  gameWinners: ("x" | "y" | null)[];
  gamesX: number;
  gamesY: number;
}

// Per-game winner (majority of that game's rallies) and the resulting
// match record - the same "most rallies wins the game, most games wins the
// match" rollup Backend/API/score.py's compute_summary already applies
// server-side for Job.winner_team_name, recomputed here so this tab can
// show the game-by-game breakdown too, not just the final name.
function summarizeScore(score: ScoreOut, teams: TeamEntry[]): ScoreSummary | null {
  if (!score.result) return null;

  const teamXName = teams.find((t) => t.id === score.config.team_x_id)?.name ?? "Team X";
  const teamYName = teams.find((t) => t.id === score.config.team_y_id)?.name ?? "Team Y";

  const gameWinners: ("x" | "y" | null)[] = score.result.games.map((game) => {
    let x = 0;
    let y = 0;
    for (const rally of score.result!.rallies) {
      if (rally.game_index !== game.game_index) continue;
      if (rally.winner === "x") x++;
      else if (rally.winner === "y") y++;
    }
    if (x === 0 && y === 0) return null;
    return x > y ? "x" : y > x ? "y" : null;
  });

  return {
    teamXName,
    teamYName,
    gameWinners,
    gamesX: gameWinners.filter((w) => w === "x").length,
    gamesY: gameWinners.filter((w) => w === "y").length,
  };
}

export function StatsTab({ jobId, results, flatEvents, onSeek, onJumpToAction }: StatsTabProps) {
  const [matchup, setMatchup] = useState<MatchupOut | null>(null);
  const [score, setScore] = useState<ScoreOut | null>(null);
  const [teams, setTeams] = useState<TeamEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.getMatchup(jobId), api.getScore(jobId), api.getTeams()])
      .then(([matchupRes, scoreRes, teamsRes]) => {
        if (cancelled) return;
        setMatchup(matchupRes);
        setScore(scoreRes);
        setTeams(teamsRes.teams);
      })
      .catch(() => undefined)
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (loading) return <LoadingSpinner minHeight={200} />;

  const scored = score && score.config.method !== "none" ? summarizeScore(score, teams) : null;

  const playerEntries = Object.entries(results.players);
  const mvp = playerEntries.slice().sort((a, b) => b[1].total_hits - a[1].total_hits)[0];
  const totalHits = playerEntries.reduce((sum, [, p]) => sum + p.total_hits, 0);
  const avgRallyDuration =
    results.rallies.length > 0 ? results.rallies.reduce((sum, r) => sum + r.duration_s, 0) / results.rallies.length : 0;

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

      {scored && (
        <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
          <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 1.5 }}>
            <EmojiEventsIcon sx={{ color: "warning.main" }} />
            <Typography variant="subtitle2">Match record</Typography>
          </Stack>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <StatTile
              value={`${scored.gamesX} - ${scored.gamesY}`}
              label={`${scored.teamXName} vs ${scored.teamYName} (games)`}
            />
            <StatTile
              value={
                scored.gamesX === scored.gamesY
                  ? "Tied"
                  : scored.gamesX > scored.gamesY
                    ? scored.teamXName
                    : scored.teamYName
              }
              label="Match leader"
            />
          </Grid>
          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
            {scored.gameWinners.map((winner, i) => (
              <Chip
                key={i}
                size="small"
                label={`Game ${i + 1}: ${winner === "x" ? scored.teamXName : winner === "y" ? scored.teamYName : "unresolved"}`}
                color={winner ? "success" : "default"}
                variant={winner ? "filled" : "outlined"}
              />
            ))}
          </Stack>
        </Card>
      )}

      {!scored && matchup?.available && (
        <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
            Win record (Team A / Team B)
          </Typography>
          <Grid container spacing={2}>
            <StatTile value={matchup.wins_a} label="Team A wins" />
            <StatTile value={matchup.wins_b} label="Team B wins" />
          </Grid>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1.5 }}>
            No named-team scoring configured for this video yet (see Setup → Scoring Determination) - showing the
            anonymous geometric Team A/B split instead.
          </Typography>
        </Card>
      )}

      {!scored && !matchup?.available && (
        <Card variant="outlined" sx={{ p: 2, mb: 3, bgcolor: "action.hover" }}>
          <Typography variant="body2" color="text.secondary">
            Win/loss stats unavailable{matchup?.reason ? `: ${matchup.reason}` : "."}
          </Typography>
        </Card>
      )}

      {matchup?.available && (
        <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
            Momentum
          </Typography>
          <WinLossTrend matchup={matchup} />
        </Card>
      )}

      {matchup?.available && matchup.radar.length > 0 && (
        <Card variant="outlined" sx={{ p: 2.5 }}>
          <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
            Win % by action
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
            How often a team won the rally when one of their players performed each action at least once.
          </Typography>
          <RadarChart radar={matchup.radar} />
        </Card>
      )}
    </Box>
  );
}
