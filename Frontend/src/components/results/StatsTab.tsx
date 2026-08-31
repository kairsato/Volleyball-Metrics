import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import { api } from "../../lib/api";
import type { MatchupOut, ScoreOut, TeamEntry } from "../../lib/types";
import { LoadingSpinner } from "../LoadingSpinner";
import { RadarChart } from "./RadarChart";
import { WinLossTrend } from "./WinLossTrend";

interface StatsTabProps {
  jobId: string;
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

export function StatsTab({ jobId }: StatsTabProps) {
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

  return (
    <Box>
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
