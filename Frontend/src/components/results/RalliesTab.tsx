import { useEffect, useRef, useState } from "react";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PlaylistPlayIcon from "@mui/icons-material/PlaylistPlay";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import { api } from "../../lib/api";
import type { Game, MatchupOut, Rally, ScoreOut, TeamEntry } from "../../lib/types";
import { formatTimestamp, type FlatEvent } from "./types";

interface RalliesTabProps {
  jobId: string;
  rallies: Rally[];
  flatEvents: FlatEvent[];
  currentTime: number;
  onSeek: (timeS: number) => void;
  onPlayAll: (timestamps: number[]) => void;
}

// What a rally's winner badge should actually show - resolved once per
// rally from whichever source (named-team scoring, or the anonymous
// geometric matchup fallback) has an opinion, so the card itself doesn't
// need to know which source it came from.
interface RallyWinnerBadge {
  teamName: string;
  // Named-team scoring (ScoreOut) carries a confidence; the anonymous
  // matchup fallback (MatchupOut) doesn't have one - always "auto" there,
  // since it's inferred the same way regardless.
  confidence: "auto" | "manual" | "uncertain";
}

function resolveWinner(
  rallyIndex: number,
  score: ScoreOut | null,
  matchup: MatchupOut | null,
  teams: TeamEntry[],
): RallyWinnerBadge | null {
  if (score?.result && score.config.method !== "none") {
    const entry = score.result.rallies.find((r) => r.rally_index === rallyIndex);
    if (entry?.winner) {
      const teamId = entry.winner === "x" ? score.config.team_x_id : score.config.team_y_id;
      const team = teams.find((t) => t.id === teamId);
      if (team) return { teamName: team.name, confidence: entry.confidence };
    }
  }

  if (matchup?.available) {
    const outcome = matchup.rallies.find((r) => r.rally_index === rallyIndex);
    if (outcome?.winning_team) {
      return { teamName: `Team ${outcome.winning_team}`, confidence: "auto" };
    }
  }

  return null;
}

function WinnerChip({ winner }: { winner: RallyWinnerBadge }) {
  return (
    <Chip
      size="small"
      icon={<EmojiEventsIcon />}
      color={winner.confidence === "uncertain" ? "default" : "success"}
      variant={winner.confidence === "uncertain" ? "outlined" : "filled"}
      label={
        winner.confidence === "manual" || winner.confidence === "uncertain"
          ? `${winner.teamName} won (${winner.confidence})`
          : `${winner.teamName} won`
      }
    />
  );
}

interface RallyCardProps {
  rally: Rally;
  events: FlatEvent[];
  isCurrent: boolean;
  winner: RallyWinnerBadge | null;
  onSeek: (timeS: number) => void;
  onPlayAll: (timestamps: number[]) => void;
}

function RallyCard({ rally, events, isCurrent, winner, onSeek, onPlayAll }: RallyCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const players = Array.from(new Set(events.map((e) => e.playerName)));

  useEffect(() => {
    if (isCurrent) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [isCurrent]);

  return (
    <Card
      ref={ref}
      variant="outlined"
      sx={{
        p: 2,
        borderColor: isCurrent ? "primary.main" : "divider",
        borderWidth: isCurrent ? 2 : 1,
        bgcolor: isCurrent ? "action.hover" : "transparent",
        transition: "background-color 0.15s ease, border-color 0.15s ease",
      }}
    >
      <Stack
        direction="row"
        spacing={1}
        sx={{ alignItems: "center", justifyContent: "space-between", mb: 1, flexWrap: "wrap" }}
      >
        <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap" }}>
          <Typography sx={{ fontWeight: 600 }}>
            Rally {rally.rally_index + 1}{" "}
            <Typography component="span" variant="body2" color="text.secondary">
              {formatTimestamp(rally.start_time_s)}-{formatTimestamp(rally.end_time_s)} · {rally.duration_s.toFixed(1)}s
            </Typography>
          </Typography>
          {isCurrent && <Chip size="small" color="primary" label="Playing" />}
          {winner && <WinnerChip winner={winner} />}
        </Stack>
        <Stack direction="row" spacing={1}>
          <Button size="small" startIcon={<PlayArrowIcon />} onClick={() => onSeek(rally.start_time_s)}>
            Play
          </Button>
          {events.length > 0 && (
            <Button
              size="small"
              startIcon={<PlaylistPlayIcon />}
              onClick={() => onPlayAll(events.map((e) => e.timestamp_s))}
            >
              Play all touches
            </Button>
          )}
        </Stack>
      </Stack>

      {players.length > 0 ? (
        <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap" }}>
          {players.map((name) => (
            <Chip key={name} size="small" label={name} variant="outlined" />
          ))}
        </Stack>
      ) : (
        <Typography variant="body2" color="text.secondary">
          No recorded touches.
        </Typography>
      )}
    </Card>
  );
}

// Bundles rallies under the game they fall within (ScoreOut.result.games'
// start/end are rally_index values, not list positions - see score.py) -
// every rally that isn't covered by any defined game (scoring not
// configured, or a rally outside the tracked games) falls into a single
// null-keyed bucket, rendered with no header at all.
function groupByGame(rallies: Rally[], games: Game[]): { game: Game | null; rallies: Rally[] }[] {
  if (games.length === 0) {
    return [{ game: null, rallies }];
  }

  const sortedGames = [...games].sort((a, b) => a.game_index - b.game_index);
  const groups: { game: Game | null; rallies: Rally[] }[] = sortedGames.map((game) => ({ game, rallies: [] }));
  const ungrouped: Rally[] = [];

  for (const rally of rallies) {
    const game = sortedGames.find(
      (g) => rally.rally_index >= g.start_rally_index && rally.rally_index <= g.end_rally_index,
    );
    if (game) {
      groups.find((g) => g.game === game)!.rallies.push(rally);
    } else {
      ungrouped.push(rally);
    }
  }

  if (ungrouped.length > 0) groups.push({ game: null, rallies: ungrouped });
  return groups.filter((g) => g.rallies.length > 0);
}

export function RalliesTab({ jobId, rallies, flatEvents, currentTime, onSeek, onPlayAll }: RalliesTabProps) {
  const [score, setScore] = useState<ScoreOut | null>(null);
  const [matchup, setMatchup] = useState<MatchupOut | null>(null);
  const [teams, setTeams] = useState<TeamEntry[]>([]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.getScore(jobId), api.getMatchup(jobId), api.getTeams()])
      .then(([scoreRes, matchupRes, teamsRes]) => {
        if (cancelled) return;
        setScore(scoreRes);
        setMatchup(matchupRes);
        setTeams(teamsRes.teams);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  if (rallies.length === 0) {
    return <Typography color="text.secondary">No rallies were detected in this video.</Typography>;
  }

  const groups = groupByGame(rallies, score?.result?.games ?? []);
  const showGameHeaders = groups.length > 1 || groups[0]?.game !== null;

  return (
    <Stack spacing={3}>
      {groups.map(({ game, rallies: groupRallies }) => (
        <Stack key={game?.game_index ?? "ungrouped"} spacing={1.5}>
          {showGameHeaders && (
            <Typography variant="subtitle2" color="text.secondary">
              {game ? `Game ${game.game_index + 1}` : "Other rallies"}
            </Typography>
          )}
          {groupRallies.map((rally) => (
            <RallyCard
              key={rally.rally_index}
              rally={rally}
              events={flatEvents.filter((e) => e.rally_index === rally.rally_index)}
              isCurrent={currentTime >= rally.start_time_s && currentTime <= rally.end_time_s}
              winner={resolveWinner(rally.rally_index, score, matchup, teams)}
              onSeek={onSeek}
              onPlayAll={onPlayAll}
            />
          ))}
        </Stack>
      ))}
    </Stack>
  );
}
