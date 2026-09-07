import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import type { Rally, RallyWinner } from "../lib/types";

interface ScoreOverlayProps {
  // Gives each rally_index its timing - joined against scoreRallies below
  // by rally_index, since RallyWinner itself carries no timing.
  rallies: Rally[];
  scoreRallies: RallyWinner[];
  teamXName: string;
  teamYName: string;
  currentTimeS: number;
}

// Current SET's point score (not a full match/sets record - see this
// component's own doc comment for why that's the right level of detail
// here) as of currentTimeS: every rally that has actually concluded by now
// (its end_time_s has passed) and had a determined winner, tallied within
// whichever game_index the most recent of those rallies belongs to.
export function computeCurrentScore(rallies: Rally[], scoreRallies: RallyWinner[], currentTimeS: number) {
  const timingByIndex = new Map(rallies.map((r) => [r.rally_index, r]));

  const concluded = scoreRallies
    .filter((sr) => {
      const timing = timingByIndex.get(sr.rally_index);
      return timing !== undefined && timing.end_time_s <= currentTimeS && sr.winner !== null;
    })
    .sort((a, b) => (timingByIndex.get(a.rally_index)?.start_time_s ?? 0) - (timingByIndex.get(b.rally_index)?.start_time_s ?? 0));

  if (concluded.length === 0) return null;

  const gameIndex = concluded[concluded.length - 1].game_index;
  let x = 0;
  let y = 0;
  for (const sr of concluded) {
    if (sr.game_index !== gameIndex) continue;
    if (sr.winner === "x") x++;
    else if (sr.winner === "y") y++;
  }
  return { gameIndex, x, y };
}

// A small top-left scoreboard (Minimap sits top-right, deliberately not
// competing for the same corner) showing the CURRENT set's point score as
// the video plays - not the full match record (sets won, other games'
// final scores), which would need more screen space than a corner overlay
// can spare; this is meant to answer "what's the score right now", the one
// thing worth glancing at mid-rally.
export function ScoreOverlay({ rallies, scoreRallies, teamXName, teamYName, currentTimeS }: ScoreOverlayProps) {
  const score = computeCurrentScore(rallies, scoreRallies, currentTimeS);
  if (!score) return null;

  return (
    <Box
      sx={{
        position: "absolute",
        top: 8,
        left: 8,
        zIndex: 1,
        pointerEvents: "none",
        borderRadius: 1,
        boxShadow: 3,
        bgcolor: "rgba(0,0,0,0.65)",
        px: 1.5,
        py: 0.75,
      }}
    >
      <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.75)", display: "block", lineHeight: 1.2 }}>
        Set {score.gameIndex + 1}
      </Typography>
      <Stack direction="row" spacing={1.25} sx={{ alignItems: "baseline" }}>
        <Typography variant="body2" noWrap sx={{ color: "#fff", fontWeight: 600, maxWidth: 120 }}>
          {teamXName}
        </Typography>
        <Typography variant="h6" sx={{ color: "#fff", fontWeight: 700, lineHeight: 1 }}>
          {score.x} - {score.y}
        </Typography>
        <Typography variant="body2" noWrap sx={{ color: "#fff", fontWeight: 600, maxWidth: 120 }}>
          {teamYName}
        </Typography>
      </Stack>
    </Box>
  );
}
