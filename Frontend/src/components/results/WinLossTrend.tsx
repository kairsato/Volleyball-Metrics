import Box from "@mui/material/Box";
import type { MatchupOut } from "../../lib/types";

const TEAM_A_COLOR = "#3b82f6";
const TEAM_B_COLOR = "#ec4899";
const WIDTH = 320;
const HEIGHT = 120;
const PADDING = 10;

interface WinLossTrendProps {
  matchup: MatchupOut;
}

// Cumulative score differential (Team A wins minus Team B wins) after each
// rally - a classic momentum line, so a run of wins reads as a climbing or
// falling slope rather than just a row of same-sized blocks.
export function WinLossTrend({ matchup }: WinLossTrendProps) {
  const rallies = matchup.rallies;
  if (rallies.length === 0) return null;

  const series: number[] = [];
  for (const rally of rallies) {
    const previous = series.length > 0 ? series[series.length - 1] : 0;
    const delta = rally.winning_team === "A" ? 1 : rally.winning_team === "B" ? -1 : 0;
    series.push(previous + delta);
  }

  const maxAbs = Math.max(1, ...series.map((v) => Math.abs(v)));
  const stepX = series.length > 1 ? (WIDTH - PADDING * 2) / (series.length - 1) : 0;
  const scaleY = (HEIGHT / 2 - PADDING) / maxAbs;
  const midY = HEIGHT / 2;

  const coords = series.map((v, i) => ({ x: PADDING + i * stepX, y: midY - v * scaleY }));
  const linePoints = coords.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <Box component="svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} sx={{ width: "100%", display: "block", color: "text.secondary" }}>
      <line x1={PADDING} y1={midY} x2={WIDTH - PADDING} y2={midY} stroke="currentColor" strokeOpacity={0.25} strokeDasharray="3 3" />
      <text x={2} y={PADDING + 4} fontSize={10} fill={TEAM_A_COLOR}>
        A
      </text>
      <text x={2} y={HEIGHT - PADDING + 2} fontSize={10} fill={TEAM_B_COLOR}>
        B
      </text>

      <polyline points={linePoints} fill="none" stroke="currentColor" strokeOpacity={0.5} strokeWidth={1.5} />

      {coords.map((p, i) => {
        const color =
          rallies[i].winning_team === "A" ? TEAM_A_COLOR : rallies[i].winning_team === "B" ? TEAM_B_COLOR : "currentColor";
        return <circle key={rallies[i].rally_index} cx={p.x} cy={p.y} r={3} fill={color} />;
      })}
    </Box>
  );
}
