import Box from "@mui/material/Box";
import type { RadarPoint } from "../../lib/types";

const GOOD_COLOR = "#22c55e";
const BAD_COLOR = "#ef4444";
const SIZE = 280;
const CENTER = SIZE / 2;
const RADIUS = SIZE * 0.34;
const SPLIT = 0.5; // radius fraction where the red zone gives way to green

function pointAt(index: number, count: number, fraction: number): { x: number; y: number } {
  const angle = -Math.PI / 2 + index * ((2 * Math.PI) / count);
  return { x: CENTER + RADIUS * fraction * Math.cos(angle), y: CENTER + RADIUS * fraction * Math.sin(angle) };
}

function polygonPath(count: number, fraction: number): string {
  return Array.from({ length: count }, (_, i) => {
    const p = pointAt(i, count, fraction);
    return `${p.x},${p.y}`;
  }).join(" ");
}

interface RadarChartProps {
  radar: RadarPoint[];
}

export function RadarChart({ radar }: RadarChartProps) {
  const count = radar.length;
  if (count === 0) return null;

  function teamPoints(key: "team_a_win_rate" | "team_b_win_rate"): string {
    return radar
      .map((point, i) => `${pointAt(i, count, point[key] ?? 0).x},${pointAt(i, count, point[key] ?? 0).y}`)
      .join(" ");
  }

  return (
    <Box
      component="svg"
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      sx={{ width: "100%", maxWidth: 320, display: "block", mx: "auto", color: "text.primary" }}
    >
      <polygon points={polygonPath(count, 1)} fill={GOOD_COLOR} fillOpacity={0.25} />
      <polygon points={polygonPath(count, SPLIT)} fill={BAD_COLOR} fillOpacity={0.3} />

      {Array.from({ length: count }, (_, i) => {
        const p = pointAt(i, count, 1);
        return <line key={i} x1={CENTER} y1={CENTER} x2={p.x} y2={p.y} stroke="currentColor" strokeOpacity={0.25} strokeWidth={1} />;
      })}

      <polygon points={teamPoints("team_a_win_rate")} fill="none" stroke="currentColor" strokeWidth={2} />
      <polygon
        points={teamPoints("team_b_win_rate")}
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeDasharray="5 4"
      />

      {radar.map((point, i) => {
        const a = pointAt(i, count, point.team_a_win_rate ?? 0);
        const b = pointAt(i, count, point.team_b_win_rate ?? 0);
        const label = pointAt(i, count, 1.2);
        return (
          <g key={point.action_type}>
            {point.sample_size_a > 0 && <circle cx={a.x} cy={a.y} r={4} fill="currentColor" />}
            {point.sample_size_b > 0 && <circle cx={b.x} cy={b.y} r={4} fill="none" stroke="currentColor" strokeWidth={2} />}
            <text x={label.x} y={label.y} fontSize={11} textAnchor="middle" dominantBaseline="middle" fill="currentColor" opacity={0.75}>
              {point.action_type}
            </text>
          </g>
        );
      })}
    </Box>
  );
}
