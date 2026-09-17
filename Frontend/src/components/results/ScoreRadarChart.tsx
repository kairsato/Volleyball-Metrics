import Box from "@mui/material/Box";

const SIZE = 280;
const CENTER = SIZE / 2;
const RADIUS = SIZE * 0.34;
const RING_FRACTIONS = [0.25, 0.5, 0.75, 1];

// Red (low score) through green (high score) - same directional meaning as
// RadarChart's red/green split zone, just continuous instead of a single
// 0.5 threshold, since a magnitude score has no natural "win/lose" split.
function colorFor(value: number): string {
  const hue = Math.max(0, Math.min(1, value)) * 120;
  return `hsl(${hue}, 70%, 45%)`;
}

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

export interface ScoreRadarPoint {
  key: string;
  label: string;
  value: number | null; // 0-1, null when there's no data for this point
}

interface ScoreRadarChartProps {
  points: ScoreRadarPoint[];
}

// Showcases a set of 0-1 composite scores (one per action category) as a
// single filled shape, unlike RadarChart's two win-rate series - used by
// AnalyticsTab/PlayerStatsPage to give an at-a-glance "shape" of a
// player's (or the whole video's) overall quality profile.
export function ScoreRadarChart({ points }: ScoreRadarChartProps) {
  const count = points.length;
  if (count === 0) return null;

  const available = points.filter((p): p is ScoreRadarPoint & { value: number } => p.value !== null);
  const overall = available.length > 0 ? available.reduce((sum, p) => sum + p.value, 0) / available.length : 0.5;
  const shapeColor = colorFor(overall);

  const seriesPath = points.map((p, i) => pointAt(i, count, p.value ?? 0)).map((pt) => `${pt.x},${pt.y}`).join(" ");

  return (
    <Box
      component="svg"
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      sx={{ width: "100%", maxWidth: 320, display: "block", mx: "auto", color: "text.primary" }}
    >
      {RING_FRACTIONS.map((f) => (
        <polygon key={f} points={polygonPath(count, f)} fill="none" stroke="currentColor" strokeOpacity={0.15} strokeWidth={1} />
      ))}
      {Array.from({ length: count }, (_, i) => {
        const p = pointAt(i, count, 1);
        return <line key={i} x1={CENTER} y1={CENTER} x2={p.x} y2={p.y} stroke="currentColor" strokeOpacity={0.2} strokeWidth={1} />;
      })}

      <polygon points={seriesPath} fill={shapeColor} fillOpacity={0.25} stroke={shapeColor} strokeWidth={2} />

      {points.map((p, i) => {
        const dot = pointAt(i, count, p.value ?? 0);
        const label = pointAt(i, count, 1.2);
        return (
          <g key={p.key}>
            {p.value !== null && <circle cx={dot.x} cy={dot.y} r={4} fill={colorFor(p.value)} />}
            <text x={label.x} y={label.y} fontSize={11} textAnchor="middle" dominantBaseline="middle" fill="currentColor" opacity={0.75}>
              {p.label}
            </text>
          </g>
        );
      })}
    </Box>
  );
}
