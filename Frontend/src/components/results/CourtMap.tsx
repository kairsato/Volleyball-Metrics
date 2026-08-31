import Box from "@mui/material/Box";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import type { ActionQualityCategory } from "../../lib/types";

// Same real-world court dimensions Backend/Analysis/PostProcessing/
// renderVideo.py's own minimap already uses (COURT_LENGTH_M/COURT_WIDTH_M)
// and the same net position action_quality.py's NET_X_M is defined
// against - an orthographic top-down map, not the perspective illustration
// CalibrationPanel's CourtDiagram draws (that one's for showing where to
// click on a real camera frame, not for plotting real (x, y) data).
const COURT_LENGTH_M = 18;
const COURT_WIDTH_M = 9;
const PX_PER_M = 20;
const VIEW_W = COURT_LENGTH_M * PX_PER_M;
const VIEW_H = COURT_WIDTH_M * PX_PER_M;

function scoreColor(score: number | null): string {
  if (score === null) return "#9e9e9e";
  if (score >= 0.66) return "#22c55e";
  if (score >= 0.33) return "#f59e0b";
  return "#ef4444";
}

interface CourtMapProps {
  category: ActionQualityCategory;
  playerNames: Record<string, string | undefined>;
  onSeek: (timeS: number) => void;
}

// Where each Set instance was actually hit from - the setter's contact
// point, colored by that instance's overall score, with a tooltip giving
// the raw numbers (height/time given) against this category's reference
// "ideal" values. Clicking a dot seeks the video to that moment, same
// "jump in" idea used everywhere else in this tab.
export function CourtMap({ category, playerNames, onSeek }: CourtMapProps) {
  const points = category.instances
    .map((instance, index) => ({ instance, index }))
    .filter(({ instance }) => instance.ball_court !== null);

  if (points.length === 0) return null;

  const heightIdeal = category.reference.height_ideal_m;
  const timeReference = category.reference.time_reference_s;

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
        Where sets were hit from (net down the middle) - colored by overall score, click a dot to jump to it.
      </Typography>
      <Box
        component="svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        sx={{
          width: "100%",
          maxWidth: 420,
          display: "block",
          border: 1,
          borderColor: "divider",
          borderRadius: 1,
          bgcolor: "action.hover",
        }}
      >
        <rect x={0} y={0} width={VIEW_W} height={VIEW_H} fill="none" stroke="currentColor" strokeOpacity={0.25} />
        <line
          x1={VIEW_W / 2}
          y1={0}
          x2={VIEW_W / 2}
          y2={VIEW_H}
          stroke="currentColor"
          strokeOpacity={0.45}
          strokeWidth={2}
        />
        {points.map(({ instance, index }) => {
          const [x, y] = instance.ball_court!;
          const name = playerNames[String(instance.player_stable_id)] ?? `Player ${instance.player_stable_id ?? "?"}`;
          const details = [
            instance.ball_height_m !== null
              ? `Height ${instance.ball_height_m.toFixed(1)}m${heightIdeal ? ` (ideal ~${heightIdeal}m)` : ""}`
              : null,
            instance.time_since_prev_touch_s !== null
              ? `Time given ${instance.time_since_prev_touch_s.toFixed(1)}s${timeReference ? ` (ideal ~${timeReference}s)` : ""}`
              : null,
          ]
            .filter((line): line is string => line !== null)
            .join(" · ");

          return (
            <Tooltip key={index} title={details ? `${name} - ${details}` : name} arrow>
              <circle
                cx={x * PX_PER_M}
                cy={y * PX_PER_M}
                r={5}
                fill={scoreColor(instance.overall_score)}
                stroke="#fff"
                strokeWidth={1}
                style={{ cursor: "pointer" }}
                onClick={() => onSeek(instance.timestamp_s)}
              />
            </Tooltip>
          );
        })}
      </Box>
    </Box>
  );
}
