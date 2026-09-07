import { memo } from "react";
import Box from "@mui/material/Box";
import type { BallTrajectoryPoint, PlayerTrajectoryFrame } from "../lib/types";

// Same real-world court dimensions/net-down-the-middle convention
// CourtMap.tsx already draws with (net crosses the court's length axis,
// not its width) - smaller here since this sits as a small corner overlay
// on top of the video rather than a full-width chart.
const PX_PER_M = 6;
// How far back the fading trail behind the current position reaches -
// long enough to read as "the ball is moving this way", short enough to
// stay a glance rather than a whole rally's path.
const TRAIL_WINDOW_S = 1.5;
const BALL_COLOR = "#38bdf8";
// On-screen width of the whole diagram. Big enough that individual players
// are distinguishable dots rather than a single smudge near the net, which
// is the point of drawing them at all. Everything else (player/ball dot
// radii, stroke widths) is defined in the SVG's own viewBox units, so it
// scales up automatically with this - no need to touch them separately.
const DISPLAY_WIDTH_PX = 260;
const PLAYER_RADIUS = 2.6;

// Same hue-per-stable_id formula PlayerTrackingOverlay uses for its
// on-video boxes, so a player's dot here is the same colour as the box
// drawn over them on the video - that shared colour is the only thing
// identifying who's who at this size, since there's no room for labels.
function colorForId(stableId: number): string {
  const hue = (stableId * 47) % 360;
  return `hsl(${hue}, 75%, 60%)`;
}

interface BallMinimapProps {
  points: BallTrajectoryPoint[];
  courtLengthM: number;
  courtWidthM: number;
  // Whoever was on court at the current playback time, already resolved by
  // the caller (VideoPlayer) via the shared findIndexAtOrBefore - same
  // "parent resolves the index, memo skips the re-render" pattern
  // currentIndex below uses. Undefined for a job with no player tracking.
  playerFrame?: PlayerTrajectoryFrame;
  // Index into `points` of "the point at or before now", already resolved
  // by the caller (VideoPlayer) via the shared findIndexAtOrBefore, falling
  // back to 0 before the very first tracked point (e.g. sitting paused at
  // 0:00) so the minimap is visible as soon as a frame renders rather than
  // only once playback has actually passed it. Taking this (a plain
  // number) as a prop instead of raw currentTimeS, combined with wrapping
  // this component in memo() below, is what lets it skip re-rendering on
  // every timeupdate tick when playback hasn't actually reached a new
  // point yet.
  currentIndex: number;
}

// A top-right court diagram showing where the ball currently is (plus a
// short fading trail of where it's just been) and where every tracked
// player is standing, as the video plays - same corner placement and
// trail+dot idiom as the pipeline's own server-side annotated-video minimap
// (see Backend/Analysis/BallDetection/ballDetection.py's draw_court_minimap),
// just drawn client-side from the decimated /ball-trajectory and
// /player-trajectory endpoints instead of baked into rendered video frames.
//
// Drawn rotated 180 degrees: the court's own coordinate system has its
// origin at the far baseline, so plotting it unrotated puts the half
// nearest the camera at the TOP of the diagram, mirroring what's on screen.
// Rotating makes near stay near, so the diagram reads the same way round as
// the video above it.
function BallMinimapImpl({ points, courtLengthM, courtWidthM, currentIndex, playerFrame }: BallMinimapProps) {
  if (points.length === 0 || currentIndex < 0) return null;

  const current = points[currentIndex];
  // Null for a frame with no court position (see results_router's
  // /ball-trajectory - a point can carry pixel coordinates without court
  // ones). The court and its players are still worth drawing in that case,
  // so this only drops the ball dot/trail rather than the whole diagram.
  const ballXY = current.x !== null && current.y !== null ? { x: current.x, y: current.y } : null;
  const trail: { x: number; y: number }[] = [];
  if (ballXY) {
    for (let i = currentIndex; i >= 0 && current.t - points[i].t <= TRAIL_WINDOW_S; i--) {
      const p = points[i];
      if (p.x === null || p.y === null) break;
      trail.unshift({ x: p.x, y: p.y });
    }
  }

  const courtPlayers = (playerFrame?.players ?? []).filter(
    (p): p is typeof p & { court_x: number; court_y: number } => p.court_x !== null && p.court_y !== null,
  );
  if (!ballXY && courtPlayers.length === 0) return null;

  const viewW = courtLengthM * PX_PER_M;
  const viewH = courtWidthM * PX_PER_M;
  // Room around the court rectangle so a tracked point just outside it (a
  // serve from behind the baseline, tracking noise) doesn't get clipped
  // right at the edge - clamped further out than that is still dropped
  // visually rather than let the dot wander arbitrarily far off the map.
  const margin = PX_PER_M * 3;
  const clampX = (x: number) => Math.min(viewW + margin, Math.max(-margin, x * PX_PER_M));
  const clampY = (y: number) => Math.min(viewH + margin, Math.max(-margin, y * PX_PER_M));

  return (
    <Box
      sx={{
        position: "absolute",
        top: 8,
        right: 8,
        zIndex: 1,
        pointerEvents: "none",
        borderRadius: 1,
        overflow: "hidden",
        boxShadow: 3,
        bgcolor: "rgba(0,0,0,0.55)",
        lineHeight: 0,
      }}
    >
      <Box
        component="svg"
        viewBox={`${-margin} ${-margin} ${viewW + margin * 2} ${viewH + margin * 2}`}
        sx={{ width: DISPLAY_WIDTH_PX, height: "auto", display: "block" }}
      >
        {/* Rotated about the court's own centre, so the margins stay
            symmetric around it and nothing needs re-laying-out. */}
        <g transform={`rotate(180 ${viewW / 2} ${viewH / 2})`}>
          <rect x={0} y={0} width={viewW} height={viewH} fill="none" stroke="#fff" strokeOpacity={0.5} strokeWidth={1} />
          <line x1={viewW / 2} y1={0} x2={viewW / 2} y2={viewH} stroke="#fff" strokeOpacity={0.7} strokeWidth={1.5} />
          {courtPlayers.map((p) => (
            <circle
              key={p.stable_id}
              cx={clampX(p.court_x)}
              cy={clampY(p.court_y)}
              r={PLAYER_RADIUS}
              fill={colorForId(p.stable_id)}
              fillOpacity={0.9}
              stroke="#0b1220"
              strokeWidth={0.6}
            />
          ))}
          {trail.length > 1 && (
            <polyline
              points={trail.map((p) => `${clampX(p.x)},${clampY(p.y)}`).join(" ")}
              fill="none"
              stroke={BALL_COLOR}
              strokeOpacity={0.6}
              strokeWidth={1.5}
            />
          )}
          {ballXY && (
            <circle cx={clampX(ballXY.x)} cy={clampY(ballXY.y)} r={3} fill={BALL_COLOR} stroke="#fff" strokeWidth={0.75} />
          )}
        </g>
      </Box>
    </Box>
  );
}

export const BallMinimap = memo(BallMinimapImpl);
