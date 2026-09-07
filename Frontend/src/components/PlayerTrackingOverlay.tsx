import { memo } from "react";
import Box from "@mui/material/Box";
import type { PlayerBox, PlayerTrajectoryFrame } from "../lib/types";
import { findIndexAtOrBefore } from "../lib/timeSeries";

const NAME_FONT_SIZE = 20;
const DEBUG_BOX_STROKE_WIDTH = 3;
const DEBUG_LABEL_FONT_SIZE = 16;

function colorForId(stableId: number): string {
  const hue = (stableId * 47) % 360;
  return `hsl(${hue}, 75%, 60%)`;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// A brief occlusion (blocked by another player, dives out of frame, net in
// the way) drops a player from one or two samples before the tracker picks
// them back up - gliding across that whole gap reads as continued motion.
// A longer disappearance (substitution, tracker never reacquires them)
// does not: linearly interpolating across many seconds would show the box
// visibly drifting toward wherever they happen to reappear, which isn't
// where they actually were in between. So only bridge gaps up to this
// long; beyond it, the caller falls back to holding the last known box.
const MAX_GLIDE_GAP_S = 2;

// Looks past the immediate next sample for the first later one that still
// has `stableId`, up to maxGapS ahead of `fromT` - lets a player who drops
// out for a couple of samples keep gliding instead of freezing and then
// snapping into place once they're picked up again.
function findNextMatch(
  frames: PlayerTrajectoryFrame[],
  fromIdx: number,
  stableId: number,
  fromT: number,
  maxGapS: number,
): { box: PlayerBox; t: number } | null {
  for (let j = fromIdx; j < frames.length; j++) {
    const frame = frames[j];
    if (frame.t - fromT > maxGapS) break;
    const match = frame.players.find((p) => p.stable_id === stableId);
    if (match) return { box: match, t: frame.t };
  }
  return null;
}

interface PlayerTrackingOverlayProps {
  frames: PlayerTrajectoryFrame[];
  currentTimeS: number;
  // The fixed resolution frames[].players[].box is measured in -
  // VideoPlayer's annotationSize, NOT the <video> element's own decoded
  // videoWidth/videoHeight (see BallTrackingOverlay's identically-named
  // prop for why those two can differ).
  videoWidth: number;
  videoHeight: number;
  // The default, simplified view: just each player's name (or "#<id>" for
  // one not yet identified), centered under their feet - no box.
  showNames: boolean;
  // Debug view: the original box outline plus a label carrying BOTH the
  // resolved name and the raw stable_id, so a box's identification can be
  // checked against its underlying tracking id. Independent of showNames -
  // both can be on together.
  showDebug: boolean;
}

// Player positions only arrive sampled ~2x/second (see
// PLAYER_TRAJECTORY_STRIDE) - holding each box at its last known sample
// until the next one arrives makes tracking visibly snap/step every half
// second, which reads as "laggy" regardless of how cheap the re-render
// itself is. Linearly interpolating each player's box between the
// surrounding two samples (matched by stable_id) makes it glide smoothly
// in between instead, at the cost of needing to recompute on every
// timeupdate tick rather than only when the sample index changes - a
// tradeoff worth making since ~12 players' worth of SVG nodes is cheap to
// re-render regardless; the actual complaint was the stepping motion, not
// render cost. When a player is missing from the very next sample (a brief
// occlusion), findNextMatch looks further ahead so the glide continues
// across the gap instead of freezing - see MAX_GLIDE_GAP_S for how far.
function PlayerTrackingOverlayImpl({
  frames,
  currentTimeS,
  videoWidth,
  videoHeight,
  showNames,
  showDebug,
}: PlayerTrackingOverlayProps) {
  if (frames.length === 0 || videoWidth <= 0 || videoHeight <= 0 || (!showNames && !showDebug)) return null;

  const idx = Math.max(0, findIndexAtOrBefore(frames, currentTimeS));
  const frameA = frames[idx];
  const frameB = idx + 1 < frames.length ? frames[idx + 1] : null;
  const nextById = frameB ? new Map(frameB.players.map((p): [number, PlayerBox] => [p.stable_id, p])) : null;

  return (
    <Box
      component="svg"
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      preserveAspectRatio="xMidYMid meet"
      sx={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }}
    >
      {frameA.players.map((p) => {
        // Prefer the immediate next sample; if this player dropped out of
        // it (occluded, off-frame), look further ahead within
        // MAX_GLIDE_GAP_S. Beyond that - or if they never reappear - just
        // hold at their last known box rather than jumping/vanishing.
        const immediate = nextById?.get(p.stable_id);
        const found = immediate
          ? { box: immediate, t: frameB!.t }
          : findNextMatch(frames, idx + 2, p.stable_id, frameA.t, MAX_GLIDE_GAP_S);
        const frac =
          found && found.t > frameA.t
            ? Math.min(1, Math.max(0, (currentTimeS - frameA.t) / (found.t - frameA.t)))
            : 0;
        const [x1, y1, x2, y2] = found
          ? [
              lerp(p.box[0], found.box.box[0], frac),
              lerp(p.box[1], found.box.box[1], frac),
              lerp(p.box[2], found.box.box[2], frac),
              lerp(p.box[3], found.box.box[3], frac),
            ]
          : p.box;
        const color = colorForId(p.stable_id);
        const cx = (x1 + x2) / 2;
        return (
          <g key={p.stable_id}>
            {showDebug && (
              <>
                <rect
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill="none"
                  stroke={color}
                  strokeWidth={DEBUG_BOX_STROKE_WIDTH}
                />
                <text
                  x={x1}
                  y={Math.max(DEBUG_LABEL_FONT_SIZE, y1 - 6)}
                  fontSize={DEBUG_LABEL_FONT_SIZE}
                  fontWeight={600}
                  fill={color}
                  stroke="#0b1220"
                  strokeWidth={0.6}
                  paintOrder="stroke"
                >
                  {p.name ? `${p.name} · #${p.stable_id}` : `#${p.stable_id}`}
                </text>
              </>
            )}
            {showNames && (
              <text
                x={cx}
                y={y2 + NAME_FONT_SIZE}
                fontSize={NAME_FONT_SIZE}
                fontWeight={600}
                textAnchor="middle"
                fill="#fff"
                stroke="#0b1220"
                strokeWidth={0.6}
                paintOrder="stroke"
              >
                {p.name ?? `#${p.stable_id}`}
              </text>
            )}
          </g>
        );
      })}
    </Box>
  );
}

export const PlayerTrackingOverlay = memo(PlayerTrackingOverlayImpl);
