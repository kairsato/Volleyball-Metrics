import { memo } from "react";
import Box from "@mui/material/Box";
import type { BallTrajectoryPoint } from "../lib/types";
import { findIndexAtOrBefore } from "../lib/timeSeries";

const BALL_COLOR = "#facc15";
const BALL_OUTLINE_COLOR = "#0b1220";
// In source-video pixel units, not screen CSS pixels - the SVG's viewBox
// (see below) scales these down together with everything else as the
// player's actual displayed size differs from the video's native
// resolution, so this stays a reasonable size on screen regardless.
const BALL_RADIUS_PX = 14;
const BALL_RING_WIDTH_PX = 4;

// Data now arrives at full per-frame density (see results_router's
// BALL_TRAJECTORY_STRIDE) rather than sparse samples, so consecutive points
// are normally only a frame apart - gliding between them is mostly just
// smoothing sub-frame timing at this point, not bridging real gaps the way
// it used to at the old, much coarser stride. A gap bigger than this still
// means the trajectory genuinely has no data there (the ball was out of
// frame - see ball_trajectory_smoother.py's out_of_frame status), so beyond
// it the ball just holds at its last known spot rather than sliding across
// a gap that isn't real motion.
const MAX_GLIDE_GAP_S = 1.0;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

interface BallTrackingOverlayProps {
  // Already filtered to px/py-having points by the caller (VideoPlayer),
  // and stable across re-renders (memoized there).
  points: BallTrajectoryPoint[];
  // Current playback time - this component resolves its own index and
  // interpolates between samples (see MAX_GLIDE_GAP_S), the same pattern
  // PlayerTrackingOverlay uses, rather than snapping to a pre-resolved
  // index the way this component originally did.
  currentTimeS: number;
  // The fixed resolution points[].px/py are actually measured in -
  // VideoPlayer's annotationSize, NOT the <video> element's own decoded
  // videoWidth/videoHeight (those change if a lower-bitrate playback
  // quality is selected; the tracked coordinates never do).
  videoWidth: number;
  videoHeight: number;
}

// Draws the ball directly on the video at its actual on-screen position
// (unlike BallMinimap's small corner court diagram) - an SVG sized to the
// video's own intrinsic resolution with preserveAspectRatio="xMidYMid
// meet" (SVG's equivalent of the video element's own object-fit:contain),
// so it letterboxes/scales/re-centers itself in lockstep with the video
// automatically as the player resizes, with no manual layout math needed.
// The marker is a hollow ring (not a filled dot) so it reads as "circling
// the ball" rather than covering it up. Deliberately just the ring, no
// trailing line - a trail (even a fading one) reads as a smoothed/predicted
// path rather than the tracker's own real position each frame; see
// BallTrajectoryOverlay for the separate, explicit "show me the shape of
// this shot" visualization that a curve is actually right for.
function BallTrackingOverlayImpl({ points, currentTimeS, videoWidth, videoHeight }: BallTrackingOverlayProps) {
  if (points.length === 0 || videoWidth <= 0 || videoHeight <= 0) return null;

  const idx = findIndexAtOrBefore(points, currentTimeS);
  if (idx < 0) return null;

  const current = points[idx];
  if (current.px === null || current.py === null) return null;

  const next = idx + 1 < points.length ? points[idx + 1] : null;
  const glideOk = next !== null && next.px !== null && next.py !== null && next.t - current.t <= MAX_GLIDE_GAP_S;
  const frac = glideOk ? Math.min(1, Math.max(0, (currentTimeS - current.t) / (next!.t - current.t))) : 0;
  const cx = glideOk ? lerp(current.px, next!.px as number, frac) : current.px;
  const cy = glideOk ? lerp(current.py, next!.py as number, frac) : current.py;

  return (
    <Box
      component="svg"
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      preserveAspectRatio="xMidYMid meet"
      sx={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }}
    >
      <circle cx={cx} cy={cy} r={BALL_RADIUS_PX} fill="none" stroke={BALL_OUTLINE_COLOR} strokeWidth={BALL_RING_WIDTH_PX + 2} />
      <circle cx={cx} cy={cy} r={BALL_RADIUS_PX} fill="none" stroke={BALL_COLOR} strokeWidth={BALL_RING_WIDTH_PX} />
    </Box>
  );
}

export const BallTrackingOverlay = memo(BallTrackingOverlayImpl);
