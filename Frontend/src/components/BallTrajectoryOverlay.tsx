import { memo, useMemo } from "react";
import Box from "@mui/material/Box";
import type { BallTrajectoryPoint } from "../lib/types";
import { findIndexAtOrBefore } from "../lib/timeSeries";

// A shot shorter than this (in real recorded points, or in total duration)
// doesn't get an arc drawn at all - too little of the flight was actually
// tracked to fit a meaningful parabola through, and a "curve" fit to 2-3
// scattered points nearby in time is more likely to mislead than to help.
const MIN_ARC_POINTS = 5;
const MIN_ARC_DURATION_S = 0.15;

// How much of a shot's own vertical excursion (apex height minus each
// end's height) to show on each side, as a fraction - see
// trajectoryClipRange. 0.5 shows the top half of the arc by height on both
// the way up and the way down.
const RISE_FRACTION = 0.5;

const ARC_COLOR = "#38bdf8";
const ARC_OUTLINE_COLOR = "#0b1220";

// Largest entry at or before `t` in a plain sorted-ascending number array -
// same idea as timeSeries.ts's findIndexAtOrBefore, just over hit
// timestamps directly rather than objects with a .t field.
function findLastAtOrBefore(sorted: number[], t: number): number {
  let lo = 0;
  let hi = sorted.length - 1;
  let result = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= t) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result;
}

// Ordinary least-squares fit of v = a + b*t + c*t^2, via the normal
// equations' closed-form 3x3 solve (Cramer's rule - cheap and exact for a
// system this size, no need for a general linear-algebra dependency just
// for this). Returns null if the system is singular (fewer than 3 distinct
// t values - see fitParabola, which never calls this with fewer than 3
// points to begin with).
function fitQuadratic(t: number[], v: number[]): [number, number, number] | null {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, sy0 = 0, sy1 = 0, sy2 = 0;
  for (let i = 0; i < t.length; i++) {
    const ti = t[i], vi = v[i];
    const ti2 = ti * ti;
    s0 += 1; s1 += ti; s2 += ti2; s3 += ti2 * ti; s4 += ti2 * ti2;
    sy0 += vi; sy1 += ti * vi; sy2 += ti2 * vi;
  }
  // |s0 s1 s2|
  // |s1 s2 s3|
  // |s2 s3 s4|
  const det =
    s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2);
  if (Math.abs(det) < 1e-9) return null;

  const detA =
    sy0 * (s2 * s4 - s3 * s3) - s1 * (sy1 * s4 - s3 * sy2) + s2 * (sy1 * s3 - s2 * sy2);
  const detB =
    s0 * (sy1 * s4 - s3 * sy2) - sy0 * (s1 * s4 - s3 * s2) + s2 * (s1 * sy2 - sy1 * s2);
  const detC =
    s0 * (s2 * sy2 - sy1 * s3) - s1 * (s1 * sy2 - sy1 * s2) + sy0 * (s1 * s3 - s2 * s2);

  return [detA / det, detB / det, detC / det];
}

// Fits one smooth parabola through a shot's own recorded points, rather
// than connecting them with a straight-segment polyline the way the raw
// track (independently noisy per frame, and accurate rather than smooth by
// design - see BallDetection.ballDetection's own module comment) would
// draw. px(t) and py(t) are fit independently, but since both are quadratic
// in the SAME shared parameter t, the (px, py) locus they trace is itself a
// genuine parabola (or, in the degenerate case where their t^2 terms are
// proportional, a straight line) - not just a curve that looks smooth.
// Returns null for fewer than 3 points (too few to meaningfully constrain a
// quadratic) or a singular fit.
interface ParabolicFit {
  x: [number, number, number]; // px(t) = x[0] + x[1]*t + x[2]*t^2
  y: [number, number, number]; // py(t) = y[0] + y[1]*t + y[2]*t^2
}

function fitParabola(path: { t: number; px: number; py: number }[]): ParabolicFit | null {
  if (path.length < 3) return null;
  const t = path.map((p) => p.t);
  const fitX = fitQuadratic(t, path.map((p) => p.px));
  const fitY = fitQuadratic(t, path.map((p) => p.py));
  if (!fitX || !fitY) return null;
  return { x: fitX, y: fitY };
}

function evalParabola(fit: ParabolicFit, t: number): { px: number; py: number } {
  return {
    px: fit.x[0] + fit.x[1] * t + fit.x[2] * t * t,
    py: fit.y[0] + fit.y[1] * t + fit.y[2] * t * t,
  };
}

// The single root of a*t^2 + b*t + c = 0 on the given side of the parabola's
// own vertex (a > 0: "lower" is the root left of the vertex, "upper" the one
// right of it - see trajectoryClipRange, the only caller). Null if the
// equation has no real root.
function quadraticRoot(a: number, b: number, c: number, side: "lower" | "upper"): number | null {
  const discriminant = b * b - 4 * a * c;
  if (discriminant < 0) return null;
  const sq = Math.sqrt(discriminant);
  return side === "lower" ? (-b - sq) / (2 * a) : (-b + sq) / (2 * a);
}

// Where to start and stop drawing the arc, based purely on the shape of the
// fitted trajectory itself rather than an absolute height: finds the fit's
// own apex (its highest point on screen, i.e. minimum py) within
// [segStart, segEnd], then - independently on each side - solves for the
// moment the ball has risen RISE_FRACTION of the way from that side's own
// starting height up to the apex. A shot is clipped relative to its OWN
// observed rise, not a fixed metres-above-the-court line, so this needs no
// camera calibration/height_m at all and treats every shot's own start and
// end height on its own terms (a serve from behind the baseline and a set
// tossed from just above the net don't share a "how high is high enough"
// answer, but they do both mean "roughly halfway up from here").
//
// Falls back to the full [segStart, segEnd] range when the fit doesn't
// curve like a real trajectory should in screen space (y[2] <= 0, e.g. too
// few/noisy points to pin down real curvature) - nothing reliable to find
// an apex from in that case, so showing the whole thing beats guessing.
function trajectoryClipRange(fit: ParabolicFit, segStart: number, segEnd: number): [number, number] {
  const [ay, by, cy] = fit.y;
  if (cy <= 0) return [segStart, segEnd];

  const tApex = Math.min(segEnd, Math.max(segStart, -by / (2 * cy)));
  const pyApex = evalParabola(fit, tApex).py;

  const pyStart = evalParabola(fit, segStart).py;
  const startTarget = pyStart - RISE_FRACTION * (pyStart - pyApex);
  const startRoot = quadraticRoot(cy, by, ay - startTarget, "lower");
  const clipStart = startRoot === null ? segStart : Math.min(Math.max(startRoot, segStart), tApex);

  const pyEnd = evalParabola(fit, segEnd).py;
  const endTarget = pyEnd - RISE_FRACTION * (pyEnd - pyApex);
  const endRoot = quadraticRoot(cy, by, ay - endTarget, "upper");
  const clipEnd = endRoot === null ? segEnd : Math.max(Math.min(endRoot, segEnd), tApex);

  return clipEnd > clipStart ? [clipStart, clipEnd] : [segStart, segEnd];
}

// Samples the fit at even time intervals across [tStart, tEnd] - the actual
// polyline drawn to the screen.
function sampleParabola(fit: ParabolicFit, tStart: number, tEnd: number): { t: number; px: number; py: number }[] {
  const SAMPLES = 40;
  const out: { t: number; px: number; py: number }[] = [];
  for (let i = 0; i <= SAMPLES; i++) {
    const s = tStart + ((tEnd - tStart) * i) / SAMPLES;
    out.push({ t: s, ...evalParabola(fit, s) });
  }
  return out;
}

interface BallTrajectoryOverlayProps {
  // Full per-frame trajectory, already filtered to px/py-having points by
  // the caller (VideoPlayer) - same array BallTrackingOverlay uses.
  points: BallTrajectoryPoint[];
  // Every player-touch timestamp, warmup-relative, sorted ascending.
  hitTimestamps: number[];
  currentTimeS: number;
  // The fixed resolution points[].px/py is measured in - see
  // BallTrackingOverlay's identically-named prop.
  videoWidth: number;
  videoHeight: number;
}

// Draws the current hit-to-hit flight as ONE smooth, FIXED parabola fit
// through its real recorded points (fitParabola/sampleParabola), distinct
// from (and independent of) BallTrackingOverlay's single point-in-time ring
// - "here's the shape of this shot" rather than "here's the ball right
// now." This is deliberately a display-only fit: the underlying tracked
// points (BallDetection.ballDetection's own output) stay exactly as
// detected, un-smoothed, since accuracy is what matters for the actual
// tracking data - see that module's comment on refine_camera_pose_from_flight.
//
// The curve is fit ONCE from the whole shot's own points (memoized on the
// shot's identity, not on currentTimeS) and then REVEALED as playback moves
// along it - the shape is precalculated and fixed, and the only thing
// currentTimeS controls is how far along that fixed shape has been drawn.
// It does NOT get re-fit with a growing point set as playback advances: an
// earlier version did exactly that, and the drawn shape visibly shifted
// underneath the ball as more data came in.
//
// Bounded by the touches immediately before/after the current time (a new
// flight starts fresh at each touch), and clipped at both ends to the
// dramatic middle of the arc based on the fitted trajectory's OWN shape
// (trajectoryClipRange) - not an absolute height, which needs a solved
// camera pose that most jobs don't have (height_m is null for every point
// otherwise, silently disabling a height-based clip). A shot with too few
// real points or too little duration (MIN_ARC_POINTS/MIN_ARC_DURATION_S)
// draws nothing rather than a "curve" fit to a handful of scattered,
// barely-separated points.
function BallTrajectoryOverlayImpl({
  points,
  hitTimestamps,
  currentTimeS,
  videoWidth,
  videoHeight,
}: BallTrajectoryOverlayProps) {
  const lastHitIdx = findLastAtOrBefore(hitTimestamps, currentTimeS);
  const segmentStart = lastHitIdx >= 0 ? hitTimestamps[lastHitIdx] : (points[0]?.t ?? 0);
  const segmentEnd = lastHitIdx + 1 < hitTimestamps.length ? hitTimestamps[lastHitIdx + 1] : Infinity;

  const path = useMemo(() => {
    if (points.length === 0) return null;

    const startIdx = Math.max(0, findIndexAtOrBefore(points, segmentStart));
    const endIdx = segmentEnd === Infinity ? points.length - 1 : Math.max(startIdx, findIndexAtOrBefore(points, segmentEnd));

    const rawPath: { t: number; px: number; py: number }[] = [];
    for (let i = startIdx; i <= endIdx; i++) {
      const p = points[i];
      if (p.px === null || p.py === null) continue;
      rawPath.push({ t: p.t, px: p.px, py: p.py });
    }

    const durationS = rawPath.length > 0 ? rawPath[rawPath.length - 1].t - rawPath[0].t : 0;
    if (rawPath.length < MIN_ARC_POINTS || durationS < MIN_ARC_DURATION_S) return null;

    const fit = fitParabola(rawPath);
    if (!fit) return null;

    const [clipStart, clipEnd] = trajectoryClipRange(fit, rawPath[0].t, rawPath[rawPath.length - 1].t);
    return sampleParabola(fit, clipStart, clipEnd);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, segmentStart, segmentEnd]);

  if (!path || videoWidth <= 0 || videoHeight <= 0) return null;
  // Only while actually within this shot's own window - the fit above
  // doesn't depend on currentTimeS, but whether to show it at all still
  // does (nothing to draw before the shot's first touch, or once we've
  // moved on to the next one).
  if (currentTimeS < segmentStart || currentTimeS >= segmentEnd) return null;

  // The curve's SHAPE is already fixed (fit above, memoized); playback only
  // controls how much of that fixed shape has been drawn so far, so the arc
  // is revealed by the ball travelling along it rather than being redrawn
  // as it goes.
  const revealed = path.filter((p) => p.t <= currentTimeS);
  if (revealed.length < 2) return null;

  const pointsAttr = revealed.map((p) => `${p.px},${p.py}`).join(" ");

  return (
    <Box
      component="svg"
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      preserveAspectRatio="xMidYMid meet"
      sx={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 1 }}
    >
      <polyline
        points={pointsAttr}
        fill="none"
        stroke={ARC_OUTLINE_COLOR}
        strokeWidth={6}
        strokeOpacity={0.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <polyline
        points={pointsAttr}
        fill="none"
        stroke={ARC_COLOR}
        strokeWidth={3}
        strokeOpacity={0.85}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Box>
  );
}

export const BallTrajectoryOverlay = memo(BallTrajectoryOverlayImpl);
