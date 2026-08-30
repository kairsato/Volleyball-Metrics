import { useEffect, useRef, useState } from "react";
import type { MouseEvent } from "react";
import "./CalibrationPanel.css";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import type { SelectChangeEvent } from "@mui/material/Select";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import { predictCourtGeometry } from "../lib/courtGeometry";
import type { Job, Point, PredictedCourtGeometry } from "../lib/types";
import { LoadingSpinner } from "./LoadingSpinner";
import { LockOverlay } from "./LockOverlay";

function LegendSwatch({ color }: { color: string }) {
  return (
    <Box
      sx={{
        width: 12,
        height: 12,
        borderRadius: "50%",
        bgcolor: "transparent",
        border: `2.5px solid ${color}`,
        flexShrink: 0,
      }}
    />
  );
}

function LegendLine({ color, dashed }: { color: string; dashed?: boolean }) {
  return (
    <Box
      sx={{
        width: 18,
        borderTop: `3px ${dashed ? "dashed" : "solid"} ${color}`,
        flexShrink: 0,
      }}
    />
  );
}

// The 4 points the user actually places, in the exact order the backend
// expects (calibration.POINT_NAMES) - both ends of the net line, then both
// ends of the far baseline. Everything else (the near baseline, both
// attack lines) is predicted from these, never marked directly.
const POINT_TITLES = ["Middle Line Left", "Middle Line Right", "Close Line Left", "Close Line Right"];

const NET_HEIGHT_OPTIONS: { value: number; label: string }[] = [
  { value: 2.43, label: "Men's / mixed (2.43m)" },
  { value: 2.24, label: "Women's (2.24m)" },
];
const DEFAULT_NET_HEIGHT_M = 2.43;

// Common volleyball court line colours - picking the one that actually
// matches the venue lets the preview enhancement (see enhanceStrength
// below) saturate pixels near that colour and desaturate everything else
// toward gray, making the lines pop against a muted floor instead of
// blending in while dragging a point onto them.
// Reference RGBs are picked for a centred, "generic" version of each hue
// (pure green ~120 degrees, pure blue ~240, etc.) rather than a specific
// brand-y shade - real court paint/tape varies a lot with lighting and
// camera colour reproduction, and matching is by hue family (see
// ENHANCE_HUE_THRESHOLD below), not an exact colour, so a broad/typical
// reference catches more real-world variation than a narrow, precise one.
const LINE_COLOR_OPTIONS: { value: string; label: string; rgb: [number, number, number] }[] = [
  { value: "green", label: "Green", rgb: [50, 200, 60] },
  { value: "white", label: "White", rgb: [255, 255, 255] },
  { value: "yellow", label: "Yellow", rgb: [255, 214, 10] },
  { value: "blue", label: "Blue", rgb: [37, 99, 235] },
  { value: "red", label: "Red", rgb: [239, 68, 68] },
  { value: "orange", label: "Orange", rgb: [249, 115, 22] },
  { value: "black", label: "Black", rgb: [25, 25, 25] },
];
const DEFAULT_LINE_COLOR = "green";
const DEFAULT_ENHANCE_STRENGTH = 25;

// How close a pixel's hue needs to be to the selected colour's hue (as a
// fraction of the full 360 degree wheel, so 0.22 is roughly 80 degrees)
// before it counts as "matching" at all - generously wide, since real
// court lines rarely land on an exact textbook hue once lighting and a
// camera's own colour reproduction are involved; a narrow match tends to
// just miss real footage entirely rather than mis-highlight the floor.
const ENHANCE_HUE_THRESHOLD = 0.22;
// Same idea but for white/black, which have no meaningful hue - matched by
// closeness in lightness instead (as a fraction of the 0-1 lightness range).
const ENHANCE_LIGHTNESS_THRESHOLD = 0.45;

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];

  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h: number;
  if (max === rn) h = (gn - bn) / d + (gn < bn ? 6 : 0);
  else if (max === gn) h = (bn - rn) / d + 2;
  else h = (rn - gn) / d + 4;
  return [h / 6, s, l];
}

function hue2rgb(p: number, q: number, t: number): number {
  let tt = t;
  if (tt < 0) tt += 1;
  if (tt > 1) tt -= 1;
  if (tt < 1 / 6) return p + (q - p) * 6 * tt;
  if (tt < 1 / 2) return q;
  if (tt < 2 / 3) return p + (q - p) * (2 / 3 - tt) * 6;
  return p;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) return [l * 255, l * 255, l * 255];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [
    hue2rgb(p, q, h + 1 / 3) * 255,
    hue2rgb(p, q, h) * 255,
    hue2rgb(p, q, h - 1 / 3) * 255,
  ];
}

// Grows every value in a w*h grid outward by `radius`, each cell taking on
// the strongest value found within that radius - a morphological max
// filter, done as two 1D passes (horizontal then vertical) rather than one
// full 2D pass, since that's the same result for a square neighbourhood at
// a fraction of the work. This is what turns a court line that's only a
// couple of pixels wide in the source frame into a visibly thicker
// highlighted band in the preview.
function dilateMax(src: Float32Array, w: number, h: number, radius: number): Float32Array {
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const rowStart = y * w;
    for (let x = 0; x < w; x++) {
      let m = 0;
      const xMin = Math.max(0, x - radius);
      const xMax = Math.min(w - 1, x + radius);
      for (let xx = xMin; xx <= xMax; xx++) {
        const v = src[rowStart + xx];
        if (v > m) m = v;
      }
      tmp[rowStart + x] = m;
    }
  }

  const out = new Float32Array(w * h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let m = 0;
      const yMin = Math.max(0, y - radius);
      const yMax = Math.min(h - 1, y + radius);
      for (let yy = yMin; yy <= yMax; yy++) {
        const v = tmp[yy * w + x];
        if (v > m) m = v;
      }
      out[y * w + x] = m;
    }
  }
  return out;
}

// Mirrors the backend's default_points(): the far baseline (smaller,
// further from the camera) sits higher and narrower than the net line
// (closer, wider) - a typical sideline view's rough shape, just a
// starting position to drag from. Chosen so the predicted near baseline
// and both attack lines land within the visible frame for this starting
// shape - a steeper far/close ratio pushes the predicted near baseline
// far outside the frame before the user has even touched a point.
function presetPoints(width: number, height: number): Point[] {
  return [
    { x: width * 0.26, y: height * 0.5 }, // middle_left
    { x: width * 0.74, y: height * 0.5 }, // middle_right
    { x: width * 0.29, y: height * 0.35 }, // far_left
    { x: width * 0.71, y: height * 0.35 }, // far_right
  ];
}

// How far past the frame edge a marker is allowed to go, as a fraction of
// that dimension - lets a corner just out of frame still be marked.
const OVERFLOW_FRACTION = 0.08;

// Loupe shown while dragging. Normally zoomed into the source frame's
// native pixels (not the scaled-down display) so fine alignment is
// actually possible; while the Preview colour enhancement is active it
// switches to showing that same enhanced picture instead - matching what's
// actually driving where you're aiming, at the cost of zooming into a
// lower-resolution (display-sized) source for as long as that's on.
const MAGNIFIER_SIZE = 195;
const MAGNIFIER_ZOOM = 3;

function clampToFrame(value: number, dimension: number): number {
  const allowance = dimension * OVERFLOW_FRACTION;
  return Math.max(-allowance, Math.min(dimension + allowance, value));
}

// Every measurement below is a fraction of the diagram's own viewBox
// dimensions, not an absolute pixel - the same shape (close line low and
// wide, far/middle line high and narrow, exactly like real perspective)
// renders correctly whether the diagram is drawn wide (a normal landscape
// recording) or tall (a phone held vertically) just by handing it a
// different viewW/viewH.
const CLOSE_Y_FRACTION = 0.542;
const CLOSE_HALF_WIDTH_FRACTION = 0.275;
const FAR_Y_FRACTION = 0.158;
const FAR_HALF_WIDTH_FRACTION = 0.131;
const ATTACK_FAR_Y_FRACTION = 0.35;
const ATTACK_FAR_HALF_WIDTH_FRACTION = 0.194;
const ATTACK_NEAR_Y_FRACTION = 0.733;
const ATTACK_NEAR_HALF_WIDTH_FRACTION = 0.356;
const NEAR_Y_FRACTION = 0.908;
const NEAR_HALF_WIDTH_FRACTION = 0.4375;

function courtDiagramGeometry(viewW: number, viewH: number) {
  const cx = viewW / 2;
  const row = (yFraction: number, halfWidthFraction: number) => ({
    left: { x: cx - viewW * halfWidthFraction, y: viewH * yFraction },
    right: { x: cx + viewW * halfWidthFraction, y: viewH * yFraction },
  });
  return {
    far: row(FAR_Y_FRACTION, FAR_HALF_WIDTH_FRACTION),
    close: row(CLOSE_Y_FRACTION, CLOSE_HALF_WIDTH_FRACTION),
    attackFar: row(ATTACK_FAR_Y_FRACTION, ATTACK_FAR_HALF_WIDTH_FRACTION),
    attackNear: row(ATTACK_NEAR_Y_FRACTION, ATTACK_NEAR_HALF_WIDTH_FRACTION),
    near: row(NEAR_Y_FRACTION, NEAR_HALF_WIDTH_FRACTION),
  };
}

// Both diagrams sit in a square box of exactly this size, side by side -
// the actual rendered diagram is fit inside via the same "contain" idea as
// object-fit (preserve its own aspect ratio, centred, letterboxed on
// whichever axis has room left over) rather than stretched to fill it.
const DIAGRAM_BOX_SIZE = 250;

// A compact schematic - not a real photo, since no single frame of a real
// match is guaranteed to show every landmark clearly - showing exactly
// what each of the 4 points refers to on an idealized court, viewed from
// a typical sideline angle. The solid quad is what you actually mark; the
// dashed lines are what gets predicted from it, same visual language as
// the live canvas. The net line (labelled Middle Line here) always stays
// the lower/wider (nearer-camera) pair and the far baseline (labelled
// Close Line) the upper/narrower one - that doesn't flip between
// orientations, only the overall frame's proportions do.
function CourtDiagram({
  viewW,
  viewH,
  caption,
  rotateDeg = 0,
}: {
  viewW: number;
  viewH: number;
  caption: string;
  rotateDeg?: number;
}) {
  const g = courtDiagramGeometry(viewW, viewH);
  const scale = Math.min(viewW, viewH);
  const points = [
    { ...g.far.left, label: "CL" },
    { ...g.far.right, label: "CR" },
    { ...g.close.left, label: "ML" },
    { ...g.close.right, label: "MR" },
  ];

  const rotated = rotateDeg !== 0;
  // The displayed aspect ratio AFTER any rotation - a 90-degree turn
  // swaps which of the viewBox's own dimensions ends up as the width.
  const displayAspect = rotated ? viewH / viewW : viewW / viewH;
  // Fit that aspect ratio inside the fixed square box, same idea as
  // object-fit: contain - whichever axis is relatively taller than the box
  // is the one that ends up flush with it, the other has room to spare.
  const postW = displayAspect >= 1 ? DIAGRAM_BOX_SIZE : DIAGRAM_BOX_SIZE * displayAspect;
  const postH = displayAspect >= 1 ? DIAGRAM_BOX_SIZE / displayAspect : DIAGRAM_BOX_SIZE;
  // The SVG's own pre-rotation width/height - swapped back from the
  // post-rotation numbers above for a rotated diagram, since the rotation
  // itself is what will swap them again on screen.
  const svgWidth = rotated ? postH : postW;
  const svgHeight = rotated ? postW : postH;

  return (
    <Box sx={{ flexShrink: 0 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
        {caption}
      </Typography>
      <Box
        sx={{
          width: DIAGRAM_BOX_SIZE,
          height: DIAGRAM_BOX_SIZE,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Box
          component="svg"
          viewBox={`0 0 ${viewW} ${viewH}`}
          sx={{
            width: svgWidth,
            height: svgHeight,
            display: "block",
            transform: rotated ? `rotate(${rotateDeg}deg)` : undefined,
            border: "1px solid rgba(127, 127, 127, 0.3)",
            borderRadius: "10px",
            bgcolor: "rgba(255,255,255,0.02)",
          }}
        >
          {/* Predicted (never marked): far attack line, near attack line,
              near baseline, and the sideline extensions down to it -
              exactly the 5 predicted lines the live canvas itself draws
              (see the .calibration-overlay svg below in this file) - each
              row's width grows steadily as it gets closer to the camera,
              matching the real court's perspective. */}
          <line x1={g.attackFar.left.x} y1={g.attackFar.left.y} x2={g.attackFar.right.x} y2={g.attackFar.right.y} className="predicted-outline" />
          <line x1={g.attackNear.left.x} y1={g.attackNear.left.y} x2={g.attackNear.right.x} y2={g.attackNear.right.y} className="predicted-outline" />
          <line x1={g.near.left.x} y1={g.near.left.y} x2={g.near.right.x} y2={g.near.right.y} className="predicted-outline" />
          <line x1={g.far.left.x} y1={g.far.left.y} x2={g.near.left.x} y2={g.near.left.y} className="predicted-outline" />
          <line x1={g.far.right.x} y1={g.far.right.y} x2={g.near.right.x} y2={g.near.right.y} className="predicted-outline" />
          {/* Marked: far baseline + net line, and the sidelines joining them */}
          <line x1={g.far.left.x} y1={g.far.left.y} x2={g.far.right.x} y2={g.far.right.y} stroke="#ffd400" strokeWidth="3" />
          <line x1={g.close.left.x} y1={g.close.left.y} x2={g.close.right.x} y2={g.close.right.y} stroke="#38bdf8" strokeWidth="3" />
          <line x1={g.far.left.x} y1={g.far.left.y} x2={g.close.left.x} y2={g.close.left.y} stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" />
          <line x1={g.far.right.x} y1={g.far.right.y} x2={g.close.right.x} y2={g.close.right.y} stroke="rgba(255,255,255,0.5)" strokeWidth="1.5" />

          {points.map((p) => (
            <g key={p.label}>
              <circle cx={p.x} cy={p.y} r={scale * 0.03} fill="none" stroke="#aa3bff" strokeWidth="3" />
              <text
                x={p.x}
                y={p.y - scale * 0.055}
                fill="#fff"
                fontSize={scale * 0.048}
                fontWeight="700"
                textAnchor="middle"
              >
                {p.label}
              </text>
            </g>
          ))}
        </Box>
      </Box>
    </Box>
  );
}

function CourtDiagramExample() {
  return (
    <Card variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        What to look for
      </Typography>
      <Stack direction="row" spacing={1.5} sx={{ mb: 1.5, overflowX: "auto" }}>
        <CourtDiagram viewW={320} viewH={180} caption="Horizontal recording" rotateDeg={-90} />
        <CourtDiagram viewW={180} viewH={320} caption="Vertical recording" />
      </Stack>
      <Stack spacing={0.75}>
        <Typography variant="body2" color="text.secondary">
          <strong>ML / MR</strong> - where the net's bottom edge meets each sideline.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          <strong>CL / CR</strong> - the baseline furthest from the camera, where it meets each
          sideline.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          The near baseline and both attack lines (dashed) are worked out automatically - you never
          need to find them yourself.
        </Typography>
      </Stack>
    </Card>
  );
}

interface CalibrationPanelProps {
  job: Job;
}

export function CalibrationPanel({ job }: CalibrationPanelProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [points, setPoints] = useState<Point[] | null>(null);
  const [netHeightM, setNetHeightM] = useState(DEFAULT_NET_HEIGHT_M);
  // Persisted server-side (see calibration.set_confirmed/court.json's
  // "confirmed" key) - same confirmed/locked/redo pattern as scoring and
  // player identification. This used to be local-only React state that
  // reset itself on every reload, which meant "Redo Court Identification"
  // looked like it didn't stick as soon as you left the page.
  const [confirmed, setConfirmed] = useState(false);
  const [redoDialogOpen, setRedoDialogOpen] = useState(false);
  const [redoing, setRedoing] = useState(false);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  // Purely a viewing aid - never sent anywhere, doesn't touch the saved
  // calibration, and isn't persisted anywhere (plain component state,
  // resets on reload) - so it stays usable even while everything else is
  // locked, and there's no "confirmed"-style save step for any of it.
  const [lineColor, setLineColor] = useState(DEFAULT_LINE_COLOR);
  const [enhanceStrength, setEnhanceStrength] = useState(DEFAULT_ENHANCE_STRENGTH);
  const [pointsVisible, setPointsVisible] = useState(true);
  // The enhanced preview redrawn as a data URL, purely so the drag
  // magnifier can show the same reference picture as the main view while
  // enhancement is on - null whenever it's off (magnifier then falls back
  // to the crisp native-resolution frame instead).
  const [enhancedDataUrl, setEnhancedDataUrl] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const enhanceCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    async function load() {
      try {
        const [url, pointsRes] = await Promise.all([
          api.getCalibrationFrame(job.id),
          api.getCalibrationPoints(job.id),
        ]);
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        objectUrl = url;
        setFrameUrl(url);
        setPoints([pointsRes.middle_left, pointsRes.middle_right, pointsRes.far_left, pointsRes.far_right]);
        setNetHeightM(pointsRes.net_height_m);
        setConfirmed(pointsRes.confirmed);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    }

    void load();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  function handleImageLoad() {
    const img = imageRef.current;
    if (!img) return;
    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    setDisplayWidth(img.clientWidth);
  }

  // The canvas is fixed at 16:9 and flexes with the sidebar/viewport width
  // (see .calibration-canvas in CalibrationPanel.css), so its rendered
  // size can change after the image has already loaded (window resize,
  // sidebar content changing height, etc.) - without this, drag math using
  // a stale `displayWidth` would drift out of sync with where the image is
  // actually drawn.
  useEffect(() => {
    const img = imageRef.current;
    if (!img) return;
    const observer = new ResizeObserver(() => setDisplayWidth(img.clientWidth));
    observer.observe(img);
    return () => observer.disconnect();
  }, [frameUrl]);

  // Redraws the enhancement overlay whenever the colour/strength picked in
  // the "Preview" card changes, or the frame/display size
  // does. Runs at display resolution rather than the source frame's native
  // resolution - this is a viewing aid, not something anything else reads
  // pixels back out of, so there's nothing to gain from processing every
  // last native pixel, only slower re-renders while dragging the slider.
  useEffect(() => {
    const canvas = enhanceCanvasRef.current;
    if (!canvas || !frameUrl || !naturalSize) return;
    if (enhanceStrength <= 0) {
      setEnhancedDataUrl(null);
      return;
    }

    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const w = Math.max(1, Math.round(displayWidth || naturalSize.width));
      const h = Math.max(1, Math.round((w * naturalSize.height) / naturalSize.width));
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(img, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);
      const data = imageData.data;
      const [tr, tg, tb] = LINE_COLOR_OPTIONS.find((c) => c.value === lineColor)?.rgb ?? [50, 200, 60];
      const [th, , tl] = rgbToHsl(tr, tg, tb);
      // White/black have no meaningful hue to match against - fall back to
      // matching by lightness instead for those two.
      const isAchromatic = Math.max(tr, tg, tb) - Math.min(tr, tg, tb) < 20;
      const strength = enhanceStrength / 100;

      // Pass 1: how much each pixel looks like the selected colour, on its
      // own - a thin court line is often only a couple of pixels wide once
      // anti-aliased/motion-blurred, so before growing anything this is
      // still a thin, easy-to-miss strip.
      const similarity = new Float32Array(w * h);
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        const [h2, , l2] = rgbToHsl(data[i], data[i + 1], data[i + 2]);
        if (isAchromatic) {
          similarity[p] = Math.max(0, 1 - Math.abs(l2 - tl) / ENHANCE_LIGHTNESS_THRESHOLD);
        } else {
          let hueDistance = Math.abs(h2 - th);
          if (hueDistance > 0.5) hueDistance = 1 - hueDistance;
          // Square-rooted so it stays high for longer before tapering off
          // near the edge of the threshold, rather than a straight linear
          // ramp - a pixel that's "pretty close" to the target hue still
          // gets a solid boost instead of needing to be a near-exact match.
          similarity[p] = Math.max(0, 1 - hueDistance / ENHANCE_HUE_THRESHOLD) ** 0.6;
        }
      }

      // Pass 2: grow that thin strip - each pixel takes on the strongest
      // match found within `radius` of it, so the highlighted area reads
      // as noticeably thicker/larger than the actual line rather than
      // just brighter in place. Radius scales with the slider alongside
      // the boost itself.
      const radius = Math.round(strength * 10);
      const dilated = radius > 0 ? dilateMax(similarity, w, h, radius) : similarity;

      // Pass 3: boost only - a pixel keeps its own hue (never recoloured
      // to "another colour") and anything the dilated match didn't reach
      // is left completely untouched, rather than desaturating the rest
      // of the image toward gray.
      for (let i = 0, p = 0; i < data.length; i += 4, p++) {
        const match = dilated[p];
        if (match <= 0) continue;

        const [h2, s2, l2] = rgbToHsl(data[i], data[i + 1], data[i + 2]);
        const boost = match * strength;
        const newS = isAchromatic ? s2 : s2 + (1 - s2) * boost;
        const newL = isAchromatic ? l2 + (tl - l2) * boost : l2;

        const [nr, ng, nb] = hslToRgb(h2, Math.min(1, Math.max(0, newS)), Math.min(1, Math.max(0, newL)));
        data[i] = nr;
        data[i + 1] = ng;
        data[i + 2] = nb;
      }
      ctx.putImageData(imageData, 0, 0);
      setEnhancedDataUrl(canvas.toDataURL("image/jpeg", 0.85));
    };
    img.src = frameUrl;

    return () => {
      cancelled = true;
    };
  }, [frameUrl, naturalSize, displayWidth, lineColor, enhanceStrength]);

  function scale(): number {
    return naturalSize && naturalSize.width > 0 ? displayWidth / naturalSize.width : 1;
  }

  function handleMouseMove(event: MouseEvent<HTMLDivElement>) {
    if (dragIndex === null || !naturalSize || !points || !imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const s = scale();
    const x = clampToFrame((event.clientX - rect.left) / s, naturalSize.width);
    const y = clampToFrame((event.clientY - rect.top) / s, naturalSize.height);

    setPoints((prev) => prev!.map((p, i) => (i === dragIndex ? { x, y } : p)));
  }

  function handleReset() {
    if (!naturalSize) return;
    setPoints(presetPoints(naturalSize.width, naturalSize.height));
  }

  async function handleSave(): Promise<boolean> {
    if (!points) return false;
    setSaving(true);
    setError(null);
    try {
      const result = await api.setCalibrationPoints(job.id, points[0], points[1], points[2], points[3], netHeightM);
      setConfirmed(result.confirmed);
      return true;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return false;
    } finally {
      setSaving(false);
    }
  }

  async function handleRedoConfirm() {
    setRedoing(true);
    setError(null);
    try {
      const result = await api.setCalibrationConfirmed(job.id, false);
      setConfirmed(result.confirmed);
      setRedoDialogOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRedoing(false);
    }
  }

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!frameUrl || !points) return <LoadingSpinner />;

  const s = scale();
  const locked = confirmed;

  // The magnifier's reference picture and coordinate space switch
  // together: the enhanced preview is drawn at display resolution, so
  // points (always tracked in native-pixel coordinates) need scaling by
  // `s` to line up with it, whereas the raw native frame needs no scaling
  // at all.
  const usingEnhancedMagnifier = enhanceStrength > 0 && !!enhancedDataUrl;
  const magnifierSource = usingEnhancedMagnifier ? enhancedDataUrl : frameUrl;
  const magnifierCoordScale = usingEnhancedMagnifier ? s : 1;
  const magnifierRefWidth = usingEnhancedMagnifier
    ? Math.max(1, Math.round(displayWidth || (naturalSize?.width ?? 1)))
    : (naturalSize?.width ?? 1);
  const magnifierRefHeight =
    usingEnhancedMagnifier && naturalSize
      ? Math.max(1, Math.round((magnifierRefWidth * naturalSize.height) / naturalSize.width))
      : (naturalSize?.height ?? 1);

  // Recomputed on every render (every drag frame) so the near baseline and
  // both attack lines redraw live as the 4 marked points move - null only
  // for a genuinely degenerate placement (e.g. two points on top of each
  // other), in which case the preview is just skipped for that frame.
  let predicted: PredictedCourtGeometry | null = null;
  try {
    predicted = predictCourtGeometry(points[0], points[1], points[2], points[3]);
  } catch {
    predicted = null;
  }

  // Clearance reserved around the canvas so the drag-overflow boundary
  // (and a handle sitting right on it, which has its own 13px radius
  // beyond that) has room to render without spilling into the sidebar -
  // reserved as padding on the OUTER box (shrinking how wide the canvas
  // itself is allowed to be) rather than as margin on the canvas (which
  // would add on top of its own 100%-width and overflow past the box it's
  // supposed to stay inside of). Computed from the actual rendered size
  // rather than a guessed constant, since that guess previously undershot
  // for some video dimensions.
  const HANDLE_RADIUS = 13;
  const overflowX = naturalSize ? naturalSize.width * OVERFLOW_FRACTION * s + HANDLE_RADIUS + 20 : 20;
  const overflowY = naturalSize ? naturalSize.height * OVERFLOW_FRACTION * s + HANDLE_RADIUS + 20 : 20;

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 2, flexShrink: 0 }}>
        Court calibration
      </Typography>

      <Stack direction={{ xs: "column", md: "row" }} spacing={2.5} sx={{ flex: 1, minHeight: 0 }}>
        <Box
          sx={{ flex: 1, minWidth: 0, height: "100%", position: "relative" }}
          style={{ padding: `${overflowY}px ${overflowX}px` }}
        >
          <LockOverlay
            active={locked}
            label="Redo Court Identification to make changes"
            onClick={() => setRedoDialogOpen(true)}
          />
          <div
            className="calibration-canvas"
            onMouseMove={handleMouseMove}
            onMouseUp={() => setDragIndex(null)}
            onMouseLeave={() => setDragIndex(null)}
          >
            <img ref={imageRef} src={frameUrl} alt="Calibration frame" draggable={false} onLoad={handleImageLoad} />

            {enhanceStrength > 0 && (
              <canvas
                ref={enhanceCanvasRef}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  pointerEvents: "none",
                  borderRadius: 10,
                }}
              />
            )}

            {pointsVisible && naturalSize && (
              <div
                className="calibration-overflow-bound"
                style={{
                  left: -naturalSize.width * OVERFLOW_FRACTION * s,
                  top: -naturalSize.height * OVERFLOW_FRACTION * s,
                  width: naturalSize.width * (1 + 2 * OVERFLOW_FRACTION) * s,
                  height: naturalSize.height * (1 + 2 * OVERFLOW_FRACTION) * s,
                }}
              />
            )}

            {pointsVisible && naturalSize && (
              <svg
                className="calibration-overlay"
                // Sized and offset to match .calibration-overflow-bound
                // exactly (rather than just the image itself) so a
                // predicted line that legitimately lands inside the
                // allowed drag range - not just on the image proper - is
                // still visible instead of getting clipped at the image's
                // own edge.
                style={{
                  left: -naturalSize.width * OVERFLOW_FRACTION * s,
                  top: -naturalSize.height * OVERFLOW_FRACTION * s,
                  width: naturalSize.width * (1 + 2 * OVERFLOW_FRACTION) * s,
                  height: naturalSize.height * (1 + 2 * OVERFLOW_FRACTION) * s,
                }}
                viewBox={`${-naturalSize.width * OVERFLOW_FRACTION} ${-naturalSize.height * OVERFLOW_FRACTION} ${
                  naturalSize.width * (1 + 2 * OVERFLOW_FRACTION)
                } ${naturalSize.height * (1 + 2 * OVERFLOW_FRACTION)}`}
                preserveAspectRatio="none"
              >
                {/* Marked: net line + far baseline, and the sidelines joining them into the visible half-court */}
                <polygon
                  points={[points[0], points[1], points[3], points[2]].map((p) => `${p.x},${p.y}`).join(" ")}
                  className="court-outline"
                />
                <line x1={points[0].x} y1={points[0].y} x2={points[1].x} y2={points[1].y} className="net-line" />

                {/* Predicted: near baseline + both attack lines */}
                {predicted && (
                  <>
                    <line
                      x1={predicted.near_left.x}
                      y1={predicted.near_left.y}
                      x2={predicted.near_right.x}
                      y2={predicted.near_right.y}
                      className="predicted-outline"
                    />
                    <line
                      x1={predicted.attack_near_left.x}
                      y1={predicted.attack_near_left.y}
                      x2={predicted.attack_near_right.x}
                      y2={predicted.attack_near_right.y}
                      className="predicted-outline"
                    />
                    <line
                      x1={predicted.attack_far_left.x}
                      y1={predicted.attack_far_left.y}
                      x2={predicted.attack_far_right.x}
                      y2={predicted.attack_far_right.y}
                      className="predicted-outline"
                    />
                    <line
                      x1={points[2].x}
                      y1={points[2].y}
                      x2={predicted.near_left.x}
                      y2={predicted.near_left.y}
                      className="predicted-outline"
                    />
                    <line
                      x1={points[3].x}
                      y1={points[3].y}
                      x2={predicted.near_right.x}
                      y2={predicted.near_right.y}
                      className="predicted-outline"
                    />
                  </>
                )}
              </svg>
            )}

            {pointsVisible &&
              naturalSize &&
              points.map((point, i) => (
                <div key={i}>
                  <div
                    className={`calibration-handle${dragIndex === i ? " dragging" : ""}`}
                    style={{ left: point.x * s, top: point.y * s }}
                    onMouseDown={(event) => {
                      event.preventDefault();
                      setDragIndex(i);
                    }}
                  />
                  <div className="calibration-handle-label" style={{ left: point.x * s + 13, top: point.y * s }}>
                    {POINT_TITLES[i]}
                  </div>
                </div>
              ))}

            {pointsVisible && dragIndex !== null && naturalSize && (
              <div
                className="calibration-magnifier"
                style={{
                  backgroundImage: `url(${magnifierSource})`,
                  backgroundSize: `${magnifierRefWidth * MAGNIFIER_ZOOM}px ${magnifierRefHeight * MAGNIFIER_ZOOM}px`,
                  backgroundPosition:
                    `${-(points[dragIndex].x * magnifierCoordScale * MAGNIFIER_ZOOM - MAGNIFIER_SIZE / 2)}px ` +
                    `${-(points[dragIndex].y * magnifierCoordScale * MAGNIFIER_ZOOM - MAGNIFIER_SIZE / 2)}px`,
                }}
              >
                <span className="magnifier-cross-v" />
                <span className="magnifier-cross-h" />
              </div>
            )}
          </div>
        </Box>

        <Stack spacing={2} sx={{ width: 550, flexShrink: 0, height: "100%", overflowY: "auto", pr: 0.5 }}>
          <CourtDiagramExample />

          <Card variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Preview
            </Typography>
            <Stack spacing={1.5}>
              <FormControlLabel
                sx={{ ml: 0, justifyContent: "space-between" }}
                labelPlacement="start"
                control={
                  <Switch
                    size="small"
                    checked={pointsVisible}
                    onChange={(e) => setPointsVisible(e.target.checked)}
                  />
                }
                label={
                  <Typography variant="body2" color="text.secondary">
                    Show points
                  </Typography>
                }
              />
              <FormControl size="small" fullWidth>
                <InputLabel id="line-color-label">Court line colour</InputLabel>
                <Select
                  labelId="line-color-label"
                  label="Court line colour"
                  value={lineColor}
                  onChange={(e: SelectChangeEvent<string>) => setLineColor(e.target.value)}
                >
                  {LINE_COLOR_OPTIONS.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Enhance strength
                </Typography>
                <Slider
                  size="small"
                  value={enhanceStrength}
                  onChange={(_, value) => setEnhanceStrength(value as number)}
                  min={0}
                  max={100}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Stack>
          </Card>

          <Card variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Legend
            </Typography>
            <Stack spacing={1}>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <LegendSwatch color="#aa3bff" />
                <Typography variant="body2" color="text.secondary">
                  marked points
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <LegendLine color="#38bdf8" />
                <Typography variant="body2" color="text.secondary">
                  net line
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <LegendLine color="#ffd400" />
                <Typography variant="body2" color="text.secondary">
                  far baseline
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <LegendLine color="rgba(0,235,160,0.85)" dashed />
                <Typography variant="body2" color="text.secondary">
                  predicted lines
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <LegendLine color="rgba(255,255,255,0.7)" dashed />
                <Typography variant="body2" color="text.secondary">
                  allowed drag range
                </Typography>
              </Stack>
            </Stack>
          </Card>

          <Card variant="outlined" sx={{ p: 2, position: "relative" }}>
            <LockOverlay
              active={locked}
              label="Redo Court Identification to make changes"
              onClick={() => setRedoDialogOpen(true)}
            />
            <Stack spacing={1.5}>
              <FormControl size="small" fullWidth disabled={locked}>
                <InputLabel id="net-height-label">Net height</InputLabel>
                <Select
                  labelId="net-height-label"
                  label="Net height"
                  value={netHeightM}
                  onChange={(e: SelectChangeEvent<number>) => setNetHeightM(Number(e.target.value))}
                >
                  {NET_HEIGHT_OPTIONS.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Button variant="outlined" color="inherit" fullWidth disabled={locked} onClick={handleReset}>
                Reset
              </Button>
            </Stack>
          </Card>

          {/* The one action button on this screen: while editing, it saves
              the calibration; once locked, it reflects that same
              confirmed/locked state as Scoring and Player Identification
              do - its job switches to opening the redo dialog rather than
              acting directly. */}
          <Button
            variant="contained"
            color={locked ? "warning" : "primary"}
            fullWidth
            disabled={saving}
            onClick={() => (locked ? setRedoDialogOpen(true) : void handleSave())}
          >
            {saving ? "Saving..." : locked ? "Redo Court Identification" : "Set Court Identification"}
          </Button>
        </Stack>
      </Stack>

      <Dialog open={redoDialogOpen} onClose={() => setRedoDialogOpen(false)}>
        <DialogTitle>Redo court identification?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This won't clear the saved calibration - it just unlocks the points for editing again
            until you set it once more.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRedoDialogOpen(false)} disabled={redoing}>
            Cancel
          </Button>
          <Button color="warning" variant="contained" disabled={redoing} onClick={() => void handleRedoConfirm()}>
            {redoing ? "Confirming..." : "Confirm"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
