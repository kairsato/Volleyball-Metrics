import { useEffect, useRef, useState } from "react";
import type { MouseEvent } from "react";
import "./CalibrationPanel.css";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import type { Job, Point } from "../lib/types";
import { LoadingSpinner } from "./LoadingSpinner";

function LegendSwatch({ color }: { color: string }) {
  return (
    <Box
      sx={{
        width: 14,
        height: 14,
        borderRadius: "50%",
        bgcolor: color,
        border: "2px solid rgba(255,255,255,0.85)",
        boxShadow: "0 0 0 1px rgba(0,0,0,0.2)",
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

interface CalibrationPanelProps {
  job: Job;
  onCalibrated: () => void;
}

const POINT_LABELS = ["1", "2", "3", "4", "5", "6"];

// Mirrors the desktop tool's reset_points() preset ratios.
const PRESET_INSET_X = 0.15;
const PRESET_INSET_Y = 0.2;
const NET_PRESET_HEIGHT_PX = 120;

// How far past the frame edge a marker is allowed to go, as a fraction of
// that dimension - lets a corner just out of frame still be marked.
const OVERFLOW_FRACTION = 0.08;

// Loupe shown while dragging, zoomed into the source frame's native pixels
// (not the scaled-down display) so fine alignment is actually possible.
const MAGNIFIER_SIZE = 150;
const MAGNIFIER_ZOOM = 3;

function clampToFrame(value: number, dimension: number): number {
  const allowance = dimension * OVERFLOW_FRACTION;
  return Math.max(-allowance, Math.min(dimension + allowance, value));
}

function presetPoints(width: number, height: number): Point[] {
  const left = width * PRESET_INSET_X;
  const right = width * (1 - PRESET_INSET_X);
  const top = height * PRESET_INSET_Y;
  const bottom = height * (1 - PRESET_INSET_Y);

  const corners: Point[] = [
    { x: left, y: top },
    { x: right, y: top },
    { x: right, y: bottom },
    { x: left, y: bottom },
  ];

  const farMid = {
    x: (corners[0].x + corners[1].x) / 2,
    y: clampToFrame((corners[0].y + corners[1].y) / 2 - NET_PRESET_HEIGHT_PX, height),
  };
  const nearMid = {
    x: (corners[3].x + corners[2].x) / 2,
    y: clampToFrame((corners[3].y + corners[2].y) / 2 - NET_PRESET_HEIGHT_PX, height),
  };

  return [...corners, farMid, nearMid];
}

export function CalibrationPanel({ job, onCalibrated }: CalibrationPanelProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [points, setPoints] = useState<Point[] | null>(null);
  const [calibrated, setCalibrated] = useState(false);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);

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
        setPoints([...pointsRes.corners, ...pointsRes.net_points]);
        setCalibrated(pointsRes.calibrated);
        if (pointsRes.calibrated) onCalibrated();
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

  async function handleSave() {
    if (!points) return;
    setSaving(true);
    setError(null);
    try {
      await api.setCalibrationPoints(job.id, points.slice(0, 4), points.slice(4, 6));
      setCalibrated(true);
      onCalibrated();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!frameUrl || !points) return <LoadingSpinner />;

  const s = scale();
  const corners = points.slice(0, 4);
  const net = points.slice(4, 6);

  // Exact clearance the canvas needs on each side so the drag-overflow
  // boundary (and a handle sitting right on it, which has its own 13px
  // radius beyond that) never runs into whatever sits next to the canvas -
  // computed from the actual rendered size rather than a guessed constant,
  // since that guess previously undershot for some video dimensions.
  const HANDLE_RADIUS = 13;
  const overflowX = naturalSize ? naturalSize.width * OVERFLOW_FRACTION * s + HANDLE_RADIUS : 0;
  const overflowY = naturalSize ? naturalSize.height * OVERFLOW_FRACTION * s + HANDLE_RADIUS : 0;
  const canvasSpacing = { left: overflowX + 20, right: overflowX + 20, top: overflowY + 20, bottom: overflowY + 20 };

  return (
    <Box sx={{ mb: 2.5 }}>
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
        Court calibration
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 2 }}>
        Drag each marker onto the spot it's labelled for below. This tells the tracker where the
        court boundary and net sit in the frame, so it can tell who's in play and how fast the
        ball is moving. Markers are semi-transparent so you can still see what's underneath them,
        and can be dragged a little past the frame edge if the real spot is just out of view.
      </Typography>

      <div
        className="calibration-canvas"
        style={{
          marginLeft: canvasSpacing.left,
          marginRight: canvasSpacing.right,
          marginTop: canvasSpacing.top,
          marginBottom: canvasSpacing.bottom,
        }}
        onMouseMove={handleMouseMove}
        onMouseUp={() => setDragIndex(null)}
        onMouseLeave={() => setDragIndex(null)}
      >
        <img
          ref={imageRef}
          src={frameUrl}
          alt="Calibration frame"
          draggable={false}
          onLoad={handleImageLoad}
        />

        {naturalSize && (
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

        {naturalSize && (
          <svg
            className="calibration-overlay"
            viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
            preserveAspectRatio="none"
          >
            <polygon
              points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
              className="court-outline"
            />
            <line x1={net[0].x} y1={net[0].y} x2={net[1].x} y2={net[1].y} className="net-line" />
          </svg>
        )}

        {naturalSize &&
          points.map((point, i) => (
            <div
              key={i}
              className={`calibration-handle${i >= 4 ? " net" : ""}${dragIndex === i ? " dragging" : ""}`}
              style={{ left: point.x * s, top: point.y * s }}
              onMouseDown={(event) => {
                event.preventDefault();
                setDragIndex(i);
              }}
            >
              {POINT_LABELS[i]}
            </div>
          ))}

        {dragIndex !== null && naturalSize && (
          <div
            className="calibration-magnifier"
            style={{
              backgroundImage: `url(${frameUrl})`,
              backgroundSize: `${naturalSize.width * MAGNIFIER_ZOOM}px ${naturalSize.height * MAGNIFIER_ZOOM}px`,
              backgroundPosition:
                `${-(points[dragIndex].x * MAGNIFIER_ZOOM - MAGNIFIER_SIZE / 2)}px ` +
                `${-(points[dragIndex].y * MAGNIFIER_ZOOM - MAGNIFIER_SIZE / 2)}px`,
            }}
          >
            <span className="magnifier-cross-v" />
            <span className="magnifier-cross-h" />
          </div>
        )}
      </div>

      <Stack direction="row" spacing={2.5} sx={{ mt: 2.5, fontSize: 13, flexWrap: "wrap" }}>
        <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
          <LegendSwatch color="color-mix(in srgb, #aa3bff 70%, transparent)" />
          <Typography variant="body2" color="text.secondary">
            1-4 court corners
          </Typography>
        </Stack>
        <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
          <LegendSwatch color="color-mix(in srgb, #38bdf8 70%, transparent)" />
          <Typography variant="body2" color="text.secondary">
            5-6 net / antenna top
          </Typography>
        </Stack>
        <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
          <LegendLine color="#ffd400" />
          <Typography variant="body2" color="text.secondary">
            court boundary
          </Typography>
        </Stack>
        <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
          <LegendLine color="#38bdf8" />
          <Typography variant="body2" color="text.secondary">
            net line
          </Typography>
        </Stack>
        <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
          <LegendLine color="rgba(255,255,255,0.7)" dashed />
          <Typography variant="body2" color="text.secondary">
            allowed drag range
          </Typography>
        </Stack>
      </Stack>

      <Stack direction="row" spacing={1.5} sx={{ mt: 2 }}>
        <Button variant="outlined" color="inherit" onClick={handleReset}>
          Reset
        </Button>
        <Button variant="contained" disabled={saving} onClick={handleSave}>
          {saving ? "Saving..." : calibrated ? "Update calibration" : "Save calibration"}
        </Button>
      </Stack>

      {calibrated && (
        <Typography variant="body2" sx={{ color: "success.main", mt: 1.5 }}>
          Calibration saved.
        </Typography>
      )}
    </Box>
  );
}
