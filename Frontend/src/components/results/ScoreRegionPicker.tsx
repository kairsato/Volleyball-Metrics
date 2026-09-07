import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Skeleton from "@mui/material/Skeleton";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import ToggleButton from "@mui/material/ToggleButton";
import ToggleButtonGroup from "@mui/material/ToggleButtonGroup";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import PanToolIcon from "@mui/icons-material/PanTool";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import { api } from "../../lib/api";
import type { OcrRegion, ScoreRegionTestResult } from "../../lib/types";
import { formatTimestamp } from "./types";

interface ScoreRegionPickerProps {
  jobId: string;
  minTimeS?: number;
  maxTimeS: number;
  region: OcrRegion | null;
  onRegionChange: (region: OcrRegion) => void;
  minConfidence: number;
  onMinConfidenceChange: (value: number) => void;
}

type Point = { x: number; y: number };
type Corner = "nw" | "ne" | "sw" | "se";
type Tool = "hand" | "zoom";

// A brand-new box being dragged out from scratch - the only case that
// needs its own local draft, since (unlike moving/resizing an existing
// box) there's no box at all until the drag ends up at least MIN_BOX_SIZE
// in both dimensions.
interface DrawDraft {
  start: Point;
  current: Point;
}

type ActiveDrag = { kind: "move"; origin: Point; startBox: OcrRegion } | { kind: "resize"; corner: Corner; startBox: OcrRegion };

const MIN_BOX_SIZE = 4;
const HANDLE_SIZE_PX = 12;
const ZOOM_MIN = 1;
const ZOOM_MAX = 4;
const ZOOM_STEP = 0.5;

// Magnifier preview shown in the top-left corner: a zoomed-in crop of
// whatever box is currently selected (mid-drag, or the already-saved
// region), so fine scoreboard digits are readable while drawing/checking
// the box rather than only at the frame's native resolution. Distinct from
// the cursor-following loupe below, which previews wherever the mouse
// currently is rather than the box itself.
const MAGNIFIER_MAX_WIDTH_PX = 220;
const MAGNIFIER_MAX_HEIGHT_PX = 140;
const MAGNIFIER_MAX_SCALE = 6;

// The zoom tool's loupe: a fixed-size natural-pixel window centered on the
// cursor, magnified into a small floating canvas that follows the mouse -
// lets a user see exactly what they're about to zoom into before clicking.
const LOUPE_SIZE_PX = 160;
const LOUPE_CROP_NATURAL_PX = 50;

const RESIZE_CURSORS: Record<Corner, string> = {
  nw: "nwse-resize",
  se: "nwse-resize",
  ne: "nesw-resize",
  sw: "nesw-resize",
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function pointInBox(point: Point, box: OcrRegion): boolean {
  return point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y && point.y <= box.y + box.height;
}

// Resizes from whichever corner is being dragged, anchored on the
// opposite corner (dragging the top-left handle keeps the bottom-right
// corner fixed, etc.) - clamped to the frame's own bounds and a minimum
// size so the box can't be dragged inside-out or off-screen.
function resizeBox(startBox: OcrRegion, corner: Corner, point: Point, naturalSize: { width: number; height: number } | null): OcrRegion {
  const maxX = naturalSize?.width ?? Infinity;
  const maxY = naturalSize?.height ?? Infinity;
  const anchorX = corner.includes("w") ? startBox.x + startBox.width : startBox.x;
  const anchorY = corner.includes("n") ? startBox.y + startBox.height : startBox.y;
  const px = clamp(point.x, 0, maxX);
  const py = clamp(point.y, 0, maxY);

  return {
    x: Math.min(anchorX, px),
    y: Math.min(anchorY, py),
    width: Math.max(MIN_BOX_SIZE, Math.abs(anchorX - px)),
    height: Math.max(MIN_BOX_SIZE, Math.abs(anchorY - py)),
  };
}

function confidenceColor(confidence: number | null): string {
  if (confidence === null) return "text.secondary";
  if (confidence >= 0.7) return "success.main";
  if (confidence >= 0.4) return "warning.main";
  return "error.main";
}

// One side's read digit (or a dash if nothing was read) plus the model's
// own confidence in it - color-coded so a low-confidence read (which can
// still be the WRONG digit, not just a missing one) stands out at a glance.
function ReadResult({ label, value, confidence }: { label: string; value: number | null; confidence: number | null }): ReactNode {
  return (
    <Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
        {label}
      </Typography>
      <Typography sx={{ fontWeight: 700, color: confidenceColor(confidence) }}>
        {value ?? "—"}
        {confidence !== null && (
          <Typography component="span" variant="caption" sx={{ ml: 0.75, fontWeight: 400 }}>
            {Math.round(confidence * 100)}% confident
          </Typography>
        )}
      </Typography>
    </Box>
  );
}

// Draw-to-select a first box from scratch; once one exists, drag its body
// to move it or a corner handle to resize it - same drag-existing-handles
// scheme as CalibrationPanel, just for one rectangle instead of court
// points. Clicking/dragging anywhere outside the current box still starts
// a fresh one, replacing it entirely. All of that only applies in the
// "hand" tool - switching to the "zoom" tool disables box editing entirely
// so clicks there only ever zoom.
export function ScoreRegionPicker({
  jobId,
  minTimeS = 0,
  maxTimeS,
  region,
  onRegionChange,
  minConfidence,
  onMinConfidenceChange,
}: ScoreRegionPickerProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [time, setTime] = useState(minTimeS);
  const [loadingFrame, setLoadingFrame] = useState(false);
  const [tool, setTool] = useState<Tool>("hand");
  const [zoom, setZoom] = useState(1);
  const [loupePoint, setLoupePoint] = useState<Point | null>(null);
  const [loupeScreen, setLoupeScreen] = useState<{ x: number; y: number } | null>(null);
  const [drawDraft, setDrawDraft] = useState<DrawDraft | null>(null);
  const [activeDrag, setActiveDrag] = useState<ActiveDrag | null>(null);
  const [testing, setTesting] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<ScoreRegionTestResult | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const magnifierCanvasRef = useRef<HTMLCanvasElement>(null);
  const loupeCanvasRef = useRef<HTMLCanvasElement>(null);

  // Any of these invalidate a previous test read - clearing it here (rather
  // than only when a new one arrives) means a stale "94% confident" result
  // never sits on screen next to a box/frame/threshold it no longer describes.
  useEffect(() => {
    setTestResult(null);
    setTestError(null);
  }, [jobId, time, region, minConfidence]);

  async function handleTest() {
    if (!region) return;
    setTesting(true);
    setTestError(null);
    try {
      const result = await api.testScoreRegion(jobId, time, region, minConfidence);
      setTestResult(result);
    } catch (err) {
      setTestError(err instanceof Error ? err.message : String(err));
    } finally {
      setTesting(false);
    }
  }

  async function loadFrame(timeS: number) {
    setLoadingFrame(true);
    try {
      const url = await api.getScoreFrame(jobId, timeS);
      setFrameUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    } finally {
      setLoadingFrame(false);
    }
  }

  useEffect(() => {
    setTime(minTimeS);
    void loadFrame(minTimeS);
    return () => setFrameUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    // Only jobId/minTimeS should reset the scrub position - maxTimeS
    // changing alone (e.g. the range's "to" field being edited) shouldn't
    // yank the frame back to the start of the range.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, minTimeS]);

  // The unzoomed scale (natural pixels -> pre-transform CSS pixels) - the
  // zoom-scaled image and every overlay (highlight box, handles) all live
  // inside one inner Box that gets `transform: scale(zoom)` together, so
  // this stays constant across zoom levels (CSS transforms never affect
  // clientWidth) and only the shared transform does the visual scaling.
  function scale(): number {
    return naturalSize && naturalSize.width > 0 ? displayWidth / naturalSize.width : 1;
  }

  // Reads the image's actual on-screen rect (which already reflects both
  // the zoom transform and any scrolling) rather than combining cached
  // scale/zoom state, so pointer math stays correct regardless of how the
  // zoomed image is currently laid out.
  function toNatural(event: ReactMouseEvent): Point | null {
    if (!imageRef.current || !naturalSize) return null;
    const rect = imageRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const x = clamp(((event.clientX - rect.left) / rect.width) * naturalSize.width, 0, naturalSize.width);
    const y = clamp(((event.clientY - rect.top) / rect.height) * naturalSize.height, 0, naturalSize.height);
    return { x, y };
  }

  function handleImageLoad() {
    const img = imageRef.current;
    if (!img) return;
    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    setDisplayWidth(img.clientWidth);
  }

  // Zooms in/out by one step, keeping whatever natural-pixel point was
  // under the cursor at the same on-screen position - done by scrolling
  // the viewport to compensate after the scale changes, rather than
  // fiddling with transform-origin (simpler to reason about, and works
  // the same regardless of current scroll position).
  function zoomAtPoint(delta: number, event: ReactMouseEvent) {
    const point = toNatural(event);
    const nextZoom = Math.round(clamp(zoom + delta, ZOOM_MIN, ZOOM_MAX) * 10) / 10;
    const container = scrollRef.current;
    const containerRect = container?.getBoundingClientRect();
    const baseScale = scale();
    setZoom(nextZoom);
    if (!point || !container || !containerRect) return;

    if (nextZoom <= ZOOM_MIN) {
      requestAnimationFrame(() => {
        if (scrollRef.current) {
          scrollRef.current.scrollLeft = 0;
          scrollRef.current.scrollTop = 0;
        }
      });
      return;
    }

    const viewportOffsetX = event.clientX - containerRect.left;
    const viewportOffsetY = event.clientY - containerRect.top;
    requestAnimationFrame(() => {
      if (!scrollRef.current) return;
      scrollRef.current.scrollLeft = point.x * baseScale * nextZoom - viewportOffsetX;
      scrollRef.current.scrollTop = point.y * baseScale * nextZoom - viewportOffsetY;
    });
  }

  function handleMouseDown(event: ReactMouseEvent) {
    if (tool !== "hand") return;
    const point = toNatural(event);
    if (!point) return;
    if (region && pointInBox(point, region)) {
      setActiveDrag({ kind: "move", origin: point, startBox: region });
    } else {
      setDrawDraft({ start: point, current: point });
    }
  }

  function handleMouseMove(event: ReactMouseEvent) {
    if (tool === "zoom") {
      const point = toNatural(event);
      setLoupePoint(point);
      setLoupeScreen(point ? { x: event.clientX, y: event.clientY } : null);
      return;
    }

    if (!drawDraft && !activeDrag) return;
    const point = toNatural(event);
    if (!point) return;

    if (drawDraft) {
      setDrawDraft({ start: drawDraft.start, current: point });
      return;
    }

    if (activeDrag!.kind === "move") {
      const { origin, startBox } = activeDrag!;
      const maxX = (naturalSize?.width ?? startBox.width) - startBox.width;
      const maxY = (naturalSize?.height ?? startBox.height) - startBox.height;
      onRegionChange({
        x: clamp(startBox.x + (point.x - origin.x), 0, Math.max(0, maxX)),
        y: clamp(startBox.y + (point.y - origin.y), 0, Math.max(0, maxY)),
        width: startBox.width,
        height: startBox.height,
      });
    } else {
      onRegionChange(resizeBox(activeDrag!.startBox, activeDrag!.corner, point, naturalSize));
    }
  }

  function endDrag() {
    if (drawDraft) {
      const x = Math.min(drawDraft.start.x, drawDraft.current.x);
      const y = Math.min(drawDraft.start.y, drawDraft.current.y);
      const width = Math.abs(drawDraft.current.x - drawDraft.start.x);
      const height = Math.abs(drawDraft.current.y - drawDraft.start.y);
      setDrawDraft(null);
      if (width >= MIN_BOX_SIZE && height >= MIN_BOX_SIZE) onRegionChange({ x, y, width, height });
    }
    setActiveDrag(null);
  }

  function handleMouseLeave() {
    endDrag();
    setLoupePoint(null);
    setLoupeScreen(null);
  }

  function handleClick(event: ReactMouseEvent) {
    if (tool !== "zoom") return;
    zoomAtPoint(ZOOM_STEP, event);
  }

  function handleContextMenu(event: ReactMouseEvent) {
    if (tool !== "zoom") return;
    event.preventDefault();
    zoomAtPoint(-ZOOM_STEP, event);
  }

  function startResize(corner: Corner) {
    return (event: ReactMouseEvent) => {
      if (tool !== "hand" || !region) return;
      event.stopPropagation();
      setActiveDrag({ kind: "resize", corner, startBox: region });
    };
  }

  const s = scale();
  // Memoized so the magnifier effect below can depend on it directly - it
  // only changes identity when the drag or the saved region actually
  // changes, not on every unrelated render.
  const liveBox = useMemo(
    () =>
      drawDraft
        ? {
            x: Math.min(drawDraft.start.x, drawDraft.current.x),
            y: Math.min(drawDraft.start.y, drawDraft.current.y),
            width: Math.abs(drawDraft.current.x - drawDraft.start.x),
            height: Math.abs(drawDraft.current.y - drawDraft.start.y),
          }
        : region,
    [drawDraft, region],
  );

  // Redraws the box magnifier's canvas from the source <img> (not the
  // downscaled/zoomed display box), cropped to liveBox and scaled up,
  // every time the selection moves/resizes - cheap enough to do on every
  // drag frame since the crop is small. imageSmoothingEnabled is off so
  // zoomed-in scoreboard digits stay crisp instead of blurring.
  useEffect(() => {
    const canvas = magnifierCanvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !liveBox || liveBox.width <= 0 || liveBox.height <= 0) return;

    const magScale = Math.min(
      MAGNIFIER_MAX_WIDTH_PX / liveBox.width,
      MAGNIFIER_MAX_HEIGHT_PX / liveBox.height,
      MAGNIFIER_MAX_SCALE,
    );
    const w = Math.max(1, Math.round(liveBox.width * magScale));
    const h = Math.max(1, Math.round(liveBox.height * magScale));
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, liveBox.x, liveBox.y, liveBox.width, liveBox.height, 0, 0, w, h);
  }, [liveBox, frameUrl]);

  // Redraws the cursor loupe the same way, cropped to a fixed natural-pixel
  // window centered on wherever the mouse currently is.
  useEffect(() => {
    const canvas = loupeCanvasRef.current;
    const img = imageRef.current;
    if (tool !== "zoom" || !canvas || !img || !loupePoint || !naturalSize) return;

    const half = LOUPE_CROP_NATURAL_PX / 2;
    const sx = clamp(loupePoint.x - half, 0, Math.max(0, naturalSize.width - LOUPE_CROP_NATURAL_PX));
    const sy = clamp(loupePoint.y - half, 0, Math.max(0, naturalSize.height - LOUPE_CROP_NATURAL_PX));

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, LOUPE_SIZE_PX, LOUPE_SIZE_PX);
    ctx.drawImage(img, sx, sy, LOUPE_CROP_NATURAL_PX, LOUPE_CROP_NATURAL_PX, 0, 0, LOUPE_SIZE_PX, LOUPE_SIZE_PX);
  }, [tool, loupePoint, frameUrl, naturalSize]);

  const containerCursor = tool === "zoom" ? "zoom-in" : activeDrag?.kind === "move" ? "grabbing" : region ? "grab" : "crosshair";
  const loupeLeft = loupeScreen ? Math.min(loupeScreen.x + 24, window.innerWidth - LOUPE_SIZE_PX - 8) : 0;
  const loupeTop = loupeScreen ? Math.min(loupeScreen.y + 24, window.innerHeight - LOUPE_SIZE_PX - 8) : 0;

  return (
    <Box>
      <Box sx={{ position: "relative" }}>
        <Box
          ref={scrollRef}
          sx={{
            position: "relative",
            maxWidth: "100%",
            maxHeight: "min(60vh, 640px)",
            overflow: zoom > 1 ? "auto" : "hidden",
            userSelect: "none",
            border: 1,
            borderColor: "divider",
            borderRadius: 1,
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={endDrag}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          onContextMenu={handleContextMenu}
        >
          {/* The spinner only replaces the image before the very first frame
              has ever loaded - once one exists, scrubbing keeps the current
              frame visible (just dimmed) while the next one loads instead of
              swapping it out for a fixed-height spinner box, which was
              causing the picture to flash away and the whole box to jump
              between the spinner's height and the image's actual height on
              every scrub. */}
          {!frameUrl ? (
            <Skeleton variant="rounded" sx={{ width: 640, maxWidth: "100%", aspectRatio: "16 / 9" }} />
          ) : (
            <Box
              sx={{
                position: "relative",
                display: "inline-block",
                lineHeight: 0,
                cursor: containerCursor,
                transform: `scale(${zoom})`,
                transformOrigin: "0 0",
              }}
            >
              <Box
                component="img"
                ref={imageRef}
                src={frameUrl}
                alt="Video frame"
                draggable={false}
                onLoad={handleImageLoad}
                sx={{
                  display: "block",
                  maxWidth: "100%",
                  height: "auto",
                  opacity: loadingFrame ? 0.5 : 1,
                  transition: "opacity 0.15s ease",
                }}
              />

              {liveBox && naturalSize && (
                <Box
                  sx={{
                    position: "absolute",
                    left: liveBox.x * s,
                    top: liveBox.y * s,
                    width: liveBox.width * s,
                    height: liveBox.height * s,
                    border: "2px solid #38bdf8",
                    bgcolor: "rgba(56,189,248,0.18)",
                    pointerEvents: "none",
                  }}
                />
              )}

              {region && naturalSize && !drawDraft && (
                <>
                  {(["nw", "ne", "sw", "se"] as const).map((corner) => (
                    <Box
                      key={corner}
                      onMouseDown={startResize(corner)}
                      sx={{
                        position: "absolute",
                        left: (corner.includes("w") ? region.x : region.x + region.width) * s,
                        top: (corner.includes("n") ? region.y : region.y + region.height) * s,
                        width: HANDLE_SIZE_PX,
                        height: HANDLE_SIZE_PX,
                        transform: "translate(-50%, -50%)",
                        borderRadius: "50%",
                        border: "2px solid #38bdf8",
                        bgcolor: "#fff",
                        cursor: tool === "hand" ? RESIZE_CURSORS[corner] : containerCursor,
                      }}
                    />
                  ))}
                </>
              )}
            </Box>
          )}
        </Box>

        {liveBox && liveBox.width > 0 && liveBox.height > 0 && (
          <Box
            sx={{
              position: "absolute",
              top: 8,
              left: 8,
              border: "2px solid #38bdf8",
              borderRadius: 1,
              overflow: "hidden",
              boxShadow: 4,
              bgcolor: "#000",
              lineHeight: 0,
              pointerEvents: "none",
            }}
          >
            <canvas ref={magnifierCanvasRef} style={{ display: "block" }} />
          </Box>
        )}

        {tool === "zoom" && loupePoint && loupeScreen && (
          <Box
            sx={{
              position: "fixed",
              left: loupeLeft,
              top: loupeTop,
              zIndex: 1500,
              pointerEvents: "none",
              border: "2px solid #38bdf8",
              borderRadius: 1,
              overflow: "hidden",
              boxShadow: 4,
              bgcolor: "#000",
              lineHeight: 0,
            }}
          >
            <canvas ref={loupeCanvasRef} width={LOUPE_SIZE_PX} height={LOUPE_SIZE_PX} style={{ display: "block" }} />
          </Box>
        )}
      </Box>

      <Stack direction="row" spacing={4} sx={{ alignItems: "flex-start", mt: 1.5, flexWrap: "wrap" }}>
        <Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
            Frame
          </Typography>
          <Stack direction="row" spacing={1.5} sx={{ alignItems: "center" }}>
            <Slider
              size="small"
              value={time}
              min={minTimeS}
              max={Math.max(minTimeS + 1, maxTimeS)}
              onChange={(_, value) => setTime(value as number)}
              onChangeCommitted={(_, value) => void loadFrame(value as number)}
              sx={{ width: 220 }}
            />
            <Typography variant="body2" color="text.secondary" sx={{ minWidth: 40 }}>
              {formatTimestamp(time)}
            </Typography>
          </Stack>
        </Box>

        <Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
            Tool
          </Typography>
          <ToggleButtonGroup size="small" value={tool} exclusive onChange={(_, value) => value && setTool(value)}>
            <Tooltip title="Move / resize the box">
              <ToggleButton value="hand" aria-label="Move and resize the box">
                <PanToolIcon fontSize="small" />
              </ToggleButton>
            </Tooltip>
            <Tooltip title="Left-click to zoom in, right-click to zoom out">
              <ToggleButton value="zoom" aria-label="Zoom in or out at the cursor">
                <ZoomInIcon fontSize="small" />
              </ToggleButton>
            </Tooltip>
          </ToggleButtonGroup>
        </Box>
      </Stack>

      <Box sx={{ mt: 3, p: 2, border: 1, borderColor: "divider", borderRadius: 1 }}>
        <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
          Test read
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
            Minimum confidence: {Math.round(minConfidence * 100)}%
          </Typography>
          <Slider
            size="small"
            value={minConfidence}
            min={0}
            max={1}
            step={0.05}
            onChange={(_, value) => onMinConfidenceChange(value as number)}
            valueLabelDisplay="auto"
            valueLabelFormat={(v) => `${Math.round(v * 100)}%`}
            sx={{ maxWidth: 320 }}
          />
        </Box>

        <Stack direction="row" spacing={2} sx={{ alignItems: "center" }}>
          <Button variant="outlined" size="small" disabled={!region || testing} onClick={() => void handleTest()}>
            {testing ? "Testing..." : "Test this frame"}
          </Button>
          <Typography variant="caption" color="text.secondary">
            Reads the box at the scrubbed time above, right now - not saved anywhere.
          </Typography>
        </Stack>

        {testError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {testError}
          </Alert>
        )}

        {testResult && (
          <Box sx={{ mt: 2 }}>
            <Stack direction="row" spacing={4} sx={{ mb: 1 }}>
              <ReadResult label="Left" value={testResult.left} confidence={testResult.left_confidence} />
              <ReadResult label="Right" value={testResult.right} confidence={testResult.right_confidence} />
            </Stack>

            {testResult.detections.length > 0 ? (
              <>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
                  What it saw:
                </Typography>
                <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", gap: 0.5 }}>
                  {testResult.detections.map((d, i) => (
                    <Chip
                      key={i}
                      size="small"
                      label={`"${d.text}" · ${Math.round(d.confidence * 100)}%`}
                      variant={d.confidence < minConfidence ? "outlined" : "filled"}
                      sx={d.confidence < minConfidence ? { opacity: 0.6 } : undefined}
                    />
                  ))}
                </Stack>
              </>
            ) : (
              <Typography variant="caption" color="text.secondary">
                Nothing recognizable in this box - try a clearer frame, or a tighter/looser region.
              </Typography>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}
