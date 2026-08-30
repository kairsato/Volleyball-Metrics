import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import Box from "@mui/material/Box";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { api } from "../../lib/api";
import type { OcrRegion } from "../../lib/types";
import { LoadingSpinner } from "../LoadingSpinner";
import { formatTimestamp } from "./types";

interface ScoreRegionPickerProps {
  jobId: string;
  minTimeS?: number;
  maxTimeS: number;
  region: OcrRegion | null;
  onRegionChange: (region: OcrRegion) => void;
}

// Magnifier preview shown in the top-left corner: a zoomed-in crop of
// whatever box is currently selected (mid-drag, or the already-saved
// region), so fine scoreboard digits are readable while drawing/checking
// the box rather than only at the frame's native resolution.
const MAGNIFIER_MAX_WIDTH_PX = 220;
const MAGNIFIER_MAX_HEIGHT_PX = 140;
const MAGNIFIER_MAX_SCALE = 6;

// Draw-to-select is the whole interaction: mouse down starts a corner,
// dragging grows the box from there, mouse up commits it. Simpler than
// CalibrationPanel's drag-existing-handles scheme since a scoreboard box
// only ever needs redrawing from scratch, not fine per-corner nudging.
export function ScoreRegionPicker({ jobId, minTimeS = 0, maxTimeS, region, onRegionChange }: ScoreRegionPickerProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [time, setTime] = useState(minTimeS);
  const [loadingFrame, setLoadingFrame] = useState(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const magnifierCanvasRef = useRef<HTMLCanvasElement>(null);

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

  function scale(): number {
    return naturalSize && naturalSize.width > 0 ? displayWidth / naturalSize.width : 1;
  }

  function toNatural(event: ReactMouseEvent<HTMLDivElement>): { x: number; y: number } | null {
    if (!imageRef.current || !naturalSize) return null;
    const rect = imageRef.current.getBoundingClientRect();
    const s = scale();
    const x = Math.max(0, Math.min(naturalSize.width, (event.clientX - rect.left) / s));
    const y = Math.max(0, Math.min(naturalSize.height, (event.clientY - rect.top) / s));
    return { x, y };
  }

  function handleImageLoad() {
    const img = imageRef.current;
    if (!img) return;
    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    setDisplayWidth(img.clientWidth);
  }

  function handleMouseDown(event: ReactMouseEvent<HTMLDivElement>) {
    const point = toNatural(event);
    if (!point) return;
    setDrawStart(point);
    setDrawCurrent(point);
  }

  function handleMouseMove(event: ReactMouseEvent<HTMLDivElement>) {
    if (!drawStart) return;
    const point = toNatural(event);
    if (point) setDrawCurrent(point);
  }

  function handleMouseUp() {
    if (!drawStart || !drawCurrent) return;
    const x = Math.min(drawStart.x, drawCurrent.x);
    const y = Math.min(drawStart.y, drawCurrent.y);
    const width = Math.abs(drawCurrent.x - drawStart.x);
    const height = Math.abs(drawCurrent.y - drawStart.y);
    setDrawStart(null);
    setDrawCurrent(null);
    if (width < 4 || height < 4) return; // a click, not a drag - ignore
    onRegionChange({ x, y, width, height });
  }

  const s = scale();
  // Memoized (not recomputed as a fresh object every render) so the
  // magnifier effect below can depend on it directly - it only changes
  // identity when the drag or the saved region actually changes.
  const liveBox = useMemo(
    () =>
      drawStart && drawCurrent
        ? {
            x: Math.min(drawStart.x, drawCurrent.x),
            y: Math.min(drawStart.y, drawCurrent.y),
            width: Math.abs(drawCurrent.x - drawStart.x),
            height: Math.abs(drawCurrent.y - drawStart.y),
          }
        : region,
    [drawStart, drawCurrent, region],
  );

  // Redraws the magnifier's canvas from the source <img> (not the
  // downscaled display box), cropped to liveBox and scaled up, every time
  // the selection moves/resizes - cheap enough to do on every drag frame
  // since the crop is small. imageSmoothingEnabled is off so zoomed-in
  // scoreboard digits stay crisp instead of blurring.
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

  return (
    <Box>
      <Typography color="text.secondary" sx={{ mb: 1.5 }}>
        Scrub to a moment where the scoreboard is clearly visible, then drag a box around it.
      </Typography>

      <Stack direction="row" spacing={2} sx={{ alignItems: "center", mb: 1.5 }}>
        <Slider
          size="small"
          value={time}
          min={minTimeS}
          max={Math.max(minTimeS + 1, maxTimeS)}
          onChange={(_, value) => setTime(value as number)}
          onChangeCommitted={(_, value) => void loadFrame(value as number)}
          sx={{ maxWidth: 320 }}
        />
        <Typography variant="body2" color="text.secondary" sx={{ minWidth: 40 }}>
          {formatTimestamp(time)}
        </Typography>
      </Stack>

      <Box
        sx={{
          position: "relative",
          display: "inline-block",
          maxWidth: "100%",
          cursor: "crosshair",
          userSelect: "none",
          border: 1,
          borderColor: "divider",
          borderRadius: 1,
          overflow: "hidden",
          lineHeight: 0,
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setDrawStart(null);
          setDrawCurrent(null);
        }}
      >
        {/* The spinner only replaces the image before the very first frame
            has ever loaded - once one exists, scrubbing keeps the current
            frame visible (just dimmed) while the next one loads instead of
            swapping it out for a fixed-height spinner box, which was
            causing the picture to flash away and the whole box to jump
            between the spinner's height and the image's actual height on
            every scrub. */}
        {!frameUrl ? (
          <LoadingSpinner minHeight={200} />
        ) : (
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
        )}

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
      </Box>
    </Box>
  );
}
