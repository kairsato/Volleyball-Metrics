import { useEffect, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { api } from "../../lib/api";
import type { CalibrationPointsOut } from "../../lib/types";
import { LoadingSpinner } from "../LoadingSpinner";

interface CourtSummaryProps {
  jobId: string;
}

// Read-only: this just shows the calibration that was actually used to
// produce the tracking data already sitting in this job's output (and, in
// turn, the team/side guesses on the Overall tab). Editing it here
// wouldn't retroactively change anything - court.json feeds the one-time
// pixel-to-court transform that tracking already ran with, and Setup only
// re-runs the lightweight consolidation step, not tracking.
export function CourtSummary({ jobId }: CourtSummaryProps) {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [points, setPoints] = useState<CalibrationPointsOut | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;

    async function load() {
      try {
        const [url, pointsRes] = await Promise.all([api.getCalibrationFrame(jobId), api.getCalibrationPoints(jobId)]);
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        objectUrl = url;
        setFrameUrl(url);
        setPoints(pointsRes);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    }

    void load();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [jobId]);

  function handleImageLoad() {
    const img = imageRef.current;
    if (!img) return;
    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
  }

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!frameUrl || !points) return <LoadingSpinner />;

  const corners = points.corners;
  const net = points.net_points;
  const cornerPath = corners.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <Box>
      <Typography color="text.secondary" sx={{ mb: 2 }}>
        {points.calibrated
          ? "The court boundary and net position that tracking used for this video. This is read-only — changing it here wouldn't affect data that's already been tracked."
          : "This video was processed without a real court calibration, so team/side and speed data may be unreliable."}
      </Typography>

      <Box sx={{ position: "relative", width: "100%", maxWidth: 640, borderRadius: 2, overflow: "hidden", lineHeight: 0 }}>
        <Box
          component="img"
          ref={imageRef}
          src={frameUrl}
          onLoad={handleImageLoad}
          alt="Calibration frame"
          sx={{ width: "100%", display: "block" }}
        />
        {naturalSize && (
          <Box
            component="svg"
            viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
            sx={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
          >
            <polygon points={cornerPath} fill="rgba(56,189,248,0.18)" stroke="#38bdf8" strokeWidth={3} />
            {net.length === 2 && (
              <line x1={net[0].x} y1={net[0].y} x2={net[1].x} y2={net[1].y} stroke="#f97316" strokeWidth={4} />
            )}
            {corners.map((p, i) => (
              <circle key={`c${i}`} cx={p.x} cy={p.y} r={8} fill="#38bdf8" stroke="#fff" strokeWidth={2} />
            ))}
            {net.map((p, i) => (
              <circle key={`n${i}`} cx={p.x} cy={p.y} r={7} fill="#f97316" stroke="#fff" strokeWidth={2} />
            ))}
          </Box>
        )}
      </Box>
    </Box>
  );
}
