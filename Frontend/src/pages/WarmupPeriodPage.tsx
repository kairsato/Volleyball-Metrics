import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { ToolPageSkeleton } from "../components/Skeletons";
import { WarmupPanel } from "../components/WarmupPanel";

// Same viewport-fit approach as CourtCalibrationPage.tsx/PlayerIdentificationPage.tsx
// - the page itself never scrolls, only WarmupPanel's own content does.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

interface WarmupPeriodPageProps {
  onJobUpdated: (job: Job) => void;
}

// The fourth full page a Setup card links to - a "back to results" header
// plus WarmupPanel filling the rest of the page. Unlike Court Calibration/
// Player Identification/Scoring Determination, saving here doesn't touch
// the job's own status or re-run any pipeline stage - it's a pure viewing/
// results filter applied at read time (see Backend/API/warmup.py), so
// there's no onSaved-triggers-navigation dance: this page just re-fetches
// the job so its own "Done"/"Not set" chip (and everywhere else that reads
// Job.warmup_confirmed) reflects the change immediately.
export function WarmupPeriodPage({ onJobUpdated }: WarmupPeriodPageProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");

  const [job, setJob] = useState<Job | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  function refresh() {
    if (!jobId) return;
    api
      .getJob(jobId)
      .then((res) => {
        setJob(res);
        onJobUpdated(res);
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }

  // onJobUpdated is a stable useCallback in App.tsx - omitted here so this
  // only re-fetches when the job in the URL actually changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refresh, [jobId]);

  useEffect(() => {
    document.title = job ? `${job.original_filename} · Warmup Period` : "Warmup Period";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [job]);

  if (!jobId) return <Navigate to="/videos" replace />;
  if (loadError) return <Alert severity="error">{loadError}</Alert>;
  if (!job) return <ToolPageSkeleton />;

  if (job.status !== "complete") {
    return (
      <Box sx={{ maxWidth: 560 }}>
        <Typography color="text.secondary" sx={{ mb: 2 }}>
          This video hasn't finished processing yet, so setting a warmup period isn't available.
        </Typography>
        <Button variant="contained" onClick={() => navigate(`/video?job=${jobId}`)}>
          Back to video
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ height: PAGE_CONTENT_HEIGHT, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Button
        size="small"
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(`/video?job=${jobId}&tab=setup`)}
        sx={{ alignSelf: "flex-start", mb: 2, flexShrink: 0 }}
      >
        Back to results
      </Button>

      <Box sx={{ flex: 1, minHeight: 0 }}>
        <WarmupPanel job={job} onSaved={refresh} />
      </Box>
    </Box>
  );
}
