import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { CalibrationPanel } from "../components/CalibrationPanel";
import { ToolPageSkeleton } from "../components/Skeletons";

// Same viewport-fit approach as ResultsView.tsx/PlayerIdentificationPage.tsx/
// ScoringDeterminationPage.tsx - the page itself never scrolls, only
// CalibrationPanel's own content does.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

// The third full page a Setup card (see SetupTab.tsx) links to - a "back to
// results" header plus CalibrationPanel filling the rest of the page.
// Calibration used to happen up front, before processing could even start;
// now it's just another post-processing setup step like Player
// Identification and Scoring Determination, reachable (and redoable) from
// here at any time. Unlike those two, saving here does kick off a pipeline
// run (see CalibrationPanel.handleSave/api.recalibrateJob) - court
// coordinates and which ball trajectory is "the ball" are re-derived from
// already-tracked data against the new calibration, not from scratch, so
// this is normally cheap rather than a full reprocess.
interface CourtCalibrationPageProps {
  onJobUpdated: (job: Job) => void;
}

export function CourtCalibrationPage({ onJobUpdated }: CourtCalibrationPageProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");

  const [job, setJob] = useState<Job | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    api
      .getJob(jobId)
      .then(setJob)
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, [jobId]);

  useEffect(() => {
    document.title = job ? `${job.original_filename} · Court Calibration` : "Court Calibration";
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
          This video hasn't finished processing yet, so court calibration isn't available.
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
        <CalibrationPanel
          job={job}
          onSaved={(updated) => {
            onJobUpdated(updated);
            // Saving just kicked off a full re-run of phase one (see
            // CalibrationPanel.handleSave) - the job is "processing" again,
            // not "complete", so there's nothing left for this page to show.
            navigate(`/video?job=${jobId}`);
          }}
        />
      </Box>
    </Box>
  );
}
