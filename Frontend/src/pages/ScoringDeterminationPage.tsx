import { useEffect, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Job, Rally } from "../lib/types";
import { ScoreSection } from "../components/results/ScoreSection";
import { ToolPageSkeleton } from "../components/Skeletons";
import { toBoundedAbsolute, VideoPlayer } from "../components/VideoPlayer";

// Same viewport-fit approach as ResultsView.tsx/PlayerIdentificationPage.tsx
// - the page itself never scrolls, only ScoreSection's own content does.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

// The other full page a Setup card (see SetupTab.tsx) links to - a "back to
// results" header plus ScoreSection filling the rest of the page. Unlike
// Player Identification, the video preview stays here: judging who won a
// rally means actually watching it, not just picking from a list. Scoring
// never changes the job's own status (only names/redoing calibration do),
// so unlike the sibling pages there's no onJobUpdated to thread through.
export function ScoringDeterminationPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");

  const [job, setJob] = useState<Job | null>(null);
  const [rallies, setRallies] = useState<Rally[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!jobId) return;
    api
      .getJob(jobId)
      .then(setJob)
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    api
      .getResults(jobId)
      .then((res) => !cancelled && setRallies(res.rallies))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  useEffect(() => {
    document.title = job ? `${job.original_filename} · Scoring Determination` : "Scoring Determination";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [job]);

  // rallies (from /results) are already relative to the warmup period's
  // start once one is confirmed - see Backend/API/warmup.py. currentTime
  // (kept in sync via VideoPlayer's onTimeUpdate below) is relative the
  // same way, so ScoreSection's rally highlighting (which compares it
  // against rallies[i].start_time_s/end_time_s) lines up; seekTo converts
  // back the other way since the raw <video> element only deals in
  // absolute time.
  const warmupStartS = job?.warmup_confirmed ? job.warmup_start_s ?? 0 : 0;
  const warmupEndS = job?.warmup_confirmed ? job.warmup_end_s ?? Infinity : Infinity;

  function seekTo(timeS: number) {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = toBoundedAbsolute(timeS, warmupStartS, warmupEndS);
    void el.play();
  }

  if (!jobId) return <Navigate to="/videos" replace />;
  if (loadError) return <Alert severity="error">{loadError}</Alert>;
  if (!job) return <ToolPageSkeleton />;

  if (job.status !== "complete") {
    return (
      <Box sx={{ maxWidth: 560 }}>
        <Typography color="text.secondary" sx={{ mb: 2 }}>
          This video hasn't finished processing yet, so scoring isn't available.
        </Typography>
        <Button variant="contained" onClick={() => navigate(`/video?job=${jobId}`)}>
          Back to video
        </Button>
      </Box>
    );
  }

  // height: "100%" lets this stretch to match the Scoring Determination +
  // Estimated Score column without distorting the video - VideoPlayer
  // itself letterboxes (objectFit: "contain") to whatever box it ends up in.
  const videoElement = (
    <Box sx={{ height: "100%" }}>
      <VideoPlayer
        videoRef={videoRef}
        src={api.sourceVideoUrl(jobId)}
        boundStartS={warmupStartS}
        boundEndS={warmupEndS}
        onTimeUpdate={setCurrentTime}
      />
    </Box>
  );

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
        <ScoreSection job={job} rallies={rallies} currentTime={currentTime} onSeek={seekTo} videoElement={videoElement} />
      </Box>
    </Box>
  );
}
