import { useCallback, useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import { PHASE_ONE_STAGES, PHASE_TWO_STAGES, phaseOneComplete } from "../lib/stages";
import type { Job } from "../lib/types";
import { CalibrationPanel } from "./CalibrationPanel";
import { LoadingSpinner } from "./LoadingSpinner";
import { ResultsView } from "./ResultsView";
import { StageProgress } from "./StageProgress";

const POLL_INTERVAL_MS = 3000;
const RUNNING_STATUSES: Job["status"][] = ["processing", "finalizing"];

interface JobWorkspaceProps {
  jobId: string;
  onJobUpdated: (job: Job) => void;
  onBackToDashboard: () => void;
}

export function JobWorkspace({ jobId, onJobUpdated, onBackToDashboard }: JobWorkspaceProps) {
  const [job, setJob] = useState<Job | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [calibrated, setCalibrated] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    let cancelled = false;

    function apply(updated: Job) {
      if (cancelled) return;
      setJob(updated);
      onJobUpdated(updated);
      if (!RUNNING_STATUSES.includes(updated.status)) {
        setCancelling(false);
        // Nothing changes this job's status on its own once it's out of a
        // running state - polling every 3s forever for as long as this
        // page stayed open was just wasted requests.
        clearInterval(timer);
      }
    }

    api.getJob(jobId).then(apply).catch((err) => {
      if (!cancelled) setLoadError(err instanceof Error ? err.message : String(err));
    });

    const timer = setInterval(() => {
      api.getJob(jobId).then(apply).catch(() => undefined);
    }, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [jobId, onJobUpdated]);

  const applyUpdate = useCallback(
    (updated: Job) => {
      setJob(updated);
      onJobUpdated(updated);
    },
    [onJobUpdated],
  );

  // Returns the updated job on success, or null if the action failed (the
  // error is already surfaced via actionError in that case).
  const applyAction = useCallback(
    async (action: () => Promise<Job>): Promise<Job | null> => {
      setActionError(null);
      try {
        const updated = await action();
        applyUpdate(updated);
        return updated;
      } catch (err) {
        setActionError(err instanceof Error ? err.message : String(err));
        return null;
      }
    },
    [applyUpdate],
  );

  function handleCancel() {
    setCancelling(true);
    void applyAction(() => api.cancelJob(jobId));
  }

  // Kicking off processing/finalizing is a fire-and-forget action from
  // here on - the video's status chip on the dashboard tracks progress, so
  // there's no need to keep watching a dedicated screen for it.
  async function handleStartProcessing() {
    const updated = await applyAction(() => api.processJob(jobId));
    if (updated) onBackToDashboard();
  }

  async function handleRetry() {
    if (!job) return;
    const updated = await applyAction(() =>
      phaseOneComplete(job.completed_stages) ? api.finalizeJob(jobId) : api.processJob(jobId),
    );
    if (updated) onBackToDashboard();
  }

  if (loadError) return <Alert severity="error">{loadError}</Alert>;
  if (!job) return <LoadingSpinner />;

  return (
    <Box>
      {actionError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {actionError}
        </Alert>
      )}

      {job.status === "uploaded" && (
        <Box sx={{ maxWidth: 900 }}>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
            {job.original_filename}
          </Typography>
          <CalibrationPanel job={job} onCalibrated={() => setCalibrated(true)} />
          {calibrated && (
            <Button variant="contained" sx={{ mt: 3 }} onClick={handleStartProcessing}>
              Start processing
            </Button>
          )}
        </Box>
      )}

      {job.status === "processing" && (
        <StageProgress
          title="Processing video"
          note="This can take a while depending on video length. You can head back to the dashboard - it'll keep going in the background."
          stages={PHASE_ONE_STAGES}
          completedStages={job.completed_stages}
          currentStage={job.stage}
          stageDurations={job.stage_durations_s}
          stageStartedAt={job.updated_at}
          onCancel={handleCancel}
          cancelling={cancelling}
        />
      )}

      {job.status === "finalizing" && (
        <StageProgress
          title="Finalizing"
          note="Consolidating stats, building the dashboard, and rendering the annotated video. You can head back to the dashboard - it'll keep going in the background."
          stages={PHASE_TWO_STAGES}
          completedStages={job.completed_stages}
          currentStage={job.stage}
          stageDurations={job.stage_durations_s}
          stageStartedAt={job.updated_at}
          onCancel={handleCancel}
          cancelling={cancelling}
        />
      )}

      {job.status === "complete" && <ResultsView job={job} onJobUpdated={applyUpdate} />}

      {job.status === "error" && (
        <Box sx={{ maxWidth: 560 }}>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 1.5 }}>
            Something went wrong
          </Typography>
          <Alert severity="error" sx={{ mb: 2 }}>
            {job.error}
          </Alert>
          <Button variant="contained" onClick={handleRetry}>
            Retry
          </Button>
        </Box>
      )}

      {job.status === "cancelled" && (
        <Box sx={{ maxWidth: 560 }}>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
            Cancelled
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            Processing was stopped before it finished.
          </Typography>
          <Button variant="contained" onClick={handleRetry}>
            Start again
          </Button>
        </Box>
      )}
    </Box>
  );
}
