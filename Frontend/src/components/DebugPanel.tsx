import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import LinearProgress from "@mui/material/LinearProgress";
import Typography from "@mui/material/Typography";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { Job, JobStatus } from "../lib/types";

interface Failure {
  name: string;
  error: string;
}

// A job's current status decides what "reprocess" actually means for it -
// mirrors jobs_router.py's own guards (process_job accepts uploaded/error/
// cancelled directly; redo_job only accepts complete, and resets to
// uploaded before process_job can run). A job already mid-pipeline or
// waiting on a human (player review) is left alone entirely rather than
// force-restarted out from under whoever's looking at it.
type ReprocessAction = "redo-and-process" | "process" | "skip";

function actionFor(status: JobStatus): ReprocessAction {
  if (status === "complete") return "redo-and-process";
  if (status === "uploaded" || status === "error" || status === "cancelled") return "process";
  return "skip"; // processing, finalizing, awaiting_player_review
}

// Settings page's Debug tab - see SettingsPage.tsx. Tools for validating
// and re-running the pipeline itself (not one game's results, which is
// what Setup/ResultsView's own "Redo" already covers per-game) - reprocess
// everything at once, or step through one game's detections model-by-model
// to spot and correct mistakes (see DebugReviewPage.tsx).
export function DebugPanel() {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number; current: string | null }>({
    done: 0,
    total: 0,
    current: null,
  });
  const [failures, setFailures] = useState<Failure[]>([]);

  useEffect(() => {
    api
      .listJobs()
      .then(setJobs)
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, []);

  const eligibleJobs = jobs.filter((j) => actionFor(j.status) !== "skip");
  const skippedJobs = jobs.filter((j) => actionFor(j.status) === "skip");
  const resetCount = eligibleJobs.filter((j) => actionFor(j.status) === "redo-and-process").length;

  async function handleReprocessAll() {
    setConfirmOpen(false);
    setReprocessing(true);
    setFailures([]);
    setProgress({ done: 0, total: eligibleJobs.length, current: null });

    for (let i = 0; i < eligibleJobs.length; i++) {
      const job = eligibleJobs[i];
      setProgress({ done: i, total: eligibleJobs.length, current: job.original_filename });
      try {
        // Only a complete job needs the reset first - redoJob resets it to
        // "uploaded" (processJob 409s on anything else), one already
        // uploaded/errored/cancelled can go straight to processJob.
        if (actionFor(job.status) === "redo-and-process") await api.redoJob(job.id);
        await api.processJob(job.id);
      } catch (err) {
        setFailures((prev) => [
          ...prev,
          { name: job.original_filename, error: err instanceof Error ? err.message : String(err) },
        ]);
      }
    }

    setProgress((p) => ({ ...p, done: eligibleJobs.length, current: null }));
    setReprocessing(false);
  }

  return (
    <Box>
      <Typography color="text.secondary" sx={{ mb: 3, maxWidth: 900 }}>
        Tools for validating and re-running the detection pipeline itself, not one game's results specifically - see
        the Configuration tab for tuning the heuristics these stages use.
      </Typography>

      {loadError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {loadError}
        </Alert>
      )}

      <Card variant="outlined" sx={{ p: 3, mb: 3, maxWidth: 900 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
          Reprocess all games
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Re-runs the detection pipeline on every game, picking the right action for each one's current state: a
          completed game is fully reset and reprocessed from scratch (identical to hitting "Redo" on it
          individually); one still uploaded, errored, or cancelled just has processing (re)started. A game already
          processing, finalizing, or awaiting player review is left alone. Useful right after changing a
          Configuration profile's heuristics, to see how the new tuning performs across the whole library at once.
          {resetCount > 0 &&
            ` Resetting a completed game clears its court calibration, player identification, and scoring setup - you'll need to redo those afterward.`}
        </Typography>

        {eligibleJobs.length === 0 ? (
          <Alert severity="info">
            {jobs.length === 0 ? "No games to reprocess yet." : "Every game is currently active - nothing eligible to reprocess right now."}
          </Alert>
        ) : (
          <>
            <Button
              variant="outlined"
              color="warning"
              disabled={reprocessing}
              onClick={() => setConfirmOpen(true)}
            >
              Reprocess {eligibleJobs.length} game{eligibleJobs.length === 1 ? "" : "s"}
            </Button>
            {skippedJobs.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                Skipping {skippedJobs.length} currently active game{skippedJobs.length === 1 ? "" : "s"}:{" "}
                {skippedJobs.map((j) => j.original_filename).join(", ")}.
              </Typography>
            )}

            {reprocessing && (
              <Box sx={{ mt: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={progress.total ? (progress.done / progress.total) * 100 : 0}
                />
                <Typography variant="caption" color="text.secondary">
                  {progress.current
                    ? `Reprocessing "${progress.current}"… (${progress.done + 1}/${progress.total})`
                    : `Done (${progress.done}/${progress.total})`}
                </Typography>
              </Box>
            )}

            {!reprocessing && progress.total > 0 && failures.length === 0 && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Queued {progress.total} game{progress.total === 1 ? "" : "s"} for reprocessing - watch their status
                on the Games page.
              </Alert>
            )}
            {!reprocessing && failures.length > 0 && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                {failures.length} of {progress.total} game{progress.total === 1 ? "" : "s"} failed to start:{" "}
                {failures.map((f) => f.name).join(", ")}.
              </Alert>
            )}
          </>
        )}
      </Card>

      <Card variant="outlined" sx={{ p: 3, maxWidth: 900 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
          Check accuracy
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Step through one game's detected rallies, ball tracking, player identification, and court calibration -
          see exactly what each model found, and correct anything it got wrong. Pick which game from the review
          page itself.
        </Typography>
        <Button variant="contained" onClick={() => navigate("/debug/review")}>
          Open accuracy review
        </Button>
      </Card>

      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Reprocess {eligibleJobs.length} games?</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            {resetCount > 0
              ? `${resetCount} of these are already complete - resetting them clears court calibration, player identification, and scoring setup before reprocessing. `
              : ""}
            This can take a long time for a large library and can't be undone.
          </Alert>
          <Typography variant="body2" color="text.secondary">
            Games affected: {eligibleJobs.map((j) => j.original_filename).join(", ")}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)}>Cancel</Button>
          <Button color="warning" variant="contained" onClick={() => void handleReprocessAll()}>
            Reprocess all
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
