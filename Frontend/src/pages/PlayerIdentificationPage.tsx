import { useEffect, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { Navigate, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { UnidentifiedPlayersSection } from "../components/results/UnidentifiedPlayersSection";
import { ToolPageSkeleton } from "../components/Skeletons";

// Same viewport-fit approach as ResultsView.tsx/ScoringDeterminationPage.tsx
// - the page itself never scrolls, only the players area below the header
// does once it's taller than what's left.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

interface PlayerIdentificationPageProps {
  onJobUpdated: (job: Job) => void;
}

// One of the two full pages a Setup card (see SetupTab.tsx) links to - just
// a "back to results" header plus the unidentified-players resolve area
// filling the rest of the page. Unlike the old combined Setup page, there's
// no video preview here: naming/merging/ignoring detections doesn't need
// it, and dropping it leaves the whole page for the player grid instead of
// splitting it in half.
export function PlayerIdentificationPage({ onJobUpdated }: PlayerIdentificationPageProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const jobId = searchParams.get("job");

  const [job, setJob] = useState<Job | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  // Player identification isn't locked behind court calibration - the
  // court boundary is itself only ever a heuristic best guess (the near
  // baseline and both attack lines are worked out automatically from the
  // points actually placed, see CalibrationPanel.tsx), so blocking one
  // algorithmic step on another wouldn't have guaranteed accuracy anyway.
  // This dialog is the softer version: a reminder that a wrong court
  // definition would silently poison player positions, worth double-
  // checking first. It only shows up when SetupTab's "Manually override"
  // click sent it here via router state - not on every page load/refresh.
  const [accuracyDialogOpen, setAccuracyDialogOpen] = useState(
    () => Boolean((location.state as { showCourtAccuracyWarning?: boolean } | null)?.showCourtAccuracyWarning),
  );

  useEffect(() => {
    if (!jobId) return;
    api
      .getJob(jobId)
      .then(setJob)
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, [jobId]);

  useEffect(() => {
    document.title = job ? `${job.original_filename} · Player Identification` : "Player Identification";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [job]);

  function handleJobUpdated(updated: Job) {
    setJob(updated);
    onJobUpdated(updated);
  }

  // One server-side call rather than the read-clear-unconfirm-finalize
  // sequence this used to drive from here. That sequence could only clear
  // what was already on screen, left a half-reset behind if the tab was
  // closed part-way, and - because it finished with finalize - reprocessed
  // stats and video over the OLD identities instead of working them out
  // again. See jobs_router.reset_players for what the reset now re-runs and
  // why the court is central to it. It clears the "confirmed" sign-off
  // too, which would otherwise leave the Setup tab reading "Done" with
  // nobody named.
  async function handleRedoPlayers() {
    if (!jobId) return;
    setActionError(null);
    try {
      onJobUpdated(await api.resetPlayers(jobId));
      navigate(`/game?job=${jobId}`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }

  if (!jobId) return <Navigate to="/games" replace />;
  if (loadError) return <Alert severity="error">{loadError}</Alert>;
  if (!job) return <ToolPageSkeleton />;

  if (job.status !== "complete") {
    return (
      <Box sx={{ maxWidth: 560 }}>
        <Alert severity="warning" sx={{ mb: 2 }}>
          This video hasn't finished processing yet, so player identification isn't available.
        </Alert>
        <Button variant="contained" onClick={() => navigate(`/game?job=${jobId}`)}>
          Back to game
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ height: PAGE_CONTENT_HEIGHT, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Button
        size="small"
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(`/game?job=${jobId}&tab=setup`)}
        sx={{ alignSelf: "flex-start", mb: 2, flexShrink: 0 }}
      >
        Back to results
      </Button>

      {actionError && (
        <Alert severity="error" sx={{ mb: 2, flexShrink: 0 }}>
          {actionError}
        </Alert>
      )}

      <Box sx={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
        <UnidentifiedPlayersSection job={job} onJobUpdated={handleJobUpdated} onRedoPlayers={handleRedoPlayers} />
      </Box>

      <Dialog open={accuracyDialogOpen} onClose={() => setAccuracyDialogOpen(false)}>
        <DialogTitle>Is the court definition accurate?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Player positions here are calculated from the court boundary set in Court Calibration.
            If a player's position looks wrong, check calibration first - that's usually the cause.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => navigate(`/game/setup/court-calibration?job=${jobId}`)}>Review calibration</Button>
          <Button variant="contained" onClick={() => setAccuracyDialogOpen(false)}>
            Looks right, continue
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
