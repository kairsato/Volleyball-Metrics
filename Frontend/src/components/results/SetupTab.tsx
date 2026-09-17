import { useState, type ComponentType } from "react";
import { useNavigate } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Divider from "@mui/material/Divider";
import Stack from "@mui/material/Stack";
import type { SxProps, Theme } from "@mui/material/styles";
import Typography from "@mui/material/Typography";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import EditIcon from "@mui/icons-material/Edit";
import GridOnIcon from "@mui/icons-material/GridOn";
import PeopleAltIcon from "@mui/icons-material/PeopleAlt";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import ScoreboardIcon from "@mui/icons-material/Scoreboard";
import TimerIcon from "@mui/icons-material/Timer";
import { api } from "../../lib/api";
import { formatDuration, formatProcessedAt, PHASE_ONE_STAGES, PHASE_TWO_STAGES, STAGE_LABELS } from "../../lib/stages";
import type { Job } from "../../lib/types";

interface SetupCardProps {
  icon: ComponentType<{ sx?: SxProps<Theme> }>;
  title: string;
  description: string;
  needsAttention: boolean;
  // When true, the un-done state renders as a neutral "Not set" chip
  // instead of the "Automatically determined" chip - for a step that's
  // genuinely opt-in (e.g. Warmup Period) and has no automatic value to
  // speak of until the user sets one, rather than something every video
  // gets an algorithmic answer for up front.
  optional?: boolean;
  onClick: () => void;
}

// Icon and title sit in their own centered row so they line up on their own
// baseline regardless of icon size - the description (and, before, this
// same icon at alignItems: "flex-start" against a two-line title) used to
// throw that off since a longer description made the row taller without
// the icon and title actually sharing a center line.
//
// The chip communicates provenance, not just completion: everything here
// starts out algorithmic (tracking's own player guesses, an OCR/heuristic
// score read, calibration's un-reviewed points) and stays labeled
// "Automatically determined" until a human actually confirms or edits it,
// at which point it becomes "User modified" - needsAttention is already
// exactly "not yet confirmed" for every one of these steps (see each
// step's own confirm() call), so no separate provenance flag is needed.
function SetupCard({ icon: Icon, title, description, needsAttention, optional, onClick }: SetupCardProps) {
  return (
    <Card variant="outlined">
      <Box sx={{ p: 2.5, pb: 1.5 }}>
        <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", mb: 0.5 }}>
          <Icon sx={{ color: "primary.main", fontSize: 28, flexShrink: 0 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
            {title}
          </Typography>
          {needsAttention ? (
            optional ? (
              <Chip size="small" color="default" variant="outlined" label="Not set" />
            ) : (
              <Chip size="small" color="warning" variant="outlined" icon={<AutoAwesomeIcon />} label="Automatically determined" />
            )
          ) : (
            <Chip size="small" color="primary" variant="outlined" icon={<EditIcon />} label="User modified" />
          )}
        </Stack>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </Box>
      <Stack direction="row" sx={{ justifyContent: "flex-end", px: 2.5, pb: 2, pt: 0.5 }}>
        <Button size="small" variant="outlined" onClick={onClick}>
          Manually override
        </Button>
      </Stack>
    </Card>
  );
}

// Every stage the pipeline can report a duration for, in the order they
// actually run - player/ball tracking through action detection (phase one),
// then stats/dashboard/video (phase two). "recalibrate" is deliberately
// left out: it only ever exists on a job that's been recalibrated at least
// once, and reprocessing here always redoes the full phase one/two set
// instead, so showing it would just be a stray extra row most jobs never
// have.
const PROCESSING_STAGES = [...PHASE_ONE_STAGES, ...PHASE_TWO_STAGES];

// Shows how long each stage took (job.stage_durations_s, the same data
// StageProgress shows live while a job is actually running - see
// lib/stages.ts) plus a way to redo the whole thing from scratch. There's
// no per-stage reprocess: the backend doesn't support re-running just one
// stage in isolation (most stages downstream of ball/player tracking depend
// on that data), and a plain calibration change already has its own cheap
// path (CalibrationPanel's recalibrate flow) that doesn't need a full redo.
function ProcessingSection({ job }: { job: Job }) {
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [reprocessing, setReprocessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const totalSeconds = PROCESSING_STAGES.reduce((sum, stage) => sum + (job.stage_durations_s[stage] ?? 0), 0);

  async function handleReprocessConfirm() {
    setReprocessing(true);
    setError(null);
    try {
      await api.redoJob(job.id);
      await api.processJob(job.id);
      // NOT navigate(`/game?job=${job.id}`) - that's the route this
      // component is already mounted on, so React Router wouldn't remount
      // anything: JobWorkspace's own job state (already fetched once as
      // "complete", which stopped its polling loop for good) never learns
      // about the fresh "processing" status, this dialog never closes, and
      // the Setup tab just sits there looking unchanged forever. /games is
      // a genuinely different route - it force-remounts, and its own
      // JobStatusChip is what actually shows live progress from here (same
      // pattern JobWorkspace.handleStartProcessing already uses to kick off
      // a fresh job's processing).
      navigate("/games");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setReprocessing(false);
    }
  }

  return (
    <Card variant="outlined" sx={{ p: 2.5 }}>
      <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", mb: 0.5 }}>
        <AccessTimeIcon sx={{ color: "primary.main", fontSize: 28, flexShrink: 0 }} />
        <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
          Processing time
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Total: {formatDuration(totalSeconds)}
        </Typography>
      </Stack>

      {/* Always rendered (even with nothing to show yet, for a job
          processed before this field existed) so this reserves the same
          line height either way - conditionally rendering the whole
          element would collapse the gap before the stage list below. */}
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 2 }}>
        {job.processed_at ? `Last ran ${formatProcessedAt(job.processed_at)}` : " "}
      </Typography>

      <Stack spacing={0.75} sx={{ mb: 2 }}>
        {PROCESSING_STAGES.map((stage) => {
          const duration = job.stage_durations_s[stage];
          return (
            <Stack key={stage} direction="row" sx={{ justifyContent: "space-between" }}>
              <Typography variant="body2" color="text.secondary">
                {STAGE_LABELS[stage] ?? stage}
              </Typography>
              <Typography variant="body2" color={duration === undefined ? "text.disabled" : "text.primary"}>
                {duration === undefined ? "Not run" : formatDuration(duration)}
              </Typography>
            </Stack>
          );
        })}
      </Stack>

      <Divider sx={{ mb: 2 }} />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Processing has no randomness in it - same video, same calibration
          in, same tracking/stats out, always - so it's not a "try again,
          maybe it tracks better this time" button. But unlike a plain
          recalibrate, reprocessing also clears the existing calibration
          itself (see jobs_router._reset_for_full_reprocess) rather than
          reapplying it, so "nothing changed" essentially never applies to a
          reprocess the way it does to, say, re-finalizing - there's always
          at least a fresh (uncalibrated) tracking pass to redo court setup
          against afterward. Steering the "just want to fix calibration"
          case toward the cheap path here, rather than after they've already
          opened the reprocess dialog, is what keeps that dialog itself
          focused on the one-way stuff (fresh player IDs) instead of
          re-explaining this distinction too. */}
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Processing is deterministic: given the same video and calibration, reprocessing always produces
        the same results. Reprocessing clears the current court calibration along with everything else, so
        if court calibration is the only thing that needs fixing, using the Court Calibration step above
        is faster and leaves player names, groupings, and score corrections in place.
      </Typography>

      <Button color="warning" variant="outlined" startIcon={<RestartAltIcon />} onClick={() => setDialogOpen(true)}>
        Reprocess video
      </Button>

      <Dialog open={dialogOpen} onClose={() => (!reprocessing ? setDialogOpen(false) : undefined)}>
        <DialogTitle>Reprocess this video?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This re-runs every stage from scratch - player tracking, ball detection, stats, dashboard,
            and the annotated video - replacing the current results. It can take a while depending on
            video length; you'll be taken to a progress screen and can head back to the dashboard from
            there while it runs.
          </DialogContentText>
          <DialogContentText sx={{ mt: 1.5 }}>
            Court calibration, player names/groupings, and any score corrections are all cleared, not
            carried forward - tracking assigns fresh player IDs each run, so old names would otherwise end
            up attached to the wrong people, and calibration marked against the previous run has no
            guaranteed relationship to this one. All three will need to be redone from the Setup tab
            afterward - if calibration is the only thing you need to fix, recalibrating from the Court
            Calibration step instead is much cheaper and keeps everything else.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)} disabled={reprocessing}>
            Cancel
          </Button>
          <Button color="warning" variant="contained" disabled={reprocessing} onClick={() => void handleReprocessConfirm()}>
            {reprocessing ? "Starting..." : "Reprocess"}
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
}

interface SetupTabProps {
  job: Job;
}

// The Setup tab's landing view: one card per setup step, each just a title,
// description, and a chip showing whether it's still the algorithm's
// unreviewed guess ("Automatically determined") or something a human has
// confirmed/edited ("User modified") - the actual work for all four happens
// on its own full page (Court Calibration, Player Identification, Scoring
// Determination, Warmup Period), reached via the same
// /game/setup/<step>?job=<id> URL pattern.
//
// Player Identification used to be locked until court calibration was
// done. It no longer is - the court boundary is itself only ever an
// algorithmic best guess (see PlayerIdentificationPage's own warning
// dialog), so gating one heuristic step behind another added friction
// without actually guaranteeing accuracy. Court calibration is still
// listed first since it's the natural place to start.
export function SetupTab({ job }: SetupTabProps) {
  const navigate = useNavigate();
  const courtCalibrated = job.completed_stages.includes("court_calibration");

  return (
    <Stack spacing={2}>
      <Alert severity="info">
        Every stat on this page is computed from the values below. They come from tracking,
        heuristics, and OCR - not a human - so they <strong>can be wrong</strong>. It's OK to
        override any of them if something looks off; just expect stats to shift once you do.
      </Alert>
      <SetupCard
        icon={GridOnIcon}
        title="Court Calibration"
        description="Mark the court boundary and net position so tracking can work out where players and the ball are."
        needsAttention={!courtCalibrated}
        onClick={() => navigate(`/game/setup/court-calibration?job=${job.id}`)}
      />
      <SetupCard
        icon={PeopleAltIcon}
        title="Player Identification"
        description="Assign names to detected players, merge duplicates, or ignore false detections."
        needsAttention={job.needs_player_id}
        onClick={() =>
          navigate(`/game/setup/player-identification?job=${job.id}`, {
            state: { showCourtAccuracyWarning: true },
          })
        }
      />
      <SetupCard
        icon={ScoreboardIcon}
        title="Scoring Determination"
        description="Determine or correct the match score - who won each rally and how rallies group into sets."
        needsAttention={job.needs_scoring_review}
        onClick={() => navigate(`/game/setup/scoring-determination?job=${job.id}`)}
      />
      <SetupCard
        icon={TimerIcon}
        title="Warmup Period"
        description="Trim off pre-game warmup (or anything after the match) - footage, the video player, thumbnails, and every rally/stat outside that window are excluded everywhere."
        needsAttention={!job.warmup_confirmed}
        optional
        onClick={() => navigate(`/game/setup/warmup-period?job=${job.id}`)}
      />
      <ProcessingSection job={job} />
    </Stack>
  );
}
