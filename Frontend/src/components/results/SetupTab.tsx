import { useState, type ComponentType } from "react";
import { useNavigate } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Divider from "@mui/material/Divider";
import Stack from "@mui/material/Stack";
import type { SxProps, Theme } from "@mui/material/styles";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import CheckCircleOutlinedIcon from "@mui/icons-material/CheckCircleOutlined";
import GridOnIcon from "@mui/icons-material/GridOn";
import LockIcon from "@mui/icons-material/Lock";
import PeopleAltIcon from "@mui/icons-material/PeopleAlt";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import ScoreboardIcon from "@mui/icons-material/Scoreboard";
import TimerIcon from "@mui/icons-material/Timer";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { api } from "../../lib/api";
import { formatDuration, formatProcessedAt, PHASE_ONE_STAGES, PHASE_TWO_STAGES, STAGE_LABELS } from "../../lib/stages";
import type { Job } from "../../lib/types";

interface SetupCardProps {
  icon: ComponentType<{ sx?: SxProps<Theme> }>;
  title: string;
  description: string;
  needsAttention: boolean;
  // When true, the un-done state renders as a neutral "Not set" chip
  // instead of the "Needs attention" warning - for a step that's genuinely
  // opt-in (e.g. Warmup Period) rather than something every video is
  // expected to eventually complete.
  optional?: boolean;
  // Locked out entirely (dimmed, unclickable) until some prerequisite is
  // met - e.g. Player Identification needs court calibration done first,
  // since identification (and everything downstream of it) depends on
  // knowing where the court actually is.
  locked?: boolean;
  lockedReason?: string;
  onClick: () => void;
}

// Icon and title sit in their own centered row so they line up on their own
// baseline regardless of icon size - the description (and, before, this
// same icon at alignItems: "flex-start" against a two-line title) used to
// throw that off since a longer description made the row taller without
// the icon and title actually sharing a center line.
function SetupCard({ icon: Icon, title, description, needsAttention, optional, locked, lockedReason, onClick }: SetupCardProps) {
  return (
    <Tooltip title={locked ? lockedReason ?? "" : ""}>
      <Card variant="outlined" sx={{ opacity: locked ? 0.6 : 1 }}>
        <CardActionArea onClick={onClick} disabled={locked} sx={{ p: 2.5 }}>
          <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", mb: 0.5 }}>
            <Icon sx={{ color: locked ? "text.disabled" : "primary.main", fontSize: 28, flexShrink: 0 }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
              {title}
            </Typography>
            {locked ? (
              <Chip size="small" color="default" variant="outlined" icon={<LockIcon />} label="Locked" />
            ) : needsAttention ? (
              optional ? (
                <Chip size="small" color="default" variant="outlined" label="Not set" />
              ) : (
                <Chip size="small" color="warning" variant="outlined" icon={<WarningAmberIcon />} label="Needs attention" />
              )
            ) : (
              <Chip size="small" color="success" variant="outlined" icon={<CheckCircleOutlinedIcon />} label="Done" />
            )}
          </Stack>
          <Typography variant="body2" color="text.secondary">
            {locked ? lockedReason : description}
          </Typography>
        </CardActionArea>
      </Card>
    </Tooltip>
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
      // NOT navigate(`/video?job=${job.id}`) - that's the route this
      // component is already mounted on, so React Router wouldn't remount
      // anything: JobWorkspace's own job state (already fetched once as
      // "complete", which stopped its polling loop for good) never learns
      // about the fresh "processing" status, this dialog never closes, and
      // the Setup tab just sits there looking unchanged forever. /videos is
      // a genuinely different route - it force-remounts, and its own
      // JobStatusChip is what actually shows live progress from here (same
      // pattern JobWorkspace.handleStartProcessing already uses to kick off
      // a fresh job's processing).
      navigate("/videos");
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

      {/* Processing has no randomness in it - same video, same
          calibration in, same tracking/stats out, always. Reprocessing
          only actually changes anything if something feeding it changed
          too (recalibrating, a newer model/build) - it's not a "try again,
          maybe it tracks better this time" button, and shouldn't be
          reached for expecting one. */}
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Processing is deterministic: with nothing changed since the last run, reprocessing produces the
        exact same results, not different ones. Only worth doing after recalibrating or changing the
        source footage.
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
            Court calibration is kept and reapplied automatically. Player names/groupings and any score
            corrections are not - tracking assigns fresh player IDs each run, so the old names would
            otherwise end up attached to the wrong people; both will need to be redone from the Setup tab
            afterward.
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
// description, and a Done/Needs attention chip - the actual work for all
// three happens on its own full page (Court Calibration, Player
// Identification, Scoring Determination), reached via the same
// /video/setup/<step>?job=<id> URL pattern.
//
// Player Identification is locked until court calibration is done -
// identification (and everything downstream of it) depends on knowing
// where the court actually is, so there's nothing useful to do there
// before calibration exists.
export function SetupTab({ job }: SetupTabProps) {
  const navigate = useNavigate();
  const courtCalibrated = job.completed_stages.includes("court_calibration");

  return (
    <Stack spacing={2}>
      <SetupCard
        icon={GridOnIcon}
        title="Court Calibration"
        description="Mark the court boundary and net position so tracking can work out where players and the ball are."
        needsAttention={!courtCalibrated}
        onClick={() => navigate(`/video/setup/court-calibration?job=${job.id}`)}
      />
      <SetupCard
        icon={PeopleAltIcon}
        title="Player Identification"
        description="Assign names to detected players, merge duplicates, or ignore false detections."
        needsAttention={job.needs_player_id}
        locked={!courtCalibrated}
        lockedReason="Court calibration must be completed first."
        onClick={() => navigate(`/video/setup/player-identification?job=${job.id}`)}
      />
      <SetupCard
        icon={ScoreboardIcon}
        title="Scoring Determination"
        description="Determine or correct the match score - who won each rally and how rallies group into games."
        needsAttention={job.needs_scoring_review}
        onClick={() => navigate(`/video/setup/scoring-determination?job=${job.id}`)}
      />
      <SetupCard
        icon={TimerIcon}
        title="Warmup Period"
        description="Trim off pre-game warmup (or anything after the match) - footage, the video player, thumbnails, and every rally/stat outside that window are excluded everywhere."
        needsAttention={!job.warmup_confirmed}
        optional
        onClick={() => navigate(`/video/setup/warmup-period?job=${job.id}`)}
      />
      <ProcessingSection job={job} />
    </Stack>
  );
}
