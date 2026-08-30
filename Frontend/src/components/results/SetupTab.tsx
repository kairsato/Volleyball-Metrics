import { useState } from "react";
import type { ComponentType } from "react";
import { useNavigate } from "react-router-dom";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import type { SxProps, Theme } from "@mui/material/styles";
import Typography from "@mui/material/Typography";
import CheckCircleOutlinedIcon from "@mui/icons-material/CheckCircleOutlined";
import CloseIcon from "@mui/icons-material/Close";
import GridOnIcon from "@mui/icons-material/GridOn";
import PeopleAltIcon from "@mui/icons-material/PeopleAlt";
import ScoreboardIcon from "@mui/icons-material/Scoreboard";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { api } from "../../lib/api";
import type { Job } from "../../lib/types";
import { CourtSummary } from "./CourtSummary";
import { RedoButton } from "./RedoButton";

interface SetupCardProps {
  icon: ComponentType<{ sx?: SxProps<Theme> }>;
  title: string;
  description: string;
  needsAttention: boolean;
  onClick: () => void;
}

// Icon and title sit in their own centered row so they line up on their own
// baseline regardless of icon size - the description (and, before, this
// same icon at alignItems: "flex-start" against a two-line title) used to
// throw that off since a longer description made the row taller without
// the icon and title actually sharing a center line.
function SetupCard({ icon: Icon, title, description, needsAttention, onClick }: SetupCardProps) {
  return (
    <Card variant="outlined">
      <CardActionArea onClick={onClick} sx={{ p: 2.5 }}>
        <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", mb: 0.5 }}>
          <Icon sx={{ color: "primary.main", fontSize: 28, flexShrink: 0 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 600, flex: 1 }}>
            {title}
          </Typography>
          {needsAttention ? (
            <Chip size="small" color="warning" variant="outlined" icon={<WarningAmberIcon />} label="Needs attention" />
          ) : (
            <Chip size="small" color="success" variant="outlined" icon={<CheckCircleOutlinedIcon />} label="Done" />
          )}
        </Stack>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </CardActionArea>
    </Card>
  );
}

interface SetupTabProps {
  job: Job;
  onJobUpdated: (job: Job) => void;
}

// The Setup tab's landing view: one card per setup step, each just a title,
// description, and a Done/Needs attention chip - the actual work happens
// either in a dialog right here (Court Calibration, which is a quick
// read-only review + redo, not worth leaving the results page for) or on
// its own full page (Player Identification and Scoring Determination,
// which both need real room to work in). Redoing court calibration doesn't
// need its own navigation the way the other two redo flows do - it just
// calls onJobUpdated, and JobWorkspace (the parent of ResultsView) swaps
// this whole results view out for the calibration/processing screens on
// its own once the job's status changes.
//
// Court Calibration is always "Done" here - SetupTab only ever renders for
// a "complete" job, and a job can't reach that status without having been
// calibrated in the first place (see jobs_router.process_job).
export function SetupTab({ job, onJobUpdated }: SetupTabProps) {
  const navigate = useNavigate();
  const [courtDialogOpen, setCourtDialogOpen] = useState(false);

  return (
    <>
      <Stack spacing={2}>
        <SetupCard
          icon={GridOnIcon}
          title="Court Calibration"
          description="Review the court boundary and net position that tracking used for this video."
          needsAttention={false}
          onClick={() => setCourtDialogOpen(true)}
        />
        <SetupCard
          icon={PeopleAltIcon}
          title="Player Identification"
          description="Assign names to detected players, merge duplicates, or ignore false detections."
          needsAttention={job.needs_player_id}
          onClick={() => navigate(`/video/setup/player-identification?job=${job.id}`)}
        />
        <SetupCard
          icon={ScoreboardIcon}
          title="Scoring Determination"
          description="Determine or correct the match score - who won each rally and how rallies group into games."
          needsAttention={job.needs_scoring_review}
          onClick={() => navigate(`/video/setup/scoring-determination?job=${job.id}`)}
        />
      </Stack>

      <Dialog open={courtDialogOpen} onClose={() => setCourtDialogOpen(false)} maxWidth="lg" fullWidth>
        <DialogTitle>Court Calibration</DialogTitle>
        <IconButton
          aria-label="Close"
          onClick={() => setCourtDialogOpen(false)}
          sx={{ position: "absolute", right: 12, top: 12, color: "text.secondary" }}
        >
          <CloseIcon />
        </IconButton>
        <DialogContent>
          <CourtSummary jobId={job.id} />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <RedoButton
            label="Redo court calibration"
            confirmTitle="Redo court calibration?"
            confirmText={`This clears every player name and ignored flag set for "${job.original_filename}" (a fresh tracking run assigns new player identities) and takes it back to the calibration step. You'll need to click "Start processing" again afterward. This can't be undone.`}
            onConfirm={async () => {
              onJobUpdated(await api.redoJob(job.id));
              setCourtDialogOpen(false);
            }}
          />
        </DialogActions>
      </Dialog>
    </>
  );
}
