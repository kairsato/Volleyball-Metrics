import type { ComponentType } from "react";
import { useNavigate } from "react-router-dom";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import type { SxProps, Theme } from "@mui/material/styles";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import CheckCircleOutlinedIcon from "@mui/icons-material/CheckCircleOutlined";
import GridOnIcon from "@mui/icons-material/GridOn";
import LockIcon from "@mui/icons-material/Lock";
import PeopleAltIcon from "@mui/icons-material/PeopleAlt";
import ScoreboardIcon from "@mui/icons-material/Scoreboard";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import type { Job } from "../../lib/types";

interface SetupCardProps {
  icon: ComponentType<{ sx?: SxProps<Theme> }>;
  title: string;
  description: string;
  needsAttention: boolean;
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
function SetupCard({ icon: Icon, title, description, needsAttention, locked, lockedReason, onClick }: SetupCardProps) {
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
              <Chip size="small" color="warning" variant="outlined" icon={<WarningAmberIcon />} label="Needs attention" />
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
    </Stack>
  );
}
