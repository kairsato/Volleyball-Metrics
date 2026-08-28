import { keyframes } from "@emotion/react";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import { formatDuration, PHASE_ONE_STAGES, PHASE_TWO_STAGES, STAGE_LABELS, useElapsedSeconds } from "../lib/stages";
import type { Job } from "../lib/types";

const LIVE_STATUSES: Job["status"][] = ["processing", "finalizing"];

const CHIP_COLOR: Record<Job["status"], "default" | "warning" | "secondary" | "success" | "error"> = {
  uploaded: "default",
  processing: "warning",
  awaiting_player_review: "secondary",
  finalizing: "warning",
  complete: "success",
  error: "error",
  cancelled: "default",
};

function statusLabel(status: Job["status"]): string {
  switch (status) {
    case "uploaded":
      return "Uploaded";
    case "processing":
      return "Processing";
    case "awaiting_player_review":
      return "Needs review";
    case "finalizing":
      return "Finalizing";
    case "complete":
      return "Complete";
    case "error":
      return "Error";
    case "cancelled":
      return "Cancelled";
  }
}

const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
`;

export function JobStatusChip({ job }: { job: Job }) {
  // Called unconditionally, before the early return below - a Hook can't be
  // called only on some renders of this component, and job.status routinely
  // flips from a LIVE_STATUSES value to a non-live one between polls while
  // this same instance stays mounted for the same job.
  const elapsed = useElapsedSeconds(job.stage ? job.updated_at : null);

  if (!LIVE_STATUSES.includes(job.status)) {
    return <Chip size="small" label={statusLabel(job.status)} color={CHIP_COLOR[job.status]} sx={{ mt: 1 }} />;
  }

  const stages = job.status === "processing" ? PHASE_ONE_STAGES : PHASE_TWO_STAGES;
  const doneCount = stages.filter((s) => job.completed_stages.includes(s)).length;
  const currentStageNumber = Math.min(doneCount + 1, stages.length);
  const currentLabel = job.stage ? (STAGE_LABELS[job.stage] ?? job.stage) : "Starting...";

  return (
    <Tooltip
      arrow
      title={
        <Box sx={{ py: 0.25 }}>
          <Typography variant="caption" sx={{ fontWeight: 600, display: "block" }}>
            {currentLabel}
          </Typography>
          <Typography variant="caption" sx={{ display: "block" }}>
            Stage {currentStageNumber} of {stages.length}
          </Typography>
          {job.stage && <Typography variant="caption">Running {formatDuration(elapsed)}</Typography>}
        </Box>
      }
    >
      <Chip
        size="small"
        color={CHIP_COLOR[job.status]}
        icon={
          <CircularProgress
            size={13}
            thickness={7}
            variant="determinate"
            value={(doneCount / stages.length) * 100}
            sx={{ color: "inherit", ml: "6px" }}
          />
        }
        label={`${statusLabel(job.status)} ${currentStageNumber}/${stages.length}`}
        sx={{ mt: 1, animation: `${pulse} 1.4s ease-in-out infinite` }}
      />
    </Tooltip>
  );
}
