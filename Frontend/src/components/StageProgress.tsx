import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Stepper from "@mui/material/Stepper";
import Typography from "@mui/material/Typography";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import { formatDuration, STAGE_LABELS, useElapsedSeconds } from "../lib/stages";

interface StageProgressProps {
  title: string;
  note?: string;
  stages: string[];
  completedStages: string[];
  currentStage: string | null;
  stageDurations: Record<string, number>;
  stageStartedAt: string;
  onCancel?: () => void;
  cancelling?: boolean;
}

export function StageProgress({
  title,
  note,
  stages,
  completedStages,
  currentStage,
  stageDurations,
  stageStartedAt,
  onCancel,
  cancelling,
}: StageProgressProps) {
  const elapsed = useElapsedSeconds(currentStage ? stageStartedAt : null);
  const doneCount = stages.filter((s) => completedStages.includes(s)).length;
  const firstIncomplete = stages.findIndex((s) => !completedStages.includes(s));

  return (
    <Box sx={{ maxWidth: 480 }}>
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 0.5 }}>
        {title}
      </Typography>
      {note && (
        <Typography color="text.secondary" sx={{ mb: 2 }}>
          {note}
        </Typography>
      )}
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {doneCount} of {stages.length} stages complete
      </Typography>

      <Stepper activeStep={firstIncomplete === -1 ? stages.length : firstIncomplete} orientation="vertical">
        {stages.map((stage) => {
          const done = completedStages.includes(stage);
          const active = !done && stage === currentStage;
          const duration = stageDurations[stage];

          return (
            <Step key={stage} completed={done}>
              <StepLabel
                icon={
                  done ? (
                    <CheckCircleIcon color="success" fontSize="small" />
                  ) : active ? (
                    <CircularProgress size={16} thickness={6} />
                  ) : (
                    <RadioButtonUncheckedIcon color="disabled" fontSize="small" />
                  )
                }
              >
                {STAGE_LABELS[stage] ?? stage}
                {done && duration !== undefined && (
                  <Typography component="span" color="text.secondary" sx={{ ml: 1 }}>
                    ({formatDuration(duration)})
                  </Typography>
                )}
                {active && (
                  <Typography component="span" color="text.secondary" sx={{ ml: 1 }}>
                    (running {formatDuration(elapsed)})
                  </Typography>
                )}
              </StepLabel>
            </Step>
          );
        })}
      </Stepper>

      {onCancel && (
        <Button color="error" variant="outlined" disabled={cancelling} onClick={onCancel} sx={{ mt: 2 }}>
          {cancelling ? "Cancelling..." : "Cancel"}
        </Button>
      )}
    </Box>
  );
}
