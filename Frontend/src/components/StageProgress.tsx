import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CircularProgress from "@mui/material/CircularProgress";
import LinearProgress from "@mui/material/LinearProgress";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Stepper from "@mui/material/Stepper";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import HourglassTopIcon from "@mui/icons-material/HourglassTop";
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
  // 1-based position in the pipeline queue while this job is still
  // waiting for a worker (see pipeline.queue_position) - only ever set
  // alongside an empty stepper (nothing's completed, nothing's active
  // yet), so a queued job doesn't read as simply stuck.
  queuePosition?: number | null;
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
  queuePosition,
}: StageProgressProps) {
  const elapsed = useElapsedSeconds(currentStage ? stageStartedAt : null);
  const doneCount = stages.filter((s) => completedStages.includes(s)).length;
  const firstIncomplete = stages.findIndex((s) => !completedStages.includes(s));
  const percent = Math.round((doneCount / stages.length) * 100);
  // Sum of every completed stage's own duration, plus however long the
  // currently-running one (if any) has been going - a running total for
  // the whole phase, not just whichever single stage happens to be active
  // right now.
  const totalElapsed = stages.reduce((sum, s) => sum + (stageDurations[s] ?? 0), 0) + (currentStage ? elapsed : 0);

  return (
    <Box sx={{ display: "flex", justifyContent: "center", pt: 4 }}>
      <Card variant="outlined" sx={{ p: 4, width: "100%", maxWidth: 520 }}>
        <Stack direction="row" sx={{ alignItems: "baseline", justifyContent: "space-between", mb: 0.5 }}>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            {title}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {percent}%
          </Typography>
        </Stack>
        {note && (
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            {note}
          </Typography>
        )}

        {queuePosition && (
          <Alert severity="info" icon={<HourglassTopIcon fontSize="inherit" />} sx={{ mb: 2 }}>
            Queued - position #{queuePosition} in line, waiting for another video to finish processing
            first. This starts on its own once it's this video's turn.
          </Alert>
        )}

        <LinearProgress variant="determinate" value={percent} sx={{ mb: 1, height: 8, borderRadius: 999 }} />
        <Stack direction="row" sx={{ justifyContent: "space-between", mb: 3 }}>
          <Typography variant="body2" color="text.secondary">
            {doneCount} of {stages.length} stages complete
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Total: {formatDuration(totalElapsed)}
          </Typography>
        </Stack>

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
          <Button color="error" variant="outlined" disabled={cancelling} onClick={onCancel} sx={{ mt: 3 }}>
            {cancelling ? "Cancelling..." : "Cancel"}
          </Button>
        )}
      </Card>
    </Box>
  );
}
