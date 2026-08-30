import { useEffect, useState } from "react";

// Court calibration used to run up front, before any of these, so it was
// listed here as the first stage - it's now a post-processing Setup tab
// step (see CourtCalibrationPage.tsx) that doesn't run as part of
// processing at all, so it's no longer one of the stages this list is
// tracking progress through.
export const PHASE_ONE_STAGES = ["player_tracking", "ball_detection", "game_status", "action_detection"];

export const PHASE_TWO_STAGES = ["consolidating", "dashboard", "rendering"];

export const STAGE_LABELS: Record<string, string> = {
  player_tracking: "Player tracking",
  ball_detection: "Ball detection",
  game_status: "Game status detection",
  action_detection: "Action detection",
  consolidating: "Consolidating stats",
  dashboard: "Generating dashboard",
  rendering: "Rendering annotated video",
};

export function phaseOneComplete(completedStages: string[]): boolean {
  return PHASE_ONE_STAGES.every((stage) => completedStages.includes(stage));
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

// Ticks once a second so a currently-running stage can show a live elapsed
// time instead of just a spinner. `since` is the job's updated_at - stage
// transitions touch that field right as the new stage starts (and nothing
// else updates it mid-stage), so it doubles as a good-enough start time
// without the backend needing to track a separate stage-start timestamp.
export function useElapsedSeconds(since: string | null): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!since) return;
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [since]);

  if (!since) return 0;
  return Math.max(0, (now - new Date(since).getTime()) / 1000);
}
