import { PlayerGroupingEditor } from "./PlayerGroupingEditor";
import type { Job } from "../lib/types";

interface PlayerReviewProps {
  job: Job;
  onFinalized: (job: Job) => void;
}

export function PlayerReview({ job, onFinalized }: PlayerReviewProps) {
  return (
    <PlayerGroupingEditor
      jobId={job.id}
      title="Verify players & assign names"
      description="Name each player, or leave blank to keep the numeric ID. If the same person shows up as more than one card - common after they're briefly hidden or step out of frame - tick their boxes and merge them into one."
      saveButtonLabel="Save names & finalize"
      onSaved={onFinalized}
    />
  );
}
