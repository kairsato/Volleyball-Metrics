import Box from "@mui/material/Box";
import type { Job } from "../lib/types";
import { VideoGrid } from "../components/VideoGrid";

interface VideosPageProps {
  jobs: Job[];
  onSelectJob: (jobId: string) => void;
  onAddVideo: () => void;
  onDeleteJob: (jobId: string) => void;
}

export function VideosPage({ jobs, onSelectJob, onAddVideo, onDeleteJob }: VideosPageProps) {
  return (
    <Box>
      <VideoGrid jobs={jobs} onSelectJob={onSelectJob} onDeleteJob={onDeleteJob} onAddVideo={onAddVideo} />
    </Box>
  );
}
