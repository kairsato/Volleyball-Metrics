import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import AddIcon from "@mui/icons-material/Add";
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
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Videos
        </Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={onAddVideo}>
          New video
        </Button>
      </Stack>

      <VideoGrid jobs={jobs} onSelectJob={onSelectJob} onDeleteJob={onDeleteJob} />
    </Box>
  );
}
