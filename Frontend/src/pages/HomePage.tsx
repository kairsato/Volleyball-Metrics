import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import AddIcon from "@mui/icons-material/Add";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { VideoGrid } from "../components/VideoGrid";

const RUNNING_STATUSES: Job["status"][] = ["processing", "finalizing", "awaiting_player_review"];
const ATTENTION_STATUSES: Job["status"][] = ["error", "cancelled"];
const RECENT_COUNT = 4;

interface AsyncStats {
  totalHits: number;
  totalPlayers: number;
}

function StatTile({ value, label }: { value: number; label: string }) {
  return (
    <Grid size={{ xs: 6, sm: 4, md: 2 }}>
      <Card variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          {value}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
      </Card>
    </Grid>
  );
}

interface HomePageProps {
  jobs: Job[];
  onSelectJob: (jobId: string) => void;
  onAddVideo: () => void;
  onViewVideos: () => void;
}

export function HomePage({ jobs, onSelectJob, onAddVideo, onViewVideos }: HomePageProps) {
  const [asyncStats, setAsyncStats] = useState<AsyncStats>({ totalHits: 0, totalPlayers: 0 });

  const completed = jobs.filter((j) => j.status === "complete").length;
  const inProgress = jobs.filter((j) => RUNNING_STATUSES.includes(j.status)).length;
  const needsAttention = jobs.filter((j) => ATTENTION_STATUSES.includes(j.status)).length;
  const recent = jobs.slice(0, RECENT_COUNT);

  useEffect(() => {
    let cancelled = false;
    const completedJobs = jobs.filter((j) => j.status === "complete");
    if (completedJobs.length === 0) return;

    Promise.all(completedJobs.map((j) => api.getResults(j.id).catch(() => null))).then((results) => {
      if (cancelled) return;
      let totalHits = 0;
      let totalPlayers = 0;
      for (const result of results) {
        if (!result) continue;
        const players = Object.values(result.players);
        totalPlayers += players.length;
        totalHits += players.reduce((sum, p) => sum + p.total_hits, 0);
      }
      setAsyncStats({ totalHits, totalPlayers });
    });

    return () => {
      cancelled = true;
    };
  }, [jobs]);

  return (
    <Box>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Home
        </Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={onAddVideo}>
          New video
        </Button>
      </Stack>

      <Grid container spacing={2} sx={{ mb: 5 }}>
        <StatTile value={jobs.length} label="Videos" />
        <StatTile value={completed} label="Completed" />
        <StatTile value={inProgress} label="In progress" />
        <StatTile value={needsAttention} label="Needs attention" />
        <StatTile value={asyncStats.totalHits} label="Hits recorded" />
        <StatTile value={asyncStats.totalPlayers} label="Players tracked" />
      </Grid>

      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 600 }}>
          Recent videos
        </Typography>
        {jobs.length > 0 && <Button onClick={onViewVideos}>View all in Videos</Button>}
      </Stack>

      <VideoGrid jobs={recent} onSelectJob={onSelectJob} />
    </Box>
  );
}
