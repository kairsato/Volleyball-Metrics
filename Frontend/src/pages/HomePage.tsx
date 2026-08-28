import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Grid from "@mui/material/Grid";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import AddIcon from "@mui/icons-material/Add";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { JobStatusChip } from "../components/JobStatusChip";

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
  onViewPlaylist: () => void;
}

export function HomePage({ jobs, onSelectJob, onAddVideo, onViewPlaylist }: HomePageProps) {
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
        {jobs.length > 0 && <Button onClick={onViewPlaylist}>View all in Playlist</Button>}
      </Stack>

      {recent.length === 0 ? (
        <Typography color="text.secondary">No videos yet - add one to get started.</Typography>
      ) : (
        <Grid container spacing={2}>
          {recent.map((job) => (
            <Grid key={job.id} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ overflow: "hidden" }}>
                <CardActionArea onClick={() => onSelectJob(job.id)}>
                  <Box
                    sx={{
                      width: "100%",
                      aspectRatio: "16 / 9",
                      display: "flex",
                      alignItems: "flex-end",
                      backgroundColor: "action.hover",
                      backgroundImage: `url(${api.thumbnailUrl(job.id)})`,
                      backgroundSize: "cover",
                      backgroundPosition: "center",
                    }}
                  >
                    <Box
                      sx={{
                        width: "100%",
                        p: 1.5,
                        background: "linear-gradient(to top, rgba(0,0,0,0.8), rgba(0,0,0,0) 75%)",
                      }}
                    >
                      <Typography noWrap sx={{ fontWeight: 600, color: "#fff" }}>
                        {job.original_filename}
                      </Typography>
                      <JobStatusChip job={job} />
                    </Box>
                  </Box>
                </CardActionArea>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
}
