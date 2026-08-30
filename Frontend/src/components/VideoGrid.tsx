import { useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Grid from "@mui/material/Grid";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutlined";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { formatTimestamp } from "./results/types";
import { JobStatusChip } from "./JobStatusChip";

function formatUploadDate(isoDate: string): string {
  return new Date(isoDate).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

// A second chip next to the video-analysis status one, shown only once a
// job is complete and something still needs a human's attention - naming
// detected players and/or reviewing uncertain scoring calls (see
// Job.needs_player_id/needs_scoring_review, computed server-side in
// jobs_router._job_out so the grid doesn't have to fetch each video's
// results/score separately just to know this).
function SetupNeededChip({ job }: { job: Job }) {
  if (job.status !== "complete" || (!job.needs_player_id && !job.needs_scoring_review)) return null;

  const reasons = [
    job.needs_player_id && "player identification",
    job.needs_scoring_review && "game scoring",
  ]
    .filter(Boolean)
    .join(" and ");

  return (
    <Tooltip title={`Needs ${reasons}`}>
      <Chip size="small" color="warning" variant="outlined" icon={<WarningAmberIcon />} label="Setup needed" sx={{ mt: 1 }} />
    </Tooltip>
  );
}

interface VideoGridProps {
  jobs: Job[];
  onSelectJob: (jobId: string) => void;
  /** Omit for a read-only grid (e.g. Home's "recent videos" preview) - only
   * pass this where deleting a video should actually be offered. */
  onDeleteJob?: (jobId: string) => void;
  /** Omit to leave out the "add video" tile entirely. */
  onAddVideo?: () => void;
}

function AddVideoCard({ onAddVideo }: { onAddVideo: () => void }) {
  return (
    <Card variant="outlined" sx={{ overflow: "hidden", borderStyle: "dashed" }}>
      <CardActionArea onClick={onAddVideo}>
        <Box
          sx={{
            width: "100%",
            aspectRatio: "16 / 9",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 0.5,
            color: "text.secondary",
          }}
        >
          <AddIcon fontSize="large" />
          <Typography sx={{ fontWeight: 600 }}>Add video</Typography>
        </Box>
      </CardActionArea>
    </Card>
  );
}

// Shared by HomePage's "recent videos" preview and the full VideosPage grid
// - the only real difference between the two is which jobs get looked up
// and passed in (a recent slice vs. the full list) and whether deleting is
// offered, both handled via props rather than two copies of this markup.
// "Add video" is a tile in the grid itself (always first) rather than a
// separate button above it, so adding one reads as just another item in
// the same list instead of a distinct page-level action.
export function VideoGrid({ jobs, onSelectJob, onDeleteJob, onAddVideo }: VideoGridProps) {
  const [deleteTarget, setDeleteTarget] = useState<Job | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function handleConfirmDelete() {
    if (!deleteTarget || !onDeleteJob) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.deleteJob(deleteTarget.id);
      onDeleteJob(deleteTarget.id);
      setDeleteTarget(null);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeleting(false);
    }
  }

  if (jobs.length === 0 && !onAddVideo) {
    return <Typography color="text.secondary">No videos yet - add one to get started.</Typography>;
  }

  return (
    <>
      <Grid container spacing={2}>
        {onAddVideo && (
          <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
            <AddVideoCard onAddVideo={onAddVideo} />
          </Grid>
        )}
        {jobs.map((job) => (
          <Grid key={job.id} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
            <Card variant="outlined" sx={{ position: "relative", overflow: "hidden" }}>
              {onDeleteJob && (
                <IconButton
                  size="small"
                  aria-label={`Delete ${job.original_filename}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    setDeleteTarget(job);
                  }}
                  sx={{
                    position: "absolute",
                    top: 6,
                    right: 6,
                    zIndex: 1,
                    color: "error.main",
                    bgcolor: "rgba(0,0,0,0.4)",
                    "&:hover": { bgcolor: "rgba(0,0,0,0.6)" },
                  }}
                >
                  <DeleteOutlineIcon fontSize="small" />
                </IconButton>
              )}
              {job.winner_team_name && (
                <Chip
                  size="small"
                  icon={<EmojiEventsIcon sx={{ fontSize: 16, color: "#facc15 !important" }} />}
                  label={`${job.winner_team_name} Win`}
                  sx={{
                    position: "absolute",
                    top: 6,
                    left: 6,
                    zIndex: 1,
                    bgcolor: "rgba(0,0,0,0.55)",
                    color: "#fff",
                    fontWeight: 600,
                  }}
                />
              )}
              {job.duration_s != null && (
                <Box
                  sx={{
                    position: "absolute",
                    bottom: 6,
                    right: 6,
                    zIndex: 1,
                    bgcolor: "rgba(0,0,0,0.7)",
                    color: "#fff",
                    px: 0.75,
                    py: 0.25,
                    borderRadius: 0.5,
                    fontSize: 12,
                    fontWeight: 600,
                    lineHeight: 1.4,
                  }}
                >
                  {formatTimestamp(job.duration_s)}
                </Box>
              )}
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
                    <Tooltip title={job.original_filename}>
                      <Typography noWrap sx={{ fontWeight: 600, color: "#fff" }}>
                        {formatUploadDate(job.created_at)}
                      </Typography>
                    </Tooltip>
                    <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                      <JobStatusChip job={job} />
                      <SetupNeededChip job={job} />
                    </Stack>
                  </Box>
                </Box>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>

      {onDeleteJob && (
        <Dialog open={deleteTarget !== null} onClose={() => setDeleteTarget(null)}>
          <DialogTitle>Delete this video?</DialogTitle>
          <DialogContent>
            <DialogContentText>
              This permanently deletes "{deleteTarget?.original_filename}" and everything generated
              from it - tracking data, stats, the annotated video, and the dashboard. This can't be
              undone.
            </DialogContentText>
            {deleteError && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {deleteError}
              </Alert>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteTarget(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button color="error" variant="contained" onClick={handleConfirmDelete} disabled={deleting}>
              {deleting ? "Deleting..." : "Delete"}
            </Button>
          </DialogActions>
        </Dialog>
      )}
    </>
  );
}
