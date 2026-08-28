import { useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Grid from "@mui/material/Grid";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutlined";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { JobStatusChip } from "../components/JobStatusChip";

interface PlaylistPageProps {
  jobs: Job[];
  onSelectJob: (jobId: string) => void;
  onAddVideo: () => void;
  onDeleteJob: (jobId: string) => void;
}

export function PlaylistPage({ jobs, onSelectJob, onAddVideo, onDeleteJob }: PlaylistPageProps) {
  const [deleteTarget, setDeleteTarget] = useState<Job | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function handleConfirmDelete() {
    if (!deleteTarget) return;
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

  return (
    <Box>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Playlist
        </Typography>
        <Button variant="contained" startIcon={<AddIcon />} onClick={onAddVideo}>
          New video
        </Button>
      </Stack>

      {jobs.length === 0 ? (
        <Typography color="text.secondary">No videos yet - add one to get started.</Typography>
      ) : (
        <Grid container spacing={2}>
          {jobs.map((job) => (
            <Grid key={job.id} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
              <Card variant="outlined" sx={{ position: "relative", overflow: "hidden" }}>
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
    </Box>
  );
}
