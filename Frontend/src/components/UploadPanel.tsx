import { useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import MovieIcon from "@mui/icons-material/Movie";
import { api } from "../lib/api";
import type { Job } from "../lib/types";

interface UploadPanelProps {
  onUploaded: (job: Job) => void;
  /** Drops the page-level heading, for use inside a dialog that already has its own title. */
  embedded?: boolean;
}

export function UploadPanel({ onUploaded, embedded }: UploadPanelProps) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function upload(file: File) {
    setUploading(true);
    setError(null);
    try {
      const job = await api.uploadVideo(file);
      onUploaded(job);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void upload(file);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void upload(file);
  }

  return (
    <Box sx={{ maxWidth: embedded ? "none" : 640 }}>
      {!embedded && (
        <>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
            Volleyball Video Analytics
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 3.5 }}>
            Upload a match video to track players, detect rallies and actions, then verify
            players and assign names before generating the final stats.
          </Typography>
        </>
      )}

      <Paper
        variant="outlined"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && inputRef.current?.click()}
        sx={{
          borderStyle: "dashed",
          borderWidth: 2,
          borderColor: dragging ? "primary.main" : "divider",
          bgcolor: dragging ? "action.hover" : "transparent",
          borderRadius: 3,
          py: 7,
          px: 3,
          textAlign: "center",
          cursor: uploading ? "default" : "pointer",
          opacity: uploading ? 0.7 : 1,
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.avi,.mkv"
          hidden
          onChange={handleChange}
          disabled={uploading}
        />
        {uploading ? (
          <CircularProgress size={28} />
        ) : (
          <>
            <MovieIcon sx={{ fontSize: 36, color: "text.secondary", mb: 1 }} />
            <Typography sx={{ mb: 0.5 }}>Drop a video here, or click to choose one</Typography>
            <Typography variant="body2" color="text.secondary">
              MP4, MOV, AVI or MKV
            </Typography>
          </>
        )}
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
}
