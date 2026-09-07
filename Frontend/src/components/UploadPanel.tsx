import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import IconButton from "@mui/material/IconButton";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import CloseIcon from "@mui/icons-material/Close";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import LibraryAddIcon from "@mui/icons-material/LibraryAdd";
import MovieIcon from "@mui/icons-material/Movie";
import { api } from "../lib/api";
import type { Job } from "../lib/types";

interface UploadPanelProps {
  onUploaded: (job: Job) => void;
  /** Drops the page-level heading, for use inside a dialog that already has its own title. */
  embedded?: boolean;
}

// A staged file paired with a stable key and a blob: preview URL - both
// generated once at stage time (not derived from name/index), since two
// different files can share a name, and reordering must not shuffle which
// preview belongs to which file. previewUrl must be revoked wherever a
// StagedVideo stops being staged (removed, cleared, uploaded, or the whole
// panel unmounts) or it leaks the underlying blob for the tab's lifetime.
interface StagedVideo {
  key: string;
  file: File;
  previewUrl: string;
}

function toStaged(files: FileList): StagedVideo[] {
  return Array.from(files).map((file) => ({
    key: typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${file.name}-${file.size}-${Math.random()}`,
    file,
    previewUrl: URL.createObjectURL(file),
  }));
}

// Multiple videos for one job are joined into a single continuous match
// (see the multi-video plan) - so there's a review step between picking
// files and actually uploading (an explicit "Upload N videos" click,
// reorderable into match order with a preview of each) rather than
// uploading the instant a file is chosen, the way a single video always
// has. Each file's date-played default (its own recording metadata, or
// today's date) is computed server-side once it's actually on disk - see
// jobs_router.upload_video - and can be corrected afterward via
// api.setVideoDatePlayed, not from this dialog.
export function UploadPanel({ onUploaded, embedded }: UploadPanelProps) {
  const [dragging, setDragging] = useState(false);
  const [staged, setStaged] = useState<StagedVideo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Revokes whatever's still staged when the panel goes away without
  // uploading (dialog closed, navigated off) - reads the latest `staged`
  // via a ref since an unmount cleanup only ever sees the value from the
  // render it was defined in otherwise.
  const stagedRef = useRef<StagedVideo[]>(staged);
  stagedRef.current = staged;
  useEffect(() => {
    return () => {
      for (const s of stagedRef.current) URL.revokeObjectURL(s.previewUrl);
    };
  }, []);

  function stageFiles(list: FileList | null) {
    if (!list || list.length === 0) return;
    // Snapshotted here, eagerly, rather than inside the setStaged updater
    // below - a FileList is live and tied to the <input> element, and
    // handleChange clears it (event.target.value = "") synchronously right
    // after calling this. A functional updater's callback isn't
    // guaranteed to run synchronously (React 18 dev mode deliberately
    // double-invokes it to catch exactly this kind of impurity), so
    // reading the live list from inside it is a race - by the time it
    // actually runs, the list can already be empty. Converting to a plain
    // array up front removes any dependency on the list still being valid
    // whenever the updater happens to execute.
    const additions = toStaged(list);
    setStaged((prev) => [...prev, ...additions]);
    setError(null);
  }

  function removeStaged(key: string) {
    setStaged((prev) => {
      const target = prev.find((s) => s.key === key);
      if (target) URL.revokeObjectURL(target.previewUrl);
      return prev.filter((s) => s.key !== key);
    });
  }

  function moveStaged(key: string, direction: -1 | 1) {
    setStaged((prev) => {
      const index = prev.findIndex((s) => s.key === key);
      const target = index + direction;
      if (index < 0 || target < 0 || target >= prev.length) return prev;
      const next = prev.slice();
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  function clearStaged() {
    for (const s of staged) URL.revokeObjectURL(s.previewUrl);
    setStaged([]);
  }

  async function handleUpload() {
    if (staged.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const job = await api.uploadVideos(staged.map((s) => s.file));
      for (const s of staged) URL.revokeObjectURL(s.previewUrl);
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
    if (!uploading) stageFiles(event.dataTransfer.files);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    stageFiles(event.target.files);
    event.target.value = "";
  }

  const multiple = staged.length > 1;

  return (
    <Box sx={{ maxWidth: embedded ? "none" : 640 }}>
      {!embedded && (
        <>
          <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
            Volleyball Video Analytics
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 2 }}>
            Upload a match video to track players, detect rallies and actions, then verify
            players and assign names before generating the final stats.
          </Typography>
        </>
      )}

      {/* Always visible, not just a hint inside the dropzone - a match
          split across several files (camera restart, battery/card swap) is
          a real, easy-to-miss case worth calling out up front rather than
          only discovering multi-select works after the fact. */}
      <Alert icon={<LibraryAddIcon fontSize="inherit" />} severity="info" variant="outlined" sx={{ mb: 2 }}>
        Recording split across multiple files? Select (or drag in) all of them at once - they'll
        be joined into one continuous match, in the order you arrange them below.
      </Alert>

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
          multiple
          hidden
          onChange={handleChange}
          disabled={uploading}
        />
        {uploading ? (
          <CircularProgress size={28} />
        ) : (
          <>
            <MovieIcon sx={{ fontSize: 36, color: "text.secondary", mb: 1 }} />
            <Typography sx={{ mb: 0.5 }}>
              Drop one or more videos here, or click to choose
            </Typography>
            <Typography variant="body2" color="text.secondary">
              MP4, MOV, AVI or MKV
            </Typography>
          </>
        )}
      </Paper>

      {staged.length > 0 && (
        <Box sx={{ mt: 2 }}>
          {multiple && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              These {staged.length} videos will be joined in this order - use the arrows to
              rearrange them:
            </Typography>
          )}
          <Stack spacing={1} sx={{ mb: 1.5 }}>
            {staged.map((item, index) => (
              <Paper
                key={item.key}
                variant="outlined"
                sx={{ p: 1, display: "flex", alignItems: "center", gap: 1.25 }}
              >
                {multiple && (
                  <Chip
                    size="small"
                    label={index + 1}
                    color="primary"
                    sx={{ fontWeight: 700, flexShrink: 0 }}
                  />
                )}
                <Box
                  component="video"
                  src={item.previewUrl}
                  controls
                  preload="metadata"
                  sx={{
                    width: 120,
                    height: 68,
                    borderRadius: 1,
                    bgcolor: "#000",
                    objectFit: "cover",
                    flexShrink: 0,
                  }}
                />
                <Box sx={{ minWidth: 0, flex: 1 }}>
                  <Typography variant="body2" noWrap title={item.file.name}>
                    {item.file.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {(item.file.size / (1024 * 1024)).toFixed(1)} MB
                  </Typography>
                </Box>
                {multiple && !uploading && (
                  <Stack sx={{ flexShrink: 0 }}>
                    <IconButton
                      size="small"
                      aria-label="Move earlier"
                      disabled={index === 0}
                      onClick={() => moveStaged(item.key, -1)}
                      sx={{ p: 0.25 }}
                    >
                      <KeyboardArrowUpIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      aria-label="Move later"
                      disabled={index === staged.length - 1}
                      onClick={() => moveStaged(item.key, 1)}
                      sx={{ p: 0.25 }}
                    >
                      <KeyboardArrowDownIcon fontSize="small" />
                    </IconButton>
                  </Stack>
                )}
                {!uploading && (
                  <IconButton
                    size="small"
                    aria-label="Remove"
                    onClick={() => removeStaged(item.key)}
                    sx={{ flexShrink: 0 }}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                )}
              </Paper>
            ))}
          </Stack>
          <Stack direction="row" spacing={1.5} sx={{ alignItems: "center" }}>
            <Button variant="contained" onClick={handleUpload} disabled={uploading}>
              {uploading
                ? "Uploading..."
                : `Upload ${staged.length} video${staged.length > 1 ? "s" : ""}`}
            </Button>
            {!uploading && (
              <Button onClick={clearStaged} color="inherit">
                Clear
              </Button>
            )}
          </Stack>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
}
