import { useEffect, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import type { Job, WarmupConfig } from "../lib/types";
import { formatTimestamp } from "./results/types";
import { LockOverlay } from "./LockOverlay";
import { VideoPlayer } from "./VideoPlayer";

interface WarmupPanelProps {
  job: Job;
  // Fired once a range is actually saved - lets the caller
  // (WarmupPeriodPage) refresh its own job state so the rest of the app
  // (video player, thumbnails, rallies/stats) picks up the new bounds.
  onSaved?: () => void;
}

// The full, unrestricted source video - deliberately not clamped to any
// already-confirmed range, since this is exactly the tool used to set that
// range in the first place (see warmup.py's module docstring).
export function WarmupPanel({ job, onSaved }: WarmupPanelProps) {
  const [config, setConfig] = useState<WarmupConfig | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [redoing, setRedoing] = useState(false);
  const [redoDialogOpen, setRedoDialogOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getWarmup(job.id)
      .then((res) => {
        if (cancelled) return;
        setConfig(res);
        setRange([res.start_s, res.end_s ?? res.duration_s ?? 0]);
      })
      .catch((err) => !cancelled && setLoadError(err instanceof Error ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, [job.id]);

  if (loadError) return <Alert severity="error">{loadError}</Alert>;
  if (!config || !range) return null;

  const duration = config.duration_s ?? 0;
  const [startS, endS] = range;
  const locked = config.confirmed;

  async function handleSave() {
    setSaving(true);
    setSaveError(null);
    try {
      const result = await api.saveWarmup(job.id, startS, endS >= duration ? null : endS);
      setConfig(result);
      onSaved?.();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleRedoConfirm() {
    setRedoing(true);
    setSaveError(null);
    try {
      const result = await api.setWarmupConfirmed(job.id, false);
      setConfig(result);
      onSaved?.();
      setRedoDialogOpen(false);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setRedoing(false);
    }
  }

  function captureCurrentTimeFor(edge: "start" | "end") {
    const el = videoRef.current;
    if (!el) return;
    const t = el.currentTime;
    setRange((prev) => {
      const [s, e] = prev ?? [t, t];
      return edge === "start" ? [Math.min(t, e), e] : [s, Math.max(t, s)];
    });
  }

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
        Warmup period
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 2 }}>
        Watch or scrub the video below, then mark where the actual match starts and ends. Once set, the video
        player, thumbnails, and every rally/stat elsewhere are restricted to this window - time 0:00 becomes this
        window's start everywhere outside this page.
      </Typography>

      <Box sx={{ position: "relative", flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
        <LockOverlay active={locked} label="Redo Warmup Period to make changes" onClick={() => setRedoDialogOpen(true)} />

        <Box sx={{ flex: 1, minHeight: 0 }}>
          <VideoPlayer videoRef={videoRef} src={api.sourceVideoUrl(job.id)} />
        </Box>

        <Box sx={{ mt: 3, flexShrink: 0 }}>
          <Slider
            value={range}
            min={0}
            max={Math.max(duration, 1)}
            step={0.1}
            disabled={locked}
            // Dragging either thumb also seeks the player to that exact
            // position - activeThumb (0 = start, 1 = end) says which one is
            // being dragged, so the video previews whichever edge is
            // currently being marked instead of just sitting still while
            // the range slider moves on its own.
            onChange={(_, value, activeThumb) => {
              const [s, e] = value as [number, number];
              setRange([s, e]);
              const el = videoRef.current;
              if (el) el.currentTime = activeThumb === 1 ? e : s;
            }}
            valueLabelDisplay="auto"
            valueLabelFormat={formatTimestamp}
            disableSwap
          />

          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mt: 1 }}>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center", flex: 1 }}>
              <Typography variant="body2" sx={{ minWidth: 90 }}>
                Start: {formatTimestamp(startS)}
              </Typography>
              <Button size="small" disabled={locked} onClick={() => captureCurrentTimeFor("start")}>
                Use current time
              </Button>
            </Stack>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center", flex: 1 }}>
              <Typography variant="body2" sx={{ minWidth: 90 }}>
                End: {formatTimestamp(endS)}
              </Typography>
              <Button size="small" disabled={locked} onClick={() => captureCurrentTimeFor("end")}>
                Use current time
              </Button>
            </Stack>
          </Stack>

          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
            Kept range: {formatTimestamp(startS)} - {formatTimestamp(endS)} ({formatTimestamp(Math.max(0, endS - startS))} long)
          </Typography>

          {saveError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {saveError}
            </Alert>
          )}

          <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
            <Button
              variant="contained"
              disabled={saving || endS <= startS}
              onClick={() => (locked ? setRedoDialogOpen(true) : void handleSave())}
            >
              {saving ? "Saving..." : locked ? "Redo Warmup Period" : "Set Warmup Period"}
            </Button>
          </Stack>
        </Box>
      </Box>

      <Dialog open={redoDialogOpen} onClose={() => (redoing ? undefined : setRedoDialogOpen(false))}>
        <DialogTitle>Redo warmup period?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This unlocks the range for editing again - until you set a new range, the whole video and every rally/
            stat become visible again everywhere, same as if no warmup period were set at all.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRedoDialogOpen(false)} disabled={redoing}>
            Cancel
          </Button>
          <Button color="warning" variant="contained" onClick={() => void handleRedoConfirm()} disabled={redoing}>
            {redoing ? "Working..." : "Confirm"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
