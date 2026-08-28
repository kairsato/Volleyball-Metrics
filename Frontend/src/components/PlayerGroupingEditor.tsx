import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Autocomplete from "@mui/material/Autocomplete";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Grid from "@mui/material/Grid";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import CloseIcon from "@mui/icons-material/Close";
import HelpOutlineIcon from "@mui/icons-material/HelpOutlined";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import VisibilityIcon from "@mui/icons-material/Visibility";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import { api } from "../lib/api";
import type { Job, Player } from "../lib/types";
import { LoadingSpinner } from "./LoadingSpinner";

interface PlayerGroupingEditorProps {
  jobId: string;
  title: string;
  description?: string;
  saveButtonLabel: string;
  /** When set, clicking save shows a confirmation dialog with this message first. */
  confirmBeforeSave?: string;
  onSaved: (job: Job) => void;
}

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// Union-find over stable_ids linked by candidate_matches (transitively - if
// A might be B and B might be C, all three show up as one cluster) so the
// UI can visually group "might be the same person" cards together instead
// of scattering them through the grid.
function clusterByCandidates(players: Player[], matches: { a: number; b: number }[]): Player[][] {
  const parent = new Map<number, number>();
  for (const p of players) parent.set(p.stable_id, p.stable_id);

  function find(x: number): number {
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root)!;
    while (parent.get(x) !== root) {
      const next = parent.get(x)!;
      parent.set(x, root);
      x = next;
    }
    return root;
  }

  for (const { a, b } of matches) {
    if (!parent.has(a) || !parent.has(b)) continue;
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  }

  const byRoot = new Map<number, Player[]>();
  for (const p of players) {
    const root = find(p.stable_id);
    if (!byRoot.has(root)) byRoot.set(root, []);
    byRoot.get(root)!.push(p);
  }

  return Array.from(byRoot.values())
    .map((group) => group.sort((a, b) => a.stable_id - b.stable_id))
    .sort((a, b) => a[0].stable_id - b[0].stable_id);
}

interface PlayerCardProps {
  player: Player;
  name: string;
  ignored: boolean;
  roster: string[];
  onNameChange: (name: string) => void;
  onToggleIgnore: () => void;
  onPreview: () => void;
}

function PlayerCard({ player, name, ignored, roster, onNameChange, onToggleIgnore, onPreview }: PlayerCardProps) {
  return (
    <Card variant="outlined" sx={{ p: 1.5, opacity: ignored ? 0.55 : 1, height: "100%" }}>
      <Box
        onClick={onPreview}
        sx={{
          position: "relative",
          width: "100%",
          aspectRatio: "3 / 4",
          borderRadius: 2,
          overflow: "hidden",
          bgcolor: "action.hover",
          mb: 1.5,
          cursor: "pointer",
          "&:hover .preview-hint": { opacity: 1 },
        }}
      >
        {player.thumbnail_base64 ? (
          <Box
            component="img"
            src={`data:image/jpeg;base64,${player.thumbnail_base64}`}
            alt={`Player ${player.stable_id}`}
            sx={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        ) : (
          <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Typography variant="caption" color="text.secondary">
              No thumbnail
            </Typography>
          </Box>
        )}
        <Box
          className="preview-hint"
          sx={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: "rgba(0,0,0,0.35)",
            opacity: 0,
            transition: "opacity 0.15s ease",
          }}
        >
          <ZoomInIcon sx={{ color: "#fff", fontSize: 32 }} />
        </Box>
      </Box>

      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
        #{player.stable_id}
      </Typography>

      <Autocomplete
        fullWidth
        size="small"
        disabled={ignored}
        options={roster}
        value={name || null}
        onChange={(_, value) => onNameChange(value ?? "")}
        renderInput={(params) => <TextField {...params} placeholder="Unassigned" />}
        sx={{ mb: 1 }}
      />

      <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap" }}>
        <Button
          size="small"
          color="inherit"
          startIcon={ignored ? <VisibilityIcon /> : <VisibilityOffIcon />}
          onClick={onToggleIgnore}
        >
          {ignored ? "Un-ignore" : "Ignore"}
        </Button>
        {!ignored && !name && <Chip size="small" label="Needs a name" color="warning" variant="outlined" />}
      </Stack>
    </Card>
  );
}

export function PlayerGroupingEditor({
  jobId,
  title,
  description,
  saveButtonLabel,
  confirmBeforeSave,
  onSaved,
}: PlayerGroupingEditorProps) {
  const [players, setPlayers] = useState<Player[] | null>(null);
  const [candidateMatches, setCandidateMatches] = useState<{ a: number; b: number }[]>([]);
  const [roster, setRoster] = useState<string[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [ignoredIds, setIgnoredIds] = useState<Set<number>>(new Set());
  const [previewPlayer, setPreviewPlayer] = useState<Player | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;

    api
      .getPlayers(jobId)
      .then((res) => {
        if (cancelled) return;
        setPlayers(res.players);
        setCandidateMatches(res.candidate_matches);
        setNames(Object.fromEntries(res.players.map((p) => [String(p.stable_id), p.name ?? ""])));
        setIgnoredIds(new Set(res.players.filter((p) => p.ignored).map((p) => p.stable_id)));
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : String(err)));

    return () => {
      cancelled = true;
    };
  }, [jobId]);

  useEffect(() => {
    api
      .getRoster()
      .then((res) => setRoster(res.players))
      .catch(() => undefined);
  }, []);

  const clusters = useMemo(() => clusterByCandidates(players ?? [], candidateMatches), [players, candidateMatches]);

  function toggleIgnored(stableId: number) {
    setIgnoredIds((prev) => {
      const next = new Set(prev);
      if (next.has(stableId)) next.delete(stableId);
      else next.add(stableId);
      return next;
    });
  }

  async function performSave() {
    if (!players) return;
    setSaving(true);
    setError(null);
    try {
      await api.updateNames(jobId, names, Array.from(ignoredIds));
      const updated = await api.finalizeJob(jobId);
      onSaved(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  function handleSaveClick() {
    if (confirmBeforeSave) setConfirmOpen(true);
    else void performSave();
  }

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!players) return <LoadingSpinner />;

  return (
    <Box sx={{ maxWidth: 1100 }}>
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
        {title}
      </Typography>
      {description && (
        <Typography color="text.secondary" sx={{ mb: 2 }}>
          {players.length === 0 ? "No players were detected in this video." : description}
        </Typography>
      )}

      {players.length > 0 && (
        <Grid container spacing={2}>
          {clusters.map((cluster) => {
            const cards = cluster.map((player) => (
              <Grid key={player.stable_id} size={{ xs: 6, sm: 4, md: 3 }}>
                <PlayerCard
                  player={player}
                  name={names[String(player.stable_id)] ?? ""}
                  ignored={ignoredIds.has(player.stable_id)}
                  roster={roster}
                  onNameChange={(name) => setNames((prev) => ({ ...prev, [String(player.stable_id)]: name }))}
                  onToggleIgnore={() => toggleIgnored(player.stable_id)}
                  onPreview={() => setPreviewPlayer(player)}
                />
              </Grid>
            ));

            if (cluster.length === 1) return cards;

            return (
              <Grid key={`cluster-${cluster[0].stable_id}`} size={12}>
                <Box sx={{ border: "1px dashed", borderColor: "warning.main", borderRadius: 2, p: 2 }}>
                  <Stack direction="row" spacing={0.75} sx={{ alignItems: "center", mb: 1.5 }}>
                    <HelpOutlineIcon fontSize="small" color="warning" />
                    <Typography variant="body2" color="warning.main">
                      These might be the same person - give them the same name if so.
                    </Typography>
                  </Stack>
                  <Grid container spacing={2}>
                    {cards}
                  </Grid>
                </Box>
              </Grid>
            );
          })}
        </Grid>
      )}

      <Button variant="contained" disabled={saving} onClick={handleSaveClick} sx={{ mt: 3 }}>
        {saving ? "Saving..." : saveButtonLabel}
      </Button>

      <Dialog open={previewPlayer !== null} onClose={() => setPreviewPlayer(null)} maxWidth="sm" fullWidth>
        {previewPlayer && (
          <>
            <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              Player #{previewPlayer.stable_id}
              <IconButton onClick={() => setPreviewPlayer(null)} size="small">
                <CloseIcon />
              </IconButton>
            </DialogTitle>
            <DialogContent>
              {previewPlayer.thumbnail_base64 ? (
                <Box
                  component="img"
                  src={`data:image/jpeg;base64,${previewPlayer.thumbnail_base64}`}
                  alt={`Player ${previewPlayer.stable_id}`}
                  sx={{ width: "100%", borderRadius: 2, display: "block" }}
                />
              ) : (
                <Typography color="text.secondary">No thumbnail available.</Typography>
              )}
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
                {previewPlayer.thumbnail_frame_idx != null
                  ? `Taken from frame ${previewPlayer.thumbnail_frame_idx}` +
                    (previewPlayer.thumbnail_timestamp_s != null
                      ? ` (${formatTimestamp(previewPlayer.thumbnail_timestamp_s)})`
                      : "")
                  : "Source frame unknown."}
              </Typography>
            </DialogContent>
          </>
        )}
      </Dialog>

      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)}>
        <DialogTitle>Save these changes?</DialogTitle>
        <DialogContent>
          <DialogContentText>{confirmBeforeSave}</DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => {
              setConfirmOpen(false);
              void performSave();
            }}
          >
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
