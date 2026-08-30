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
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircle";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import VisibilityIcon from "@mui/icons-material/Visibility";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import { api } from "../lib/api";
import type { CandidateMatch, Job, Player } from "../lib/types";
import { LoadingSpinner } from "./LoadingSpinner";

interface PlayerGroupingEditorProps {
  jobId: string;
  title: string;
  description?: string;
  saveButtonLabel: string;
  /** When set, clicking save shows a confirmation dialog with this message first. */
  confirmBeforeSave?: string;
  /** When set, the timestamp shown in the preview dialog becomes clickable
   * and seeks the (externally rendered) video player there - omitted where
   * there's no video player nearby to seek, e.g. the pre-processing review. */
  onSeek?: (timeS: number, pause?: boolean) => void;
  onSaved: (job: Job) => void;
}

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// Only pairs at least this confident get the visible "might be the same
// person" dashed-box treatment - the backend already limits what it sends
// to appearance matches worth mentioning at all, but that bar alone still
// let dozens of pairs "have some resemblance" without genuinely looking
// like the same person. Pairs below this bar still influence card ORDER
// (see arrangePlayers) so related-looking players end up near each other
// even when they're not confident enough to flag outright.
const GROUP_DISPLAY_MIN_CONFIDENCE = 0.65;

function buildUnionFind(ids: number[], pairs: { a: number; b: number }[]): (id: number) => number {
  const parent = new Map<number, number>();
  for (const id of ids) parent.set(id, id);

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

  for (const { a, b } of pairs) {
    if (!parent.has(a) || !parent.has(b)) continue;
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  }

  return find;
}

// Arranges players into display chunks: consecutive cards sharing a
// high-confidence candidate match are grouped into one "might be the same
// person" box; everything else renders as its own standalone card. The
// overall ORDER (which chunk comes before which, and where an unboxed card
// sits) is instead driven by every candidate match regardless of
// confidence, so a pair too uncertain to box up still ends up positioned
// next to each other - a softer "these might be worth comparing" hint via
// proximity instead of a false-confidence box.
function arrangePlayers(players: Player[], matches: CandidateMatch[]): Player[][] {
  const ids = players.map((p) => p.stable_id);
  const findOrderRoot = buildUnionFind(ids, matches);
  const findDisplayRoot = buildUnionFind(
    ids,
    matches.filter((m) => m.confidence >= GROUP_DISPLAY_MIN_CONFIDENCE),
  );

  const groupMinId = new Map<number, number>();
  for (const p of players) {
    const root = findOrderRoot(p.stable_id);
    groupMinId.set(root, Math.min(groupMinId.get(root) ?? Infinity, p.stable_id));
  }

  const sorted = [...players].sort((a, b) => {
    const ga = groupMinId.get(findOrderRoot(a.stable_id))!;
    const gb = groupMinId.get(findOrderRoot(b.stable_id))!;
    return ga !== gb ? ga - gb : a.stable_id - b.stable_id;
  });

  const chunks: Player[][] = [];
  for (const p of sorted) {
    const root = findDisplayRoot(p.stable_id);
    const last = chunks[chunks.length - 1];
    if (last && findDisplayRoot(last[0].stable_id) === root) {
      last.push(p);
    } else {
      chunks.push([p]);
    }
  }
  return chunks;
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
  onSeek,
  onSaved,
}: PlayerGroupingEditorProps) {
  const [players, setPlayers] = useState<Player[] | null>(null);
  const [candidateMatches, setCandidateMatches] = useState<CandidateMatch[]>([]);
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

  const clusters = useMemo(() => arrangePlayers(players ?? [], candidateMatches), [players, candidateMatches]);

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
              <Box sx={{ width: "100%", height: 420, borderRadius: 2, overflow: "hidden", bgcolor: "action.hover" }}>
                {previewPlayer.thumbnail_base64 ? (
                  <Box
                    component="img"
                    src={`data:image/jpeg;base64,${previewPlayer.thumbnail_base64}`}
                    alt={`Player ${previewPlayer.stable_id}`}
                    sx={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
                  />
                ) : (
                  <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <Typography color="text.secondary">No thumbnail available.</Typography>
                  </Box>
                )}
              </Box>

              {previewPlayer.thumbnail_frame_idx != null ? (
                onSeek && previewPlayer.thumbnail_timestamp_s != null ? (
                  <Button
                    size="small"
                    startIcon={<PlayCircleOutlineIcon />}
                    sx={{ mt: 1.5 }}
                    onClick={() => {
                      onSeek(previewPlayer.thumbnail_timestamp_s!, true);
                      setPreviewPlayer(null);
                    }}
                  >
                    Frame {previewPlayer.thumbnail_frame_idx} ({formatTimestamp(previewPlayer.thumbnail_timestamp_s)}) - jump to it
                  </Button>
                ) : (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
                    Taken from frame {previewPlayer.thumbnail_frame_idx}
                    {previewPlayer.thumbnail_timestamp_s != null && ` (${formatTimestamp(previewPlayer.thumbnail_timestamp_s)})`}
                  </Typography>
                )
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
                  Source frame unknown.
                </Typography>
              )}
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
