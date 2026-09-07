import { useEffect, useState } from "react";
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
import IconButton from "@mui/material/IconButton";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import CheckIcon from "@mui/icons-material/Check";
import CloseIcon from "@mui/icons-material/Close";
import HelpOutlineIcon from "@mui/icons-material/HelpOutlined";
import PersonIcon from "@mui/icons-material/Person";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import UndoIcon from "@mui/icons-material/Undo";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { api } from "../../lib/api";
import type { Job, Player } from "../../lib/types";
import { LockOverlay } from "../LockOverlay";
import { CardTilesSkeleton } from "../Skeletons";
import { RedoButton } from "./RedoButton";
import { formatTimestamp } from "./types";

// Matches the identified-player grid on the Players page, so a thumbnail
// tile reads the same size whether it's showing up there or here.
const PLAYER_TILE_WIDTH = 150;
const PLAYER_TILE_HEIGHT = 280;
const VISIBLE_THUMB_COUNT = 3;

// `showIdentificationBox` opts into player.identification_thumbnail_base64
// (a copy of the thumbnail with a white highlight box around the subject) -
// only set server-side when another player's box actually crowds into the
// crop. Pass it only for tiles that are still being decided (the
// Unidentified group, and its selections in ResolvePanel above); the
// resolved Identified/Ignored tiles below, and every other page that shows
// a player's thumbnail (Players/Teams/Stats), always get the plain photo.
function PlayerThumbnail({ player, showIdentificationBox = false }: { player: Player; showIdentificationBox?: boolean }) {
  const src = (showIdentificationBox && player.identification_thumbnail_base64) || player.thumbnail_base64;
  return src ? (
    <Box
      component="img"
      src={`data:image/jpeg;base64,${src}`}
      alt={`Player ${player.stable_id}`}
      sx={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
    />
  ) : (
    <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <PersonIcon sx={{ fontSize: 48, color: "text.disabled" }} />
    </Box>
  );
}

function UnidentifiedPlayerTile({
  player,
  selected,
  onSelect,
}: {
  player: Player;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <Tooltip
      title={player.thumbnail_timestamp_s != null ? formatTimestamp(player.thumbnail_timestamp_s) : "Timestamp unknown"}
    >
      <Card
        onClick={onSelect}
        variant="outlined"
        sx={{
          width: PLAYER_TILE_WIDTH,
          flexShrink: 0,
          overflow: "hidden",
          cursor: "pointer",
          borderColor: selected ? "primary.main" : "divider",
          borderWidth: selected ? 2 : 1,
          transition: "border-color 0.1s ease",
        }}
      >
        <Box sx={{ position: "relative", width: "100%", height: PLAYER_TILE_HEIGHT, bgcolor: "action.hover" }}>
          <PlayerThumbnail player={player} showIdentificationBox />
          {selected && (
            <Box
              sx={{
                position: "absolute",
                top: 6,
                right: 6,
                width: 22,
                height: 22,
                borderRadius: "50%",
                bgcolor: "primary.main",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <CheckIcon sx={{ fontSize: 14, color: "#fff" }} />
            </Box>
          )}
        </Box>
      </Card>
    </Tooltip>
  );
}

// A lighter tile for the Identified/Ignored groups - just a thumbnail, a
// label (the assigned name, or "Ignored"), and one small action to send
// the player back to Unidentified for correction. Not selectable/bulk-
// actionable like UnidentifiedPlayerTile - moving one player at a time
// back is enough for what's meant to be an occasional correction, not the
// primary workflow.
function ResolvedPlayerTile({
  player,
  label,
  undoTooltip,
  onUndo,
  disabled,
}: {
  player: Player;
  label: string;
  undoTooltip: string;
  onUndo: () => void;
  disabled: boolean;
}) {
  return (
    <Card variant="outlined" sx={{ width: PLAYER_TILE_WIDTH, flexShrink: 0, overflow: "hidden" }}>
      <Box sx={{ position: "relative", width: "100%", height: PLAYER_TILE_HEIGHT * 0.6, bgcolor: "action.hover" }}>
        <PlayerThumbnail player={player} />
      </Box>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", px: 1, py: 0.5 }}>
        <Typography variant="caption" noWrap title={label} sx={{ flex: 1, minWidth: 0 }}>
          {label}
        </Typography>
        <Tooltip title={undoTooltip}>
          <span>
            <IconButton size="small" onClick={onUndo} disabled={disabled}>
              <UndoIcon fontSize="inherit" />
            </IconButton>
          </span>
        </Tooltip>
      </Stack>
    </Card>
  );
}

function GroupSection({
  title,
  count,
  emptyHint,
  children,
}: {
  title: string;
  count: number;
  emptyHint: string;
  children: React.ReactNode;
}) {
  return (
    <Box>
      <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 1 }}>
        <Typography variant="subtitle2">{title}</Typography>
        <Chip size="small" label={count} />
      </Stack>
      {count === 0 ? (
        <Typography variant="body2" color="text.secondary">
          {emptyHint}
        </Typography>
      ) : (
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>{children}</Box>
      )}
    </Box>
  );
}

// Always rendered (not just once something's selected) so the controls
// don't jump around - it just sits disabled/empty until 1+ tiles below are
// selected.
function ResolvePanel({
  selectedPlayers,
  roster,
  onAssign,
  onIgnore,
  onClearSelection,
  saving,
  disabled,
  error,
}: {
  selectedPlayers: Player[];
  roster: string[];
  onAssign: (name: string) => void;
  onIgnore: () => void;
  onClearSelection: () => void;
  saving: boolean;
  disabled: boolean;
  error: string | null;
}) {
  const [name, setName] = useState("");
  const hasSelection = selectedPlayers.length > 0;
  const visiblePlayers = selectedPlayers.slice(0, VISIBLE_THUMB_COUNT);
  const overflowCount = selectedPlayers.length - visiblePlayers.length;

  useEffect(() => {
    if (!hasSelection) setName("");
  }, [hasSelection]);

  return (
    <Card variant="outlined" sx={{ p: 2, mb: 2 }}>
      <Stack direction="row" spacing={2} sx={{ alignItems: "center", flexWrap: "wrap" }}>
        {hasSelection ? (
          <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
            {visiblePlayers.map((p) => (
              <Tooltip key={p.stable_id} title={`Player #${p.stable_id}`}>
                <Box sx={{ width: 40, height: 40, borderRadius: 1, overflow: "hidden", bgcolor: "action.hover", flexShrink: 0 }}>
                  <PlayerThumbnail player={p} showIdentificationBox />
                </Box>
              </Tooltip>
            ))}
            {overflowCount > 0 && (
              <Tooltip title={selectedPlayers.slice(VISIBLE_THUMB_COUNT).map((p) => `#${p.stable_id}`).join(", ")}>
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: 1,
                    bgcolor: "action.hover",
                    flexShrink: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Typography variant="caption" color="text.secondary">
                    +{overflowCount}
                  </Typography>
                </Box>
              </Tooltip>
            )}
          </Stack>
        ) : null}

        <Typography variant="body2" color="text.secondary" sx={{ minWidth: 76 }}>
          {selectedPlayers.length} selected
        </Typography>

        <Autocomplete
          size="small"
          options={roster}
          value={name || null}
          onChange={(_, value) => setName(value ?? "")}
          disabled={disabled || saving || !hasSelection}
          renderInput={(params) => <TextField {...params} placeholder="Pick a name" />}
          sx={{ width: 220 }}
        />

        <Button variant="contained" disabled={disabled || saving || !hasSelection || !name} onClick={() => onAssign(name)}>
          Assign
        </Button>
        <Button
          variant="outlined"
          color="inherit"
          startIcon={<VisibilityOffIcon />}
          disabled={disabled || saving || !hasSelection}
          onClick={onIgnore}
        >
          Ignore
        </Button>

        {hasSelection && (
          <IconButton size="small" onClick={onClearSelection} disabled={disabled || saving} sx={{ ml: "auto" }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        )}
      </Stack>

      {saving && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
          Saving...
        </Typography>
      )}
      {error && (
        <Typography variant="caption" color="error" sx={{ display: "block", mt: 1 }}>
          {error}
        </Typography>
      )}
    </Card>
  );
}

interface UnidentifiedPlayersSectionProps {
  job: Job;
  onJobUpdated: (job: Job) => void;
  onRedoPlayers: () => Promise<void>;
}

// The single-video player-naming screen, reached from the Setup tab's
// Player Identification card. Detected players are always in exactly one
// of three groups - Unidentified, Identified, Ignored - and every
// assign/ignore/undo action persists to the backend immediately (see
// resolve() below), so nothing is lost if this page is left mid-review.
//
// "Confirm Player Identification" - disabled until nothing's left
// Unidentified - is what actually triggers the expensive reprocessing
// (consolidating stats, rebuilding the dashboard, re-rendering the
// annotated video) and persists a "confirmed" flag (players.py's
// player_config.json), the same state-management pattern ScoreSection
// uses for scoring: once confirmed, the whole section locks (see
// LockOverlay) and the button turns into "Redo Player Identification",
// which only unlocks editing again - distinct from "Reset all players"
// below, which actually clears every name/ignored flag and starts over.
export function UnidentifiedPlayersSection({ job, onJobUpdated, onRedoPlayers }: UnidentifiedPlayersSectionProps) {
  const [players, setPlayers] = useState<Player[] | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [roster, setRoster] = useState<string[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [resolving, setResolving] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [redoDialogOpen, setRedoDialogOpen] = useState(false);
  const [redoing, setRedoing] = useState(false);

  const refresh = () =>
    api.getPlayers(job.id).then((res) => {
      setPlayers(res.players);
      setConfirmed(res.confirmed);
    });

  useEffect(() => {
    let cancelled = false;
    setPlayers(null);
    refresh().catch(() => !cancelled && undefined);
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  useEffect(() => {
    api
      .getRoster()
      .then((res) => setRoster(res.players))
      .catch(() => undefined);
  }, []);

  if (players === null) {
    return (
      <Stack direction={{ xs: "column", md: "row" }} spacing={3}>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Skeleton variant="rounded" height={72} sx={{ mb: 2 }} />
          <Stack spacing={3}>
            {["Unidentified players", "Identified players", "Ignored players"].map((label) => (
              <Box key={label}>
                <Skeleton variant="text" width={160} sx={{ mb: 1 }} />
                <CardTilesSkeleton width={PLAYER_TILE_WIDTH} height={PLAYER_TILE_HEIGHT} count={4} />
              </Box>
            ))}
          </Stack>
        </Box>
        <Stack spacing={1.5} sx={{ width: 260, flexShrink: 0 }}>
          <Skeleton variant="rounded" height={36} />
          <Skeleton variant="rounded" height={36} />
        </Stack>
      </Stack>
    );
  }

  if (players.length === 0) {
    return <Typography color="text.secondary">No players were detected in this video.</Typography>;
  }

  const unidentified = players.filter((p) => !p.ignored && !p.name?.trim());
  const identified = players.filter((p) => !p.ignored && p.name?.trim());
  const ignoredPlayers = players.filter((p) => p.ignored);
  const ignoredIds = ignoredPlayers.map((p) => p.stable_id);
  const selectedPlayers = unidentified.filter((p) => selectedIds.has(p.stable_id));
  const confirmDisabled = unidentified.length > 0;

  async function resolve(action: () => Promise<void>) {
    setResolving(true);
    setResolveError(null);
    try {
      await action();
      await refresh();
      setSelectedIds(new Set());
    } catch (err) {
      setResolveError(err instanceof Error ? err.message : String(err));
    } finally {
      setResolving(false);
    }
  }

  function handleAssign(name: string) {
    if (selectedPlayers.length === 0) return;
    void resolve(async () => {
      const names = Object.fromEntries(selectedPlayers.map((p) => [String(p.stable_id), name]));
      await api.updateNames(job.id, names, ignoredIds);
    });
  }

  function handleIgnore() {
    if (selectedPlayers.length === 0) return;
    void resolve(async () => {
      const ignored = [...ignoredIds, ...selectedPlayers.map((p) => p.stable_id)];
      await api.updateNames(job.id, {}, ignored);
    });
  }

  function handleUnName(stableId: number) {
    void resolve(async () => {
      await api.updateNames(job.id, { [String(stableId)]: "" }, ignoredIds);
    });
  }

  function handleUnIgnore(stableId: number) {
    void resolve(async () => {
      await api.updateNames(job.id, {}, ignoredIds.filter((id) => id !== stableId));
    });
  }

  function toggleSelected(stableId: number) {
    setResolveError(null);
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(stableId)) next.delete(stableId);
      else next.add(stableId);
      return next;
    });
  }

  async function handleConfirm() {
    setConfirming(true);
    setResolveError(null);
    try {
      onJobUpdated(await api.finalizeJob(job.id));
      const res = await api.setPlayersConfirmed(job.id, true);
      setConfirmed(res.confirmed);
    } catch (err) {
      setResolveError(err instanceof Error ? err.message : String(err));
    } finally {
      setConfirming(false);
    }
  }

  async function handleConfirmRedo() {
    setRedoing(true);
    try {
      const res = await api.setPlayersConfirmed(job.id, false);
      setConfirmed(res.confirmed);
      setRedoDialogOpen(false);
    } finally {
      setRedoing(false);
    }
  }

  const locked = confirmed || resolving || confirming;

  return (
    <Stack direction={{ xs: "column", md: "row" }} spacing={3} sx={{ position: "relative" }}>
      <LockOverlay
        active={confirmed}
        label="Redo Player Identification to make changes"
        onClick={() => setRedoDialogOpen(true)}
      />

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <ResolvePanel
          selectedPlayers={selectedPlayers}
          roster={roster}
          onAssign={handleAssign}
          onIgnore={handleIgnore}
          onClearSelection={() => {
            setResolveError(null);
            setSelectedIds(new Set());
          }}
          saving={resolving}
          disabled={locked}
          error={resolveError}
        />

        <Stack spacing={3}>
          <GroupSection title="Unidentified players" count={unidentified.length} emptyHint="Nothing waiting to be named.">
            {unidentified.map((player) => (
              <UnidentifiedPlayerTile
                key={player.stable_id}
                player={player}
                selected={selectedIds.has(player.stable_id)}
                onSelect={() => !locked && toggleSelected(player.stable_id)}
              />
            ))}
          </GroupSection>

          <GroupSection title="Identified players" count={identified.length} emptyHint="Nobody's been named yet.">
            {identified.map((player) => (
              <ResolvedPlayerTile
                key={player.stable_id}
                player={player}
                label={player.name ?? ""}
                undoTooltip="Send back to Unidentified"
                onUndo={() => handleUnName(player.stable_id)}
                disabled={locked}
              />
            ))}
          </GroupSection>

          <GroupSection title="Ignored players" count={ignoredPlayers.length} emptyHint="Nobody's been ignored.">
            {ignoredPlayers.map((player) => (
              <ResolvedPlayerTile
                key={player.stable_id}
                player={player}
                label="Ignored"
                undoTooltip="Send back to Unidentified"
                onUndo={() => handleUnIgnore(player.stable_id)}
                disabled={locked}
              />
            ))}
          </GroupSection>
        </Stack>
      </Box>

      {/* Confirm/Redo + the destructive full reset live in their own
          right-hand column rather than inline with the groups - the same
          "act on the whole review" action set ScoreSection keeps together
          in its own card, just on this page's right side instead. */}
      <Stack spacing={1.5} sx={{ width: 260, flexShrink: 0 }}>
        {confirmed ? (
          <Button
            variant="contained"
            color="warning"
            startIcon={<RestartAltIcon />}
            fullWidth
            onClick={() => setRedoDialogOpen(true)}
          >
            Redo Player Identification
          </Button>
        ) : (
          <>
            <Tooltip title={confirmDisabled ? "Every player needs a name or Ignore before confirming" : ""}>
              <span style={{ display: "block" }}>
                <Button
                  variant="contained"
                  fullWidth
                  disabled={confirmDisabled || confirming}
                  onClick={() => void handleConfirm()}
                >
                  {confirming ? "Confirming..." : "Confirm Player Identification"}
                </Button>
              </span>
            </Tooltip>
            {confirmDisabled && (
              <Typography variant="caption" color="warning.main" sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <HelpOutlineIcon fontSize="inherit" /> {unidentified.length} still unidentified - resolve everyone
                before confirming.
              </Typography>
            )}
          </>
        )}

        <RedoButton
          label="Reset all players"
          confirmTitle="Reset all players?"
          confirmText={`This clears every player name and ignored flag set for "${job.original_filename}" - everyone goes back to unidentified - and reprocesses its stats, dashboard, and video. This can't be undone.`}
          onConfirm={onRedoPlayers}
          disabled={locked}
          fullWidth
        />
      </Stack>

      <Dialog open={redoDialogOpen} onClose={() => (redoing ? undefined : setRedoDialogOpen(false))}>
        <DialogTitle>Redo player identification?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This won't change any names or ignored flags - it just unlocks player identification for
            editing again until you confirm it once more.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRedoDialogOpen(false)} disabled={redoing}>
            Cancel
          </Button>
          <Button color="warning" variant="contained" onClick={() => void handleConfirmRedo()} disabled={redoing}>
            {redoing ? "Working..." : "Confirm"}
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
