import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
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
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import CloseIcon from "@mui/icons-material/Close";
import HelpOutlineIcon from "@mui/icons-material/HelpOutlined";
import PersonIcon from "@mui/icons-material/Person";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import UndoIcon from "@mui/icons-material/Undo";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { api } from "../../lib/api";
import type { CandidateGroup, Job, NamesUpdateOut, Player } from "../../lib/types";
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

// Shared by every tile so hovering any player thumbnail - unidentified,
// identified, or ignored - always shows the same "which detection is this"
// reference: its stable id (what candidate-match hints, undo actions, and
// the raw player_positions.json log all key on) plus when its thumbnail was
// captured.
function playerHoverLabel(player: Player): string {
  const timestamp = player.thumbnail_timestamp_s != null ? formatTimestamp(player.thumbnail_timestamp_s) : "unknown";
  return `Player #${player.stable_id} · ${timestamp}`;
}

function UnidentifiedPlayerTile({
  player,
  selected,
  onSelect,
  onPreview,
}: {
  player: Player;
  selected: boolean;
  onSelect: () => void;
  onPreview: () => void;
}) {
  return (
    <Tooltip title={playerHoverLabel(player)}>
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
          // The magnifier is revealed on hover rather than always drawn, so
          // a wall of tiles stays readable as photos; it is always present
          // for keyboard/touch users, just transparent until focused.
          "&:hover .preview-button, & .preview-button:focus-visible": { opacity: 1 },
        }}
      >
        <Box sx={{ position: "relative", width: "100%", height: PLAYER_TILE_HEIGHT, bgcolor: "action.hover" }}>
          <PlayerThumbnail player={player} showIdentificationBox />

          <Tooltip title="See the whole frame this photo came from">
            <IconButton
              className="preview-button"
              size="small"
              aria-label={`See the whole frame player #${player.stable_id} was photographed in`}
              // The tile itself toggles selection - opening the preview must
              // not also select the player the reviewer was only inspecting.
              onClick={(event) => {
                event.stopPropagation();
                onPreview();
              }}
              sx={{
                position: "absolute",
                top: 6,
                left: 6,
                opacity: 0,
                transition: "opacity 0.1s ease",
                bgcolor: "rgba(0, 0, 0, 0.55)",
                color: "#fff",
                "&:hover": { bgcolor: "rgba(0, 0, 0, 0.75)" },
              }}
            >
              <ZoomInIcon fontSize="small" />
            </IconButton>
          </Tooltip>
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
  onPreview,
  disabled,
}: {
  player: Player;
  label: string;
  undoTooltip: string;
  onUndo: () => void;
  onPreview: () => void;
  disabled: boolean;
}) {
  return (
    <Card
      variant="outlined"
      sx={{
        width: PLAYER_TILE_WIDTH,
        flexShrink: 0,
        overflow: "hidden",
        "&:hover .preview-button, & .preview-button:focus-visible": { opacity: 1 },
      }}
    >
      <Tooltip title={playerHoverLabel(player)}>
        <Box sx={{ position: "relative", width: "100%", height: PLAYER_TILE_HEIGHT * 0.6, bgcolor: "action.hover" }}>
          <PlayerThumbnail player={player} />

          {/* Ignored players deliberately arrive with no photo (see
              players.list_players), so this is the only way to see who one
              is before putting them back - and it costs a frame decode only
              for the one a reviewer actually opens. */}
          <Tooltip title="See the whole frame this player was detected in">
            <IconButton
              className="preview-button"
              size="small"
              aria-label={`See the whole frame player #${player.stable_id} was detected in`}
              onClick={(event) => {
                event.stopPropagation();
                onPreview();
              }}
              sx={{
                position: "absolute",
                top: 4,
                left: 4,
                opacity: player.thumbnail_base64 ? 0 : 1,
                transition: "opacity 0.1s ease",
                bgcolor: "rgba(0, 0, 0, 0.55)",
                color: "#fff",
                "&:hover": { bgcolor: "rgba(0, 0, 0, 0.75)" },
              }}
            >
              <ZoomInIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Tooltip>
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

// One "these are probably all the same person" suggestion - see
// API/services/players.build_candidate_groups for how the grouping is
// built and why it is deliberately generous.
//
// Rendered above the flat Unidentified grid because acting on a group is
// the thing that actually removes work: one name for a dozen tiles instead
// of a dozen separate decisions on photos of the same player. Membership
// stays fully editable - every thumbnail is the same toggle it is in the
// grid below, so dropping the one tile that doesn't belong costs a click -
// since a group is a suggestion the backend has committed nothing to. The
// naming itself still goes through the ResolvePanel above, so there is
// exactly one code path that writes names, and its simultaneous-name guard
// applies to a group assign just as it does to a hand-picked selection.
function SuggestedGroupCard({
  group,
  members,
  selectedIds,
  onToggle,
  onPreview,
  onSelectAll,
  disabled,
}: {
  group: CandidateGroup;
  members: Player[];
  selectedIds: Set<number>;
  onToggle: (stableId: number) => void;
  onPreview: (player: Player) => void;
  onSelectAll: () => void;
  disabled: boolean;
}) {
  const allSelected = members.every((p) => selectedIds.has(p.stable_id));

  return (
    <Card variant="outlined" sx={{ p: 1.5, mb: 1.5 }}>
      <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 1, flexWrap: "wrap" }}>
        <Typography variant="subtitle2">
          {group.suggested_name ? `Probably ${group.suggested_name}` : "Probably the same person"}
        </Typography>
        <Chip size="small" label={`${members.length} photos`} />
        <Tooltip title="How alike the least-similar pair in this group looks - the whole group is only as good as its weakest link">
          <Chip size="small" variant="outlined" label={`${Math.round(group.confidence * 100)}%`} />
        </Tooltip>
        <Button size="small" onClick={onSelectAll} disabled={disabled || allSelected} sx={{ ml: "auto" }}>
          {allSelected ? "All selected" : "Select all"}
        </Button>
      </Stack>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
        {members.map((player) => (
          <UnidentifiedPlayerTile
            key={player.stable_id}
            player={player}
            selected={selectedIds.has(player.stable_id)}
            onSelect={() => !disabled && onToggle(player.stable_id)}
            onPreview={() => onPreview(player)}
          />
        ))}
      </Box>
    </Card>
  );
}

// The whole frame a tile's photo was cut from, with the same white box on
// the subject - opened from a tile's magnifier. A 220px crop shows what
// someone looks like but throws away everything a reviewer uses to place
// them: who they were next to, where on the court, what was happening. The
// image is fetched by the browser only when this opens (see
// api.playerFrameUrl), never as part of the players payload.
function PlayerFramePreview({
  jobId,
  player,
  onClose,
}: {
  jobId: string;
  player: Player | null;
  onClose: () => void;
}) {
  return (
    <Dialog open={player !== null} onClose={onClose} maxWidth="lg" fullWidth>
      {player && (
        <>
          <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <span>Player #{player.stable_id}</span>
            <Typography variant="body2" color="text.secondary">
              {player.thumbnail_timestamp_s != null ? formatTimestamp(player.thumbnail_timestamp_s) : "unknown time"}
            </Typography>
            <IconButton size="small" onClick={onClose} sx={{ ml: "auto" }} aria-label="Close">
              <CloseIcon fontSize="small" />
            </IconButton>
          </DialogTitle>
          <DialogContent>
            <Box
              component="img"
              src={api.playerFrameUrl(jobId, player.stable_id)}
              alt={`Whole frame containing player #${player.stable_id}`}
              sx={{ width: "100%", height: "auto", display: "block", borderRadius: 1 }}
            />
          </DialogContent>
        </>
      )}
    </Dialog>
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
  suggestedName,
  onAssign,
  onIgnore,
  onClearSelection,
  saving,
  disabled,
  error,
}: {
  selectedPlayers: Player[];
  roster: string[];
  // Pre-filled when the current selection is exactly one suggested group
  // that already has a named member - see SuggestedGroupCard.
  suggestedName: string | null;
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

  // Seeding from the suggestion rather than making the reviewer retype a
  // name the group already knows. Only re-runs when the selection empties
  // or the suggestion itself changes (i.e. a different group was picked),
  // so overriding it by hand and then adding/dropping a tile keeps the
  // typed name.
  useEffect(() => {
    if (!hasSelection) setName("");
    else if (suggestedName) setName(suggestedName);
  }, [hasSelection, suggestedName]);

  return (
    <Card variant="outlined" sx={{ p: 2, mb: 2 }}>
      <Stack direction="row" spacing={2} sx={{ alignItems: "center", flexWrap: "wrap" }}>
        {hasSelection ? (
          <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
            {visiblePlayers.map((p) => (
              <Tooltip key={p.stable_id} title={playerHoverLabel(p)}>
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
  const [candidateGroups, setCandidateGroups] = useState<CandidateGroup[]>([]);
  const [previewPlayer, setPreviewPlayer] = useState<Player | null>(null);
  const [provisional, setProvisional] = useState(false);
  const navigate = useNavigate();
  const [resolving, setResolving] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [redoDialogOpen, setRedoDialogOpen] = useState(false);
  const [redoing, setRedoing] = useState(false);

  const refresh = () =>
    api.getPlayers(job.id).then((res) => {
      setPlayers(res.players);
      setCandidateGroups(res.candidate_groups);
      setProvisional(res.provisional);
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

  // Tracking deliberately keeps every detection and defers working out who
  // is who until a court exists, since every signal that decides identity is
  // court-derived (see API/services/players.identification_is_provisional).
  // Showing the raw tracklets here would be thousands of tiles of the crowd.
  if (provisional) {
    return (
      <Stack spacing={2} sx={{ maxWidth: 560 }}>
        <Alert severity="info">
          <AlertTitle>Calibrate the court first</AlertTitle>
          Everyone on camera has been tracked, but who is who can&apos;t be worked out until the
          court is marked - it&apos;s what separates the players from everyone else in the room, and
          what tells one player&apos;s fragments from another&apos;s. Mark the court, then recalibrate
          this video and the players will be here.
        </Alert>
        <Button variant="contained" onClick={() => navigate(`/game/setup/court-calibration?job=${job.id}`)}>
          Go to court calibration
        </Button>
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

  // Groups are resolved against the CURRENT unidentified tiles rather than
  // the stable_ids the backend grouped, so naming or ignoring part of a
  // group shrinks it here immediately (applyNamesUpdate never refetches)
  // and a group whose members have all been dealt with disappears on its
  // own. A group with only one tile left is no longer a grouping.
  const byStableId = new Map(unidentified.map((p) => [p.stable_id, p]));
  const suggestedGroups = candidateGroups
    .map((group) => ({
      group,
      members: group.stable_ids.map((id) => byStableId.get(id)).filter((p): p is Player => p !== undefined),
    }))
    .filter(({ members }) => members.length > 1);
  const groupedIds = new Set(suggestedGroups.flatMap(({ members }) => members.map((p) => p.stable_id)));
  // Only offer a group's name once its whole group is selected - a partial
  // selection is the reviewer disagreeing with the grouping, which is
  // exactly when guessing a name for them would be wrong.
  const selectedSuggestedName =
    suggestedGroups.find(
      ({ group, members }) =>
        group.suggested_name && members.every((p) => selectedIds.has(p.stable_id)),
    )?.group.suggested_name ?? null;
  const ungrouped = unidentified.filter((p) => !groupedIds.has(p.stable_id));

  // Applies a /players/names response locally instead of re-fetching the
  // whole list (see resolve() below) - save_names/save_ignored on the
  // backend both return the FULL current names/ignored state (not just
  // what this request changed), so this is enough to bring every player's
  // name/ignored flag fully up to date without touching thumbnail data at
  // all (which never changes here, and is what makes a full refetch slow -
  // GET /players re-reads this job's whole player_positions.json, which for
  // a long match is hundreds of MB, just to re-derive images that already
  // haven't changed).
  function applyNamesUpdate(result: NamesUpdateOut) {
    setPlayers((prev) =>
      prev === null
        ? prev
        : prev.map((p) => ({
            ...p,
            name: result.names[String(p.stable_id)] ?? null,
            ignored: result.ignored.includes(p.stable_id),
          })),
    );
  }

  async function resolve(action: () => Promise<NamesUpdateOut>) {
    setResolving(true);
    setResolveError(null);
    try {
      applyNamesUpdate(await action());
      setSelectedIds(new Set());
    } catch (err) {
      setResolveError(err instanceof Error ? err.message : String(err));
    } finally {
      setResolving(false);
    }
  }

  function handleAssign(name: string) {
    if (selectedPlayers.length === 0) return;
    void resolve(() => {
      const names = Object.fromEntries(selectedPlayers.map((p) => [String(p.stable_id), name]));
      return api.updateNames(job.id, names, ignoredIds);
    });
  }

  function handleIgnore() {
    if (selectedPlayers.length === 0) return;
    void resolve(() => {
      const ignored = [...ignoredIds, ...selectedPlayers.map((p) => p.stable_id)];
      return api.updateNames(job.id, {}, ignored);
    });
  }

  function handleUnName(stableId: number) {
    void resolve(() => api.updateNames(job.id, { [String(stableId)]: "" }, ignoredIds));
  }

  function handleUnIgnore(stableId: number) {
    void resolve(() => api.updateNames(job.id, {}, ignoredIds.filter((id) => id !== stableId)));
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

      <PlayerFramePreview jobId={job.id} player={previewPlayer} onClose={() => setPreviewPlayer(null)} />

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <ResolvePanel
          selectedPlayers={selectedPlayers}
          roster={roster}
          suggestedName={selectedSuggestedName}
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
          {suggestedGroups.length > 0 && (
            <Box>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 1 }}>
                <Typography variant="subtitle2">Suggested groups</Typography>
                <Chip size="small" label={suggestedGroups.length} />
                <Typography variant="caption" color="text.secondary">
                  Photos that look like the same player - check them, then name the whole group at once
                </Typography>
              </Stack>
              {suggestedGroups.map(({ group, members }) => (
                <SuggestedGroupCard
                  key={group.stable_ids.join("-")}
                  group={group}
                  members={members}
                  selectedIds={selectedIds}
                  onToggle={toggleSelected}
                  onPreview={setPreviewPlayer}
                  onSelectAll={() =>
                    setSelectedIds((prev) => {
                      const next = new Set(prev);
                      members.forEach((p) => next.add(p.stable_id));
                      return next;
                    })
                  }
                  disabled={locked}
                />
              ))}
            </Box>
          )}

          <GroupSection
            title={suggestedGroups.length > 0 ? "Other unidentified players" : "Unidentified players"}
            count={ungrouped.length}
            emptyHint="Nothing waiting to be named."
          >
            {ungrouped.map((player) => (
              <UnidentifiedPlayerTile
                key={player.stable_id}
                player={player}
                selected={selectedIds.has(player.stable_id)}
                onSelect={() => !locked && toggleSelected(player.stable_id)}
                onPreview={() => setPreviewPlayer(player)}
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
                onPreview={() => setPreviewPlayer(player)}
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
                onPreview={() => setPreviewPlayer(player)}
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
          confirmText={`This clears every player name and ignored flag set for "${job.original_filename}" and works out who the players are again from scratch, using the saved court to decide who was actually on it. The court calibration itself is kept. Stats, dashboard and video are reprocessed afterwards. This can't be undone.`}
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
