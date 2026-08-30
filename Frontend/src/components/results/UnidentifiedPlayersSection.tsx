import { useEffect, useState } from "react";
import Autocomplete from "@mui/material/Autocomplete";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import CheckIcon from "@mui/icons-material/Check";
import CloseIcon from "@mui/icons-material/Close";
import HelpOutlineIcon from "@mui/icons-material/HelpOutlined";
import PersonIcon from "@mui/icons-material/Person";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { api } from "../../lib/api";
import type { Job, Player } from "../../lib/types";
import { LoadingSpinner } from "../LoadingSpinner";
import { RedoButton } from "./RedoButton";
import { formatTimestamp } from "./types";

// Matches the identified-player grid on the Players page, so a thumbnail
// tile reads the same size whether it's showing up there or here.
const PLAYER_TILE_WIDTH = 150;
const PLAYER_TILE_HEIGHT = 280;
const VISIBLE_THUMB_COUNT = 3;

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
          {player.thumbnail_base64 ? (
            <Box
              component="img"
              src={`data:image/jpeg;base64,${player.thumbnail_base64}`}
              alt={`Player ${player.stable_id}`}
              sx={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            />
          ) : (
            <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <PersonIcon sx={{ fontSize: 48, color: "text.disabled" }} />
            </Box>
          )}
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
  error,
}: {
  selectedPlayers: Player[];
  roster: string[];
  onAssign: (name: string) => void;
  onIgnore: () => void;
  onClearSelection: () => void;
  saving: boolean;
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
                  {p.thumbnail_base64 ? (
                    <Box
                      component="img"
                      src={`data:image/jpeg;base64,${p.thumbnail_base64}`}
                      alt={`Player ${p.stable_id}`}
                      sx={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
                    />
                  ) : (
                    <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <PersonIcon fontSize="small" sx={{ color: "text.disabled" }} />
                    </Box>
                  )}
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
          disabled={saving || !hasSelection}
          renderInput={(params) => <TextField {...params} placeholder="Pick a name" />}
          sx={{ width: 220 }}
        />

        <Button variant="contained" disabled={saving || !hasSelection || !name} onClick={() => onAssign(name)}>
          Assign
        </Button>
        <Button
          variant="outlined"
          color="inherit"
          startIcon={<VisibilityOffIcon />}
          disabled={saving || !hasSelection}
          onClick={onIgnore}
        >
          Ignore
        </Button>

        {hasSelection && (
          <IconButton size="small" onClick={onClearSelection} disabled={saving} sx={{ ml: "auto" }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        )}
      </Stack>

      {saving && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
          Saving - this reprocesses stats for this video, so it'll take a moment...
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

// The single-video equivalent of the Players page's old "Unidentified"
// tab: every one of this video's still-unnamed detections, resolvable
// right here via a dropdown-assign-or-ignore panel. Once nothing's left to
// resolve, this becomes the "Redo identification" entry point instead.
export function UnidentifiedPlayersSection({ job, onJobUpdated, onRedoPlayers }: UnidentifiedPlayersSectionProps) {
  const [players, setPlayers] = useState<Player[] | null>(null);
  const [roster, setRoster] = useState<string[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [resolving, setResolving] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);

  const refresh = () => api.getPlayers(job.id).then((res) => setPlayers(res.players));

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

  if (players === null) return <LoadingSpinner minHeight={160} />;

  const unidentified = players.filter((p) => !p.ignored && !p.name?.trim());
  const hasPlayers = players.length > 0;
  const needsSetup = unidentified.length > 0;

  if (!hasPlayers) {
    return <Typography color="text.secondary">No players were detected in this video.</Typography>;
  }

  if (!needsSetup) {
    return (
      <RedoButton
        label="Redo identification"
        confirmTitle="Redo player identification?"
        confirmText={`This clears every player name and ignored flag set for "${job.original_filename}" - everyone goes back to unidentified - and reprocesses its stats, dashboard, and video. This can't be undone.`}
        onConfirm={onRedoPlayers}
      />
    );
  }

  const ignoredIds = players.filter((p) => p.ignored).map((p) => p.stable_id);
  const selectedPlayers = unidentified.filter((p) => selectedIds.has(p.stable_id));

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
      if (selectedPlayers.length === unidentified.length) onJobUpdated(await api.finalizeJob(job.id));
    });
  }

  function handleIgnore() {
    if (selectedPlayers.length === 0) return;
    void resolve(async () => {
      const ignored = [...ignoredIds, ...selectedPlayers.map((p) => p.stable_id)];
      await api.updateNames(job.id, {}, ignored);
      if (selectedPlayers.length === unidentified.length) onJobUpdated(await api.finalizeJob(job.id));
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

  if (unidentified.length === 0) {
    return (
      <Typography color="text.secondary" sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <HelpOutlineIcon fontSize="small" /> Nothing waiting to be named right now.
      </Typography>
    );
  }

  return (
    <>
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
        error={resolveError}
      />

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
        {unidentified.map((player) => (
          <UnidentifiedPlayerTile
            key={player.stable_id}
            player={player}
            selected={selectedIds.has(player.stable_id)}
            onSelect={() => toggleSelected(player.stable_id)}
          />
        ))}
      </Box>
    </>
  );
}
