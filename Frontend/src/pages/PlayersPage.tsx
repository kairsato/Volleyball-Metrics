import { useEffect, useMemo, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import MenuItem from "@mui/material/MenuItem";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { Link as RouterLink } from "react-router-dom";
import { api } from "../lib/api";
import type { PlayerProfileSummary, TeamEntry } from "../lib/types";
import { PageHeader } from "../components/PageHeader";
import { PlayerPhotoCarousel } from "../components/PlayerPhotoCarousel";

interface IdentifiedPlayer {
  name: string;
  thumbnails: string[];
  totalHits: number;
  ralliesParticipated: number;
  gameCount: number;
}

function toIdentifiedPlayer(p: PlayerProfileSummary): IdentifiedPlayer {
  return {
    name: p.name,
    thumbnails: p.thumbnail_base64 ? [p.thumbnail_base64] : [],
    totalHits: p.total_hits,
    ralliesParticipated: p.rallies_participated,
    gameCount: p.game_count,
  };
}

// Shared by every player tile on this page so the whole grid reads as one
// consistent set of same-sized cards.
const PLAYER_TILE_WIDTH = 150;
const PLAYER_TILE_HEIGHT = 280;

function IdentifiedPlayerCard({ player }: { player: IdentifiedPlayer }) {
  return (
    <Card
      variant="outlined"
      sx={{
        position: "relative",
        width: "100%",
        aspectRatio: `${PLAYER_TILE_WIDTH} / ${PLAYER_TILE_HEIGHT}`,
        overflow: "hidden",
      }}
    >
      <CardActionArea component={RouterLink} to={`/player?name=${encodeURIComponent(player.name)}`} sx={{ height: "100%" }}>
        <Box sx={{ width: "100%", height: "100%", bgcolor: "action.hover" }}>
          <PlayerPhotoCarousel thumbnails={player.thumbnails} alt={player.name} />
        </Box>

        <Box
          sx={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            p: 1,
            background: "linear-gradient(to top, rgba(0,0,0,0.75), rgba(0,0,0,0) 100%)",
          }}
        >
          <Typography noWrap sx={{ fontWeight: 600, color: "#fff" }}>
            {player.name}
          </Typography>
        </Box>
      </CardActionArea>
    </Card>
  );
}

function AddPlayerDialog({
  open,
  onClose,
  onAdd,
}: {
  open: boolean;
  onClose: () => void;
  onAdd: (name: string) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAdd() {
    const trimmed = name.trim();
    if (!trimmed) return;
    setSaving(true);
    setError(null);
    try {
      await onAdd(trimmed);
      setName("");
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog open={open} onClose={saving ? undefined : onClose} maxWidth="xs" fullWidth>
      <DialogTitle>Add a player</DialogTitle>
      <DialogContent>
        <TextField
          autoFocus
          fullWidth
          placeholder="Player's name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void handleAdd();
          }}
          sx={{ mt: 1 }}
        />
        {error && (
          <Typography variant="caption" color="error" sx={{ display: "block", mt: 1 }}>
            {error}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={saving}>
          Cancel
        </Button>
        <Button variant="contained" disabled={saving || !name.trim()} onClick={() => void handleAdd()}>
          {saving ? "Adding..." : "Add"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// Every named player at once, across every game - linked to from each
// game's own Setup page so there's always a one-click way to see everyone
// rather than just the people in whatever game you happen to be looking
// at. Naming/merging/ignoring still-unidentified detections happens on
// that per-game Setup page now, not here.
//
// Backed entirely by Backend/API/player_profiles.py's persisted store
// (api.getPlayerProfiles()) - one fast request instead of the
// getResults+getPlayers-per-completed-job aggregation this page used to do
// client-side on every visit.
export function PlayersPage() {
  const [players, setPlayers] = useState<IdentifiedPlayer[] | null>(null);
  const [roster, setRoster] = useState<string[]>([]);
  const [teams, setTeams] = useState<TeamEntry[]>([]);
  const [addPlayerOpen, setAddPlayerOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [teamFilter, setTeamFilter] = useState<string>("all");

  useEffect(() => {
    let cancelled = false;
    api
      .getPlayerProfiles()
      .then((res) => !cancelled && setPlayers(res.players.map(toIdentifiedPlayer)))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    api
      .getRoster()
      .then((res) => setRoster(res.players))
      .catch(() => undefined);
    api
      .getTeams()
      .then((res) => setTeams(res.teams))
      .catch(() => undefined);
  }, []);

  // Every roster name gets a card, whether or not it has stats yet - a
  // name added to the roster (or typed while naming someone on a game,
  // which adds it here too) but not yet attached to any finalized game
  // shows up as a zero-stat placeholder rather than not appearing at all.
  const allPlayers: IdentifiedPlayer[] = players
    ? [
        ...players,
        ...roster
          .filter((name) => !players.some((p) => p.name === name))
          .map((name) => ({ name, thumbnails: [], totalHits: 0, ralliesParticipated: 0, gameCount: 0 })),
      ]
    : [];

  const selectedTeam = teams.find((t) => t.id === teamFilter);
  const displayedPlayers = useMemo(() => {
    const query = search.trim().toLowerCase();
    return allPlayers.filter(
      (player) =>
        (query === "" || player.name.toLowerCase().includes(query)) &&
        (teamFilter === "all" || (selectedTeam?.players.includes(player.name) ?? false)),
    );
    // allPlayers is rebuilt every render from `players`/`roster` - depend on
    // those directly instead so this doesn't recompute on every keystroke's
    // unrelated re-render for no reason.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players, roster, search, teamFilter, selectedTeam]);

  async function handleAddToRoster(name: string) {
    const res = await api.addToRoster(name);
    setRoster(res.players);
  }

  return (
    <Box>
      <PageHeader title="Players" addLabel="Add player" onAdd={() => setAddPlayerOpen(true)}>
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
          <TextField
            size="small"
            placeholder="Search players..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            sx={{ width: { xs: "100%", sm: 240 } }}
          />
          <TextField
            select
            size="small"
            label="Team"
            value={teamFilter}
            onChange={(event) => setTeamFilter(event.target.value)}
            sx={{ width: { xs: "100%", sm: 180 } }}
          >
            <MenuItem value="all">All players</MenuItem>
            {teams.map((team) => (
              <MenuItem key={team.id} value={team.id}>
                {team.name}
              </MenuItem>
            ))}
          </TextField>
        </Stack>
      </PageHeader>

      {players === null ? (
        // Same grid the real cards render into (see below) - so the loading
        // state already has the right shape/column count instead of
        // rearranging itself the moment real data shows up.
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: `repeat(auto-fill, minmax(${PLAYER_TILE_WIDTH}px, 1fr))`,
            gap: 2,
          }}
        >
          {Array.from({ length: 10 }).map((_, i) => (
            <Skeleton
              key={i}
              variant="rounded"
              sx={{ width: "100%", aspectRatio: `${PLAYER_TILE_WIDTH} / ${PLAYER_TILE_HEIGHT}` }}
            />
          ))}
        </Box>
      ) : displayedPlayers.length === 0 ? (
        <Typography color="text.secondary">No players match your search/filter.</Typography>
      ) : (
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: `repeat(auto-fill, minmax(${PLAYER_TILE_WIDTH}px, 1fr))`,
            gap: 2,
          }}
        >
          {displayedPlayers.map((player) => (
            <IdentifiedPlayerCard key={player.name} player={player} />
          ))}
        </Box>
      )}

      <AddPlayerDialog open={addPlayerOpen} onClose={() => setAddPlayerOpen(false)} onAdd={handleAddToRoster} />
    </Box>
  );
}
