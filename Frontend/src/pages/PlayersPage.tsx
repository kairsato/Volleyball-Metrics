import { useCallback, useEffect, useMemo, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import MenuItem from "@mui/material/MenuItem";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import { Link as RouterLink } from "react-router-dom";
import { api } from "../lib/api";
import type { Job, TeamEntry } from "../lib/types";
import { LoadingSpinner } from "../components/LoadingSpinner";
import { PageHeader } from "../components/PageHeader";
import { PlayerPhotoCarousel } from "../components/PlayerPhotoCarousel";

interface IdentifiedPlayer {
  name: string;
  // Every thumbnail found for this name across every video, not just the
  // first one - lets the card slowly cross-fade between them instead of
  // being stuck on whichever video happened to be processed first.
  thumbnails: string[];
  totalHits: number;
  ralliesParticipated: number;
  videoCount: number;
}

async function loadIdentifiedPlayers(completeJobs: Job[]): Promise<IdentifiedPlayer[]> {
  const perJob = await Promise.all(
    completeJobs.map(async (job) => {
      const [results, players] = await Promise.all([
        api.getResults(job.id).catch(() => null),
        api.getPlayers(job.id).catch(() => null),
      ]);
      return { results, players };
    }),
  );

  const byName = new Map<string, IdentifiedPlayer>();

  for (const { results, players } of perJob) {
    const thumbByStableId = new Map<number, string | null>();
    for (const p of players?.players ?? []) thumbByStableId.set(p.stable_id, p.thumbnail_base64);

    if (!results) continue;
    for (const [stableIdStr, stat] of Object.entries(results.players)) {
      const name = stat.name?.trim();
      if (!name) continue;

      const existing = byName.get(name) ?? {
        name,
        thumbnails: [],
        totalHits: 0,
        ralliesParticipated: 0,
        videoCount: 0,
      };
      existing.totalHits += stat.total_hits;
      existing.ralliesParticipated += stat.rallies_participated;
      existing.videoCount += 1;
      const thumb = thumbByStableId.get(Number(stableIdStr));
      if (thumb) existing.thumbnails.push(thumb);
      byName.set(name, existing);
    }
  }

  return Array.from(byName.values()).sort((a, b) => b.totalHits - a.totalHits);
}

// Shared by every player tile on this page so the whole grid reads as one
// consistent set of same-sized cards.
const PLAYER_TILE_WIDTH = 150;
const PLAYER_TILE_HEIGHT = 280;

function IdentifiedPlayerCard({ player }: { player: IdentifiedPlayer }) {
  return (
    <Card variant="outlined" sx={{ position: "relative", width: PLAYER_TILE_WIDTH, height: PLAYER_TILE_HEIGHT, overflow: "hidden" }}>
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

interface PlayersPageProps {
  jobs: Job[];
}

// Every named player at once, across every video - linked to from each
// video's own Setup page so there's always a one-click way to see everyone
// rather than just the people in whatever video you happen to be looking
// at. Naming/merging/ignoring still-unidentified detections happens on
// that per-video Setup page now, not here.
export function PlayersPage({ jobs }: PlayersPageProps) {
  const [players, setPlayers] = useState<IdentifiedPlayer[] | null>(null);
  const [roster, setRoster] = useState<string[]>([]);
  const [teams, setTeams] = useState<TeamEntry[]>([]);
  const [addPlayerOpen, setAddPlayerOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [teamFilter, setTeamFilter] = useState<string>("all");

  const completeJobs = jobs.filter((j) => j.status === "complete");
  const completeJobIds = completeJobs.map((j) => j.id).join(",");

  const refresh = useCallback(() => {
    return loadIdentifiedPlayers(completeJobs).then(setPlayers);
    // completeJobIds is a stable proxy for completeJobs's identity - re-fetching
    // on every jobs poll (which creates new array/object references every 3s)
    // would otherwise refetch results/players for every completed video constantly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completeJobIds]);

  useEffect(() => {
    let cancelled = false;
    setPlayers(null);
    refresh().catch(() => !cancelled && undefined);
    return () => {
      cancelled = true;
    };
  }, [refresh]);

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
  // name added to the roster (or typed while naming someone on a video,
  // which adds it here too) but not yet attached to any finalized video
  // shows up as a zero-stat placeholder rather than not appearing at all.
  const allPlayers: IdentifiedPlayer[] = players
    ? [
        ...players,
        ...roster
          .filter((name) => !players.some((p) => p.name === name))
          .map((name) => ({ name, thumbnails: [], totalHits: 0, ralliesParticipated: 0, videoCount: 0 })),
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
            sx={{ minWidth: 240 }}
          />
          <TextField
            select
            size="small"
            label="Team"
            value={teamFilter}
            onChange={(event) => setTeamFilter(event.target.value)}
            sx={{ minWidth: 180 }}
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
        <LoadingSpinner minHeight={160} />
      ) : displayedPlayers.length === 0 ? (
        <Typography color="text.secondary">No players match your search/filter.</Typography>
      ) : (
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
          {displayedPlayers.map((player) => (
            <IdentifiedPlayerCard key={player.name} player={player} />
          ))}
        </Box>
      )}

      <AddPlayerDialog open={addPlayerOpen} onClose={() => setAddPlayerOpen(false)} onAdd={handleAddToRoster} />
    </Box>
  );
}
