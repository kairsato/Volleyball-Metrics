import { useEffect, useState } from "react";
import { keyframes } from "@emotion/react";
import Autocomplete from "@mui/material/Autocomplete";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutlined";
import EditIcon from "@mui/icons-material/Edit";
import GroupsIcon from "@mui/icons-material/Groups";
import { Link as RouterLink } from "react-router-dom";
import { api } from "../lib/api";
import type { Job, TeamEntry } from "../lib/types";
import { LoadingSpinner } from "../components/LoadingSpinner";
import { PageHeader } from "../components/PageHeader";
import { PlayerPhotoCarousel } from "../components/PlayerPhotoCarousel";

// Matches the Players page's card size (150x280) - 6 of them side by side
// is the box's fixed footprint regardless of how many players are on the
// team; fewer just leaves empty space, more triggers the auto-scroll strip.
const MEMBER_TILE_WIDTH = 150;
const TEAM_BOX_HEIGHT = 280;
const VISIBLE_MEMBER_COUNT = 6;
const TEAM_BOX_WIDTH = MEMBER_TILE_WIDTH * VISIBLE_MEMBER_COUNT;

// A gentle, deliberate pace rather than a fast ticker - this is meant to be
// glanced at, not read.
const SCROLL_SPEED_PX_PER_S = 25;
const SCROLL_START_DELAY_S = 2;

const scrollLeft = keyframes`
  from { transform: translateX(0); }
  to { transform: translateX(-50%); }
`;

// A horizontal strip of player photo tiles - static if everyone fits in
// the box's fixed 6-wide footprint, otherwise an auto-scrolling loop: sits
// still for 2s so the first few members are actually readable, then
// scrolls smoothly and continuously (the member list is duplicated back to
// back and the animation travels exactly one copy's width, so the loop
// point is seamless) rather than requiring anyone to drag/scroll manually.
function TeamMemberStrip({ players, thumbnailsByName }: { players: string[]; thumbnailsByName: Map<string, string[]> }) {
  const shouldScroll = players.length > VISIBLE_MEMBER_COUNT;
  const trackWidthPx = MEMBER_TILE_WIDTH * players.length;
  const durationS = trackWidthPx / SCROLL_SPEED_PX_PER_S;
  const displayPlayers = shouldScroll ? [...players, ...players] : players;

  return (
    <Box sx={{ width: "100%", height: "100%", overflow: "hidden" }}>
      <Box
        sx={{
          display: "flex",
          height: "100%",
          width: shouldScroll ? trackWidthPx * 2 : "100%",
          animation: shouldScroll ? `${scrollLeft} ${durationS}s linear ${SCROLL_START_DELAY_S}s infinite` : "none",
        }}
      >
        {displayPlayers.map((name, i) => (
          <Tooltip key={`${name}-${i}`} title={name}>
            <Box sx={{ flexShrink: 0, width: MEMBER_TILE_WIDTH, height: "100%" }}>
              <PlayerPhotoCarousel thumbnails={thumbnailsByName.get(name) ?? []} alt={name} />
            </Box>
          </Tooltip>
        ))}
      </Box>
    </Box>
  );
}

// Only needs names + thumbnails, not full stats, so this skips the
// heavier getResults call the Players page's aggregation makes - a plain
// getPlayers per completed job already carries both.
async function loadThumbnailsByName(completeJobs: Job[]): Promise<Map<string, string[]>> {
  const perJob = await Promise.all(completeJobs.map((job) => api.getPlayers(job.id).catch(() => null)));

  const byName = new Map<string, string[]>();
  for (const res of perJob) {
    if (!res) continue;
    for (const p of res.players) {
      const name = p.name?.trim();
      if (!name || p.ignored || !p.thumbnail_base64) continue;
      if (!byName.has(name)) byName.set(name, []);
      byName.get(name)!.push(p.thumbnail_base64);
    }
  }
  return byName;
}

interface TeamDialogProps {
  open: boolean;
  onClose: () => void;
  roster: string[];
  initial?: TeamEntry;
  onSave: (name: string, players: string[]) => Promise<void>;
}

function TeamDialog({ open, onClose, roster, initial, onSave }: TeamDialogProps) {
  const [name, setName] = useState(initial?.name ?? "");
  const [players, setPlayers] = useState<string[]>(initial?.players ?? []);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName(initial?.name ?? "");
      setPlayers(initial?.players ?? []);
      setError(null);
    }
  }, [open, initial]);

  async function handleSave() {
    const trimmed = name.trim();
    if (!trimmed) return;
    setSaving(true);
    setError(null);
    try {
      await onSave(trimmed, players);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog open={open} onClose={saving ? undefined : onClose} maxWidth="xs" fullWidth>
      <DialogTitle>{initial ? "Edit team" : "Add a team"}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <TextField
            autoFocus
            fullWidth
            label="Team name"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <Autocomplete
            multiple
            options={roster}
            value={players}
            onChange={(_, value) => setPlayers(value)}
            renderInput={(params) => <TextField {...params} label="Players" placeholder="Add a player" />}
          />
        </Stack>
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
        <Button variant="contained" disabled={saving || !name.trim()} onClick={() => void handleSave()}>
          {saving ? "Saving..." : "Save"}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// A fixed 900x280 box (six 150x280 player slots, matching the Players
// page's card size) - the "thumbnail" is the auto-scrolling strip above
// instead of a single image, with the team name overlaid bottom-left the
// same way a video's filename is. The whole card links to this team's
// stats page; Edit/Delete stay as their own icon buttons on top, each
// stopping propagation so clicking them doesn't also navigate.
function TeamCard({
  team,
  thumbnailsByName,
  onEdit,
  onRequestDelete,
}: {
  team: TeamEntry;
  thumbnailsByName: Map<string, string[]>;
  onEdit: () => void;
  onRequestDelete: () => void;
}) {
  return (
    <Card variant="outlined" sx={{ position: "relative", width: TEAM_BOX_WIDTH, height: TEAM_BOX_HEIGHT, overflow: "hidden" }}>
      <Stack direction="row" spacing={0.5} sx={{ position: "absolute", top: 6, right: 6, zIndex: 1 }}>
        <IconButton
          size="small"
          aria-label={`Edit ${team.name}`}
          onClick={(event) => {
            event.stopPropagation();
            onEdit();
          }}
          sx={{ color: "#fff", bgcolor: "rgba(0,0,0,0.4)", "&:hover": { bgcolor: "rgba(0,0,0,0.6)" } }}
        >
          <EditIcon fontSize="small" />
        </IconButton>
        <IconButton
          size="small"
          aria-label={`Delete ${team.name}`}
          onClick={(event) => {
            event.stopPropagation();
            onRequestDelete();
          }}
          sx={{ color: "error.main", bgcolor: "rgba(0,0,0,0.4)", "&:hover": { bgcolor: "rgba(0,0,0,0.6)" } }}
        >
          <DeleteOutlineIcon fontSize="small" />
        </IconButton>
      </Stack>

      <CardActionArea
        component={RouterLink}
        to={`/team?id=${team.id}`}
        sx={{ width: "100%", height: "100%", position: "relative", bgcolor: "action.hover" }}
      >
        {team.players.length === 0 ? (
          <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", p: 2 }}>
            <Typography color="text.secondary" align="center" variant="body2">
              No players yet - edit this team to add some.
            </Typography>
          </Box>
        ) : (
          <TeamMemberStrip players={team.players} thumbnailsByName={thumbnailsByName} />
        )}

        <Box
          sx={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            p: 1.5,
            background: "linear-gradient(to top, rgba(0,0,0,0.8), rgba(0,0,0,0) 100%)",
            pointerEvents: "none",
          }}
        >
          <Typography noWrap sx={{ fontWeight: 600, color: "#fff" }}>
            {team.name}
          </Typography>
          <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.8)" }}>
            {team.players.length} player{team.players.length === 1 ? "" : "s"}
          </Typography>
        </Box>
      </CardActionArea>
    </Card>
  );
}

type SortOrder = "name" | "player_count";

interface TeamsPageProps {
  jobs: Job[];
}

export function TeamsPage({ jobs }: TeamsPageProps) {
  const [teams, setTeams] = useState<TeamEntry[] | null>(null);
  const [roster, setRoster] = useState<string[]>([]);
  const [thumbnailsByName, setThumbnailsByName] = useState<Map<string, string[]>>(new Map());
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<TeamEntry | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<TeamEntry | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [sortOrder, setSortOrder] = useState<SortOrder>("name");

  const completeJobs = jobs.filter((j) => j.status === "complete");
  const completeJobIds = completeJobs.map((j) => j.id).join(",");

  useEffect(() => {
    api.getTeams().then((res) => setTeams(res.teams));
    api.getRoster().then((res) => setRoster(res.players));
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadThumbnailsByName(completeJobs).then((res) => !cancelled && setThumbnailsByName(res));
    return () => {
      cancelled = true;
    };
    // completeJobIds is a stable proxy for completeJobs's identity - see
    // PlayersPage for why this can't just depend on completeJobs itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completeJobIds]);

  async function handleCreate(name: string, players: string[]) {
    const res = await api.createTeam(name, players);
    setTeams(res.teams);
  }

  async function handleUpdate(name: string, players: string[]) {
    if (!editTarget) return;
    const res = await api.updateTeam(editTarget.id, name, players);
    setTeams(res.teams);
  }

  async function handleConfirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      const res = await api.deleteTeam(deleteTarget.id);
      setTeams(res.teams);
      setDeleteTarget(null);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeleting(false);
    }
  }

  const displayedTeams = (teams ?? [])
    .filter((team) => team.name.toLowerCase().includes(search.trim().toLowerCase()))
    .sort((a, b) => (sortOrder === "name" ? a.name.localeCompare(b.name) : b.players.length - a.players.length));

  return (
    <Box>
      <PageHeader title="Teams" addLabel="Add team" onAdd={() => setDialogOpen(true)}>
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
          <TextField
            size="small"
            placeholder="Search teams..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            sx={{ minWidth: 240 }}
          />
          <TextField
            select
            size="small"
            label="Sort by"
            value={sortOrder}
            onChange={(event) => setSortOrder(event.target.value as SortOrder)}
            sx={{ minWidth: 180 }}
          >
            <MenuItem value="name">Name</MenuItem>
            <MenuItem value="player_count">Player count</MenuItem>
          </TextField>
        </Stack>
      </PageHeader>

      {teams === null ? (
        <LoadingSpinner minHeight={160} />
      ) : (
        <>
          {teams.length === 0 && (
            <Typography color="text.secondary" sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
              <GroupsIcon fontSize="small" /> No teams yet - group roster players together to see them here.
            </Typography>
          )}
          {teams.length > 0 && displayedTeams.length === 0 && (
            <Typography color="text.secondary">No teams match your search.</Typography>
          )}
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
            {displayedTeams.map((team) => (
              <TeamCard
                key={team.id}
                team={team}
                thumbnailsByName={thumbnailsByName}
                onEdit={() => setEditTarget(team)}
                onRequestDelete={() => setDeleteTarget(team)}
              />
            ))}
          </Box>
        </>
      )}

      <TeamDialog open={dialogOpen} onClose={() => setDialogOpen(false)} roster={roster} onSave={handleCreate} />
      <TeamDialog
        open={editTarget !== null}
        onClose={() => setEditTarget(null)}
        roster={roster}
        initial={editTarget ?? undefined}
        onSave={handleUpdate}
      />

      <Dialog open={deleteTarget !== null} onClose={() => setDeleteTarget(null)}>
        <DialogTitle>Delete this team?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This removes "{deleteTarget?.name}" - it doesn't affect the players in it or anything they're named on
            in any video, just this grouping.
          </DialogContentText>
          {deleteError && (
            <Typography variant="caption" color="error" sx={{ display: "block", mt: 2 }}>
              {deleteError}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteTarget(null)} disabled={deleting}>
            Cancel
          </Button>
          <Button color="error" variant="contained" onClick={() => void handleConfirmDelete()} disabled={deleting}>
            {deleting ? "Deleting..." : "Delete"}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
