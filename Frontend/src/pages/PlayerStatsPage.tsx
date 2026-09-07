import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Grid from "@mui/material/Grid";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutlined";
import PersonIcon from "@mui/icons-material/Person";
import { Link as RouterLink, Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Job, RadarPoint } from "../lib/types";
import { RadarChart } from "../components/results/RadarChart";
import { RowsSkeleton, StatTilesSkeleton } from "../components/Skeletons";

const ACTION_COLORS: Record<string, string> = {
  serve: "#3b82f6",
  spike: "#f97316",
  set: "#22c55e",
  dig: "#eab308",
  block: "#ec4899",
  hit: "#10b981",
};

interface VideoBreakdown {
  jobId: string;
  jobTitle: string;
  totalHits: number;
  ralliesParticipated: number;
  rallyEndingTouches: number;
}

interface PlayerProfile {
  thumbnail: string | null;
  totalHits: number;
  ralliesParticipated: number;
  rallyEndingTouches: number;
  hitsByType: Record<string, number>;
  videos: VideoBreakdown[];
}

async function loadProfile(name: string, completeJobs: Job[]): Promise<PlayerProfile> {
  const perJob = await Promise.all(
    completeJobs.map(async (job) => {
      const results = await api.getResults(job.id).catch(() => null);
      return { job, results };
    }),
  );

  const profile: PlayerProfile = {
    thumbnail: null,
    totalHits: 0,
    ralliesParticipated: 0,
    rallyEndingTouches: 0,
    hitsByType: {},
    videos: [],
  };

  for (const { job, results } of perJob) {
    if (!results) continue;
    const entry = Object.entries(results.players).find(([, stat]) => stat.name?.trim() === name);
    if (!entry) continue;
    const [stableIdStr, stat] = entry;

    profile.totalHits += stat.total_hits;
    profile.ralliesParticipated += stat.rallies_participated;
    profile.rallyEndingTouches += stat.rally_ending_touches;
    for (const [type, count] of Object.entries(stat.hits_by_type)) {
      profile.hitsByType[type] = (profile.hitsByType[type] ?? 0) + count;
    }
    profile.videos.push({
      jobId: job.id,
      jobTitle: job.original_filename,
      totalHits: stat.total_hits,
      ralliesParticipated: stat.rallies_participated,
      rallyEndingTouches: stat.rally_ending_touches,
    });

    if (!profile.thumbnail) {
      const players = await api.getPlayers(job.id).catch(() => null);
      const match = players?.players.find((p) => String(p.stable_id) === stableIdStr);
      if (match?.thumbnail_base64) profile.thumbnail = match.thumbnail_base64;
    }
  }

  profile.videos.sort((a, b) => b.totalHits - a.totalHits);
  return profile;
}

function StatTile({ value, label }: { value: string | number; label: string }) {
  return (
    <Grid size={{ xs: 6, sm: 3 }}>
      <Card variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          {value}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
      </Card>
    </Grid>
  );
}

interface PlayerStatsPageProps {
  jobs: Job[];
}

// One player's stats rolled up across every video they appear in -
// reached by clicking a player card on the Players page.
export function PlayerStatsPage({ jobs }: PlayerStatsPageProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const name = searchParams.get("name");
  const [profile, setProfile] = useState<PlayerProfile | null>(null);
  const [radar, setRadar] = useState<RadarPoint[] | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const completeJobs = jobs.filter((j) => j.status === "complete");
  const completeJobIds = completeJobs.map((j) => j.id).join(",");

  useEffect(() => {
    document.title = name ?? "Volleyball Metrics";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [name]);

  useEffect(() => {
    if (!name) return;
    let cancelled = false;
    setProfile(null);
    loadProfile(name, completeJobs).then((res) => !cancelled && setProfile(res));
    return () => {
      cancelled = true;
    };
    // completeJobIds is a stable proxy for completeJobs's identity - see
    // PlayersPage for why this can't just depend on completeJobs itself.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name, completeJobIds]);

  useEffect(() => {
    if (!name) return;
    let cancelled = false;
    setRadar(null);
    api
      .getPlayerRadar(name)
      .then((res) => !cancelled && setRadar(res.radar))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [name]);

  if (!name) return <Navigate to="/players" replace />;

  async function handleConfirmDelete() {
    if (!name) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.removeFromRoster(name);
      navigate("/players");
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
      setDeleting(false);
    }
  }

  const hitsByTypeTotal = profile ? Object.values(profile.hitsByType).reduce((a, b) => a + b, 0) : 0;
  const rallyEndingTouchRate =
    profile && profile.ralliesParticipated > 0 ? profile.rallyEndingTouches / profile.ralliesParticipated : null;

  return (
    <Box sx={{ maxWidth: 900 }}>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 2 }}>
        <Button component={RouterLink} to="/players" startIcon={<ArrowBackIcon />}>
          Players
        </Button>
        <Button color="error" startIcon={<DeleteOutlineIcon />} onClick={() => setDeleteOpen(true)}>
          Delete
        </Button>
      </Stack>

      {profile === null ? (
        <>
          <Stack direction="row" spacing={3} sx={{ alignItems: "center", mb: 4 }}>
            <Skeleton variant="rounded" width={96} height={120} />
            <Skeleton variant="text" width={200} height={48} />
          </Stack>
          <Box sx={{ mb: 4 }}>
            <StatTilesSkeleton count={4} size={{ xs: 6, sm: 3 }} />
          </Box>
          <Skeleton variant="text" width={80} sx={{ mb: 1 }} />
          <RowsSkeleton count={3} />
        </>
      ) : (
        <>
          <Stack direction="row" spacing={3} sx={{ alignItems: "center", mb: 4 }}>
            <Box sx={{ width: 96, height: 120, borderRadius: 2, overflow: "hidden", bgcolor: "action.hover", flexShrink: 0 }}>
              {profile.thumbnail ? (
                <Box
                  component="img"
                  src={`data:image/jpeg;base64,${profile.thumbnail}`}
                  alt={name}
                  sx={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
                />
              ) : (
                <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <PersonIcon sx={{ fontSize: 40, color: "text.disabled" }} />
                </Box>
              )}
            </Box>
            <Typography variant="h4" sx={{ fontWeight: 700 }}>
              {name}
            </Typography>
          </Stack>

          <Grid container spacing={2} sx={{ mb: 4 }}>
            <StatTile value={profile.totalHits} label="Total hits" />
            <StatTile value={profile.ralliesParticipated} label="Rallies participated" />
            <StatTile value={profile.videos.length} label="Videos" />
            <StatTile
              value={rallyEndingTouchRate != null ? `${(rallyEndingTouchRate * 100).toFixed(0)}%` : "-"}
              label="Rally-ending touch rate"
            />
          </Grid>

          {hitsByTypeTotal > 0 && (
            <Box sx={{ mb: 4 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Action mix
              </Typography>
              <Stack direction="row" sx={{ height: 10, borderRadius: 999, overflow: "hidden" }}>
                {Object.entries(profile.hitsByType).map(([type, count]) => (
                  <Box
                    key={type}
                    sx={{ width: `${(count / hitsByTypeTotal) * 100}%`, bgcolor: ACTION_COLORS[type] ?? "grey.500" }}
                  />
                ))}
              </Stack>
              <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }}>
                {Object.entries(profile.hitsByType).map(([type, count]) => (
                  <Stack key={type} direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: ACTION_COLORS[type] ?? "grey.500" }} />
                    <Typography variant="caption" color="text.secondary">
                      {type} ({count})
                    </Typography>
                  </Stack>
                ))}
              </Stack>
            </Box>
          )}

          {radar?.some((p) => p.sample_size_a > 0) && (
            <Box sx={{ mb: 4 }}>
              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                Win % by action
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                How often {name}'s side won the rally after they performed each action, across every video with a
                usable team split.
              </Typography>
              <RadarChart radar={radar} />
            </Box>
          )}

          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            By video
          </Typography>
          {profile.videos.length === 0 ? (
            <Typography color="text.secondary">Not recorded in any finished video yet.</Typography>
          ) : (
            <Stack spacing={1}>
              {profile.videos.map((v) => (
                <Card
                  key={v.jobId}
                  variant="outlined"
                  sx={{ p: 1.5, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}
                >
                  <Box sx={{ minWidth: 0 }}>
                    <Typography noWrap sx={{ fontWeight: 600 }}>
                      {v.jobTitle}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {v.totalHits} hits · {v.ralliesParticipated} rallies · {v.rallyEndingTouches} rally-ending
                    </Typography>
                  </Box>
                  <Button size="small" component={RouterLink} to={`/video?job=${v.jobId}`} sx={{ flexShrink: 0 }}>
                    Open
                  </Button>
                </Card>
              ))}
            </Stack>
          )}
        </>
      )}

      <Dialog open={deleteOpen} onClose={() => setDeleteOpen(false)}>
        <DialogTitle>Delete this player?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This removes "{name}" from the roster. It doesn't un-name them on any video they're already tagged in -
            do that from the video's Setup tab instead.
          </DialogContentText>
          {deleteError && (
            <Typography variant="caption" color="error" sx={{ display: "block", mt: 2 }}>
              {deleteError}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteOpen(false)} disabled={deleting}>
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
