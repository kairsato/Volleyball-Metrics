import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Grid from "@mui/material/Grid";
import PersonIcon from "@mui/icons-material/Person";
import Typography from "@mui/material/Typography";
import { api } from "../lib/api";
import type { Job } from "../lib/types";
import { LoadingSpinner } from "../components/LoadingSpinner";
import { PlayerRoster } from "../components/PlayerRoster";

interface PlayerAggregate {
  name: string;
  thumbnail: string | null;
  totalHits: number;
  ralliesParticipated: number;
  videoCount: number;
}

async function loadAggregates(completeJobs: Job[]): Promise<PlayerAggregate[]> {
  const perJob = await Promise.all(
    completeJobs.map(async (job) => {
      const [results, players] = await Promise.all([
        api.getResults(job.id).catch(() => null),
        api.getPlayers(job.id).catch(() => null),
      ]);
      return { results, players };
    }),
  );

  const byName = new Map<string, PlayerAggregate>();
  for (const { results, players } of perJob) {
    if (!results) continue;
    const thumbByStableId = new Map<number, string | null>();
    for (const p of players?.players ?? []) thumbByStableId.set(p.stable_id, p.thumbnail_base64);

    for (const [stableIdStr, stat] of Object.entries(results.players)) {
      const name = stat.name?.trim();
      if (!name) continue;

      const existing = byName.get(name) ?? {
        name,
        thumbnail: null,
        totalHits: 0,
        ralliesParticipated: 0,
        videoCount: 0,
      };
      existing.totalHits += stat.total_hits;
      existing.ralliesParticipated += stat.rallies_participated;
      existing.videoCount += 1;
      if (!existing.thumbnail) existing.thumbnail = thumbByStableId.get(Number(stableIdStr)) ?? null;
      byName.set(name, existing);
    }
  }

  return Array.from(byName.values()).sort((a, b) => b.totalHits - a.totalHits);
}

function PlayerCard({ player }: { player: PlayerAggregate }) {
  return (
    <Card variant="outlined" sx={{ overflow: "hidden", height: "100%" }}>
      <Box sx={{ width: "100%", aspectRatio: "4 / 5", bgcolor: "action.hover" }}>
        {player.thumbnail ? (
          <Box
            component="img"
            src={`data:image/jpeg;base64,${player.thumbnail}`}
            alt={player.name}
            sx={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
          />
        ) : (
          <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <PersonIcon sx={{ fontSize: 48, color: "text.disabled" }} />
          </Box>
        )}
      </Box>
      <Box sx={{ p: 1.5 }}>
        <Typography noWrap sx={{ fontWeight: 600 }}>
          {player.name}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
          {player.totalHits} hits · {player.ralliesParticipated} rallies
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {player.videoCount} video{player.videoCount === 1 ? "" : "s"}
        </Typography>
      </Box>
    </Card>
  );
}

interface StatsPageProps {
  jobs: Job[];
}

export function StatsPage({ jobs }: StatsPageProps) {
  const [players, setPlayers] = useState<PlayerAggregate[] | null>(null);

  const completeJobs = jobs.filter((j) => j.status === "complete");
  const completeJobIds = completeJobs.map((j) => j.id).join(",");

  useEffect(() => {
    let cancelled = false;
    setPlayers(null);
    loadAggregates(completeJobs).then((res) => !cancelled && setPlayers(res));
    return () => {
      cancelled = true;
    };
    // completeJobIds is a stable proxy for completeJobs's identity - re-fetching
    // on every jobs poll (which creates new array/object references every 3s)
    // would otherwise refetch results/players for every completed video constantly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completeJobIds]);

  return (
    <Box>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
        Stats
      </Typography>

      <PlayerRoster />

      <Typography variant="h5" sx={{ fontWeight: 600, mb: 2 }}>
        Players
      </Typography>

      {players === null ? (
        <LoadingSpinner minHeight={160} />
      ) : players.length === 0 ? (
        <Typography color="text.secondary">
          No named players yet - assign names to players on a finished video's Setup tab to see them here.
        </Typography>
      ) : (
        <Grid container spacing={2}>
          {players.map((player) => (
            <Grid key={player.name} size={{ xs: 6, sm: 4, md: 3, lg: 2 }}>
              <PlayerCard player={player} />
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
}
