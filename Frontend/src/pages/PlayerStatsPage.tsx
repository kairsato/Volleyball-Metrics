import { useEffect, useMemo, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import FormControl from "@mui/material/FormControl";
import Grid from "@mui/material/Grid";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutlined";
import PersonIcon from "@mui/icons-material/Person";
import { Link as RouterLink, Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { PlayerGame, PlayerProfile } from "../lib/types";
import { RadarChart } from "../components/results/RadarChart";
import { ScoreRadarChart, type ScoreRadarPoint } from "../components/results/ScoreRadarChart";
import { RowsSkeleton, StatTilesSkeleton } from "../components/Skeletons";

const ACTION_COLORS: Record<string, string> = {
  serve: "#3b82f6",
  spike: "#f97316",
  set: "#22c55e",
  dig: "#eab308",
  block: "#ec4899",
  hit: "#10b981",
};

const QUALITY_CATEGORY_KEYS = ["serve", "receive", "set", "spike", "block"] as const;
const QUALITY_CATEGORY_LABELS: Record<string, string> = {
  serve: "Serve",
  receive: "Receive",
  set: "Set",
  spike: "Spike",
  block: "Block",
};

const GOOD_COLOR = "#22c55e";
const BAD_COLOR = "#ef4444";
const MEDIAN_GAP_THRESHOLD = 0.03;

function formatPercent(value: number | null): string {
  return value !== null ? `${Math.round(value * 100)}%` : "-";
}

function StatTile({ value, label }: { value: string | number; label: string }) {
  return (
    <Grid size={{ xs: 6, sm: 4, md: 12 / 5 }}>
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

// --------------------------------------------------------------------------
// Win/loss trend - cumulative win-minus-loss differential across every game
// this player is resolvable onto a team roster for, in chronological order.
// Same "climbing/falling slope reads as a streak" idea as
// results/WinLossTrend.tsx's within-game momentum chart, just one point per
// GAME instead of per rally, and filterable to a single team since a player
// can belong to more than one.
// --------------------------------------------------------------------------

const TREND_WIDTH = 520;
const TREND_HEIGHT = 140;
const TREND_PADDING = 14;

function PlayerWinLossTrend({ games }: { games: PlayerGame[] }) {
  if (games.length === 0) return null;

  const series: number[] = [];
  for (const game of games) {
    const previous = series.length > 0 ? series[series.length - 1] : 0;
    series.push(previous + (game.result === "win" ? 1 : game.result === "loss" ? -1 : 0));
  }

  const maxAbs = Math.max(1, ...series.map((v) => Math.abs(v)));
  const stepX = games.length > 1 ? (TREND_WIDTH - TREND_PADDING * 2) / (games.length - 1) : 0;
  const scaleY = (TREND_HEIGHT / 2 - TREND_PADDING) / maxAbs;
  const midY = TREND_HEIGHT / 2;

  const coords = series.map((v, i) => ({ x: TREND_PADDING + i * stepX, y: midY - v * scaleY }));
  const linePoints = coords.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <Box
      component="svg"
      viewBox={`0 0 ${TREND_WIDTH} ${TREND_HEIGHT}`}
      sx={{ width: "100%", display: "block", color: "text.secondary" }}
    >
      <line
        x1={TREND_PADDING}
        y1={midY}
        x2={TREND_WIDTH - TREND_PADDING}
        y2={midY}
        stroke="currentColor"
        strokeOpacity={0.25}
        strokeDasharray="3 3"
      />
      <polyline points={linePoints} fill="none" stroke="currentColor" strokeOpacity={0.5} strokeWidth={1.5} />
      {coords.map((p, i) => {
        const color = games[i].result === "win" ? GOOD_COLOR : games[i].result === "loss" ? BAD_COLOR : "currentColor";
        return (
          <g key={games[i].job_id}>
            <circle cx={p.x} cy={p.y} r={4} fill={color} />
            <title>
              {games[i].original_filename} - {games[i].result ?? "undecided"}
            </title>
          </g>
        );
      })}
    </Box>
  );
}

interface WinLossSectionProps {
  games: PlayerGame[];
}

function WinLossSection({ games }: WinLossSectionProps) {
  const teamOptions = useMemo(() => {
    const byId = new Map<string, string>();
    for (const g of games) {
      if (g.team_id && g.team_name) byId.set(g.team_id, g.team_name);
    }
    return Array.from(byId.entries()).map(([id, name]) => ({ id, name }));
  }, [games]);

  const [teamFilter, setTeamFilter] = useState<string>("all");

  const decidedGames = useMemo(() => {
    return games
      .filter((g) => g.result !== null && (teamFilter === "all" || g.team_id === teamFilter))
      // Chronological (oldest first) for the trend - games arrive newest-first.
      .slice()
      .reverse();
  }, [games, teamFilter]);

  const wins = decidedGames.filter((g) => g.result === "win").length;
  const losses = decidedGames.filter((g) => g.result === "loss").length;

  if (teamOptions.length === 0) {
    return null;
  }

  return (
    <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 1.5, flexWrap: "wrap", gap: 1 }}>
        <Typography variant="subtitle2">Win/loss trend</Typography>
        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel id="player-team-filter-label">Team</InputLabel>
          <Select
            labelId="player-team-filter-label"
            label="Team"
            value={teamFilter}
            onChange={(event) => setTeamFilter(event.target.value)}
          >
            <MenuItem value="all">Every team</MenuItem>
            {teamOptions.map((t) => (
              <MenuItem key={t.id} value={t.id}>
                {t.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Stack>

      {decidedGames.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No decided games for this team yet.
        </Typography>
      ) : (
        <>
          <Stack direction="row" spacing={3} sx={{ mb: 2 }}>
            <Box>
              <Typography variant="h5" sx={{ fontWeight: 700, color: GOOD_COLOR }}>
                {wins}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Wins
              </Typography>
            </Box>
            <Box>
              <Typography variant="h5" sx={{ fontWeight: 700, color: BAD_COLOR }}>
                {losses}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Losses
              </Typography>
            </Box>
            <Box>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                {wins + losses > 0 ? `${Math.round((wins / (wins + losses)) * 100)}%` : "-"}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Win rate
              </Typography>
            </Box>
          </Stack>
          <PlayerWinLossTrend games={decidedGames} />
        </>
      )}
    </Card>
  );
}

// --------------------------------------------------------------------------
// Action mix by game - a compact stacked bar per game (most recent first),
// so a shift in what a player's actually doing game to game (more digs
// lately, fewer serves) is visible at a glance rather than buried in the
// per-game list below.
// --------------------------------------------------------------------------

function ActionMixByGame({ games }: { games: PlayerGame[] }) {
  const shown = games.filter((g) => g.total_hits > 0).slice(0, 8);
  if (shown.length === 0) return null;

  return (
    <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
      <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
        Action mix by game
      </Typography>
      <Stack spacing={1.25}>
        {shown.map((g) => (
          <Stack key={g.job_id} direction="row" spacing={1.5} sx={{ alignItems: "center" }}>
            <Typography variant="caption" color="text.secondary" noWrap sx={{ width: 130, flexShrink: 0 }}>
              {g.original_filename}
            </Typography>
            <Stack direction="row" sx={{ flex: 1, height: 8, borderRadius: 999, overflow: "hidden" }}>
              {Object.entries(g.hits_by_type)
                .filter(([, count]) => count > 0)
                .map(([type, count]) => (
                  <Box
                    key={type}
                    sx={{ width: `${(count / g.total_hits) * 100}%`, bgcolor: ACTION_COLORS[type] ?? "grey.500" }}
                    title={`${type}: ${count}`}
                  />
                ))}
            </Stack>
            <Typography variant="caption" color="text.secondary" sx={{ width: 34, textAlign: "right", flexShrink: 0 }}>
              {g.total_hits}
            </Typography>
          </Stack>
        ))}
      </Stack>
    </Card>
  );
}

// --------------------------------------------------------------------------
// Quality profile - the same category-card language AnalyticsTab uses for a
// single video's action quality, applied across every game via
// PlayerProfile.quality_categories. A "Show" selector switches between the
// cross-game aggregate ("Overall") and one specific game's score, mirroring
// AnalyticsTab's own player filter.
// --------------------------------------------------------------------------

function QualitySection({ profile }: { profile: PlayerProfile }) {
  const [showGame, setShowGame] = useState<string>("overall");

  const hasAnyQuality = Object.values(profile.quality_categories).some((c) => c.count > 0);
  if (!hasAnyQuality) return null;

  const gameOptions = profile.games.filter((g) =>
    Object.values(profile.quality_categories).some((c) => c.matches.some((m) => m.job_id === g.job_id)),
  );

  function scoreFor(key: string): { value: number | null; medianValue: number | null } {
    const category = profile.quality_categories[key];
    if (!category) return { value: null, medianValue: null };
    if (showGame === "overall") return { value: category.average_score, medianValue: category.median_score };
    const match = category.matches.find((m) => m.job_id === showGame);
    return { value: match?.average_score ?? null, medianValue: category.median_score };
  }

  const radarPoints: ScoreRadarPoint[] = QUALITY_CATEGORY_KEYS.map((key) => ({
    key,
    label: QUALITY_CATEGORY_LABELS[key],
    value: scoreFor(key).value,
  }));
  const hasRadarData = radarPoints.some((p) => p.value !== null);

  return (
    <Box sx={{ mb: 3 }}>
      <Stack direction="row" spacing={2} sx={{ mb: 2, alignItems: "center", flexWrap: "wrap" }}>
        <Typography variant="subtitle2">Action quality</Typography>
        <FormControl size="small" sx={{ minWidth: 200, ml: "auto" }}>
          <InputLabel id="player-quality-game-label">Show</InputLabel>
          <Select
            labelId="player-quality-game-label"
            label="Show"
            value={showGame}
            onChange={(event) => setShowGame(event.target.value)}
          >
            <MenuItem value="overall">Overall</MenuItem>
            {gameOptions.map((g) => (
              <MenuItem key={g.job_id} value={g.job_id}>
                {g.original_filename}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Stack>

      {hasRadarData && (
        <Card variant="outlined" sx={{ p: 2.5, mb: 2 }}>
          <ScoreRadarChart points={radarPoints} />
        </Card>
      )}

      <Grid container spacing={2}>
        {QUALITY_CATEGORY_KEYS.map((key) => {
          const category = profile.quality_categories[key];
          if (!category || category.count === 0) return null;
          const { value, medianValue } = scoreFor(key);
          const diff = value !== null && medianValue !== null ? value - medianValue : null;
          const isNeutral = diff !== null && Math.abs(diff) < MEDIAN_GAP_THRESHOLD;

          return (
            <Grid key={key} size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined" sx={{ p: 2 }}>
                <Stack direction="row" sx={{ alignItems: "baseline", justifyContent: "space-between", mb: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {QUALITY_CATEGORY_LABELS[key]}
                  </Typography>
                  <Stack sx={{ alignItems: "flex-end" }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
                      {formatPercent(value)}
                    </Typography>
                    {medianValue !== null && (
                      <Typography variant="caption" color="text.secondary">
                        own median {formatPercent(medianValue)}
                      </Typography>
                    )}
                  </Stack>
                </Stack>
                {showGame === "overall" ? (
                  <Stack spacing={0.5}>
                    {category.matches.map((m) => {
                      const mDiff = category.median_score !== null ? m.average_score - category.median_score : null;
                      const mNeutral = mDiff !== null && Math.abs(mDiff) < MEDIAN_GAP_THRESHOLD;
                      return (
                        <Stack key={m.job_id} direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
                          <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0 }}>
                            {m.original_filename}
                          </Typography>
                          <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexShrink: 0 }}>
                            <Typography variant="caption">{formatPercent(m.average_score)}</Typography>
                            {mDiff !== null && (
                              <Typography
                                variant="caption"
                                sx={{
                                  color: mNeutral ? "text.secondary" : mDiff > 0 ? GOOD_COLOR : BAD_COLOR,
                                  fontWeight: mNeutral ? 400 : 600,
                                  minWidth: 44,
                                  textAlign: "right",
                                }}
                              >
                                {mNeutral ? "at median" : `${mDiff > 0 ? "+" : ""}${Math.round(mDiff * 100)}%`}
                              </Typography>
                            )}
                          </Stack>
                        </Stack>
                      );
                    })}
                  </Stack>
                ) : (
                  diff !== null && (
                    <Typography variant="caption" sx={{ color: isNeutral ? "text.secondary" : diff > 0 ? GOOD_COLOR : BAD_COLOR }}>
                      {isNeutral ? "at their own median" : `${diff > 0 ? "+" : ""}${Math.round(diff * 100)}% vs their own median`}
                    </Typography>
                  )
                )}
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}

// One player's stats rolled up across every game they appear in - reached
// by clicking a player card on the Players page. Everything here comes
// from a single api.getPlayerProfile(name) call, backed by Backend/API/
// player_profiles.py's persisted store - no client-side cross-game
// aggregation happens on this page anymore.
export function PlayerStatsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const name = searchParams.get("name");
  const [profile, setProfile] = useState<PlayerProfile | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

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
    api
      .getPlayerProfile(name)
      .then((res) => !cancelled && setProfile(res))
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

  const hitsByTypeTotal = profile ? Object.values(profile.hits_by_type).reduce((a, b) => a + b, 0) : 0;
  const rallyEndingTouchRate =
    profile && profile.rallies_participated > 0 ? profile.rally_ending_touches / profile.rallies_participated : null;
  const decidedGames = profile ? profile.games.filter((g) => g.result !== null) : [];
  const totalWins = decidedGames.filter((g) => g.result === "win").length;
  const totalLosses = decidedGames.filter((g) => g.result === "loss").length;

  return (
    <Box sx={{ maxWidth: 1100 }}>
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
            <StatTilesSkeleton count={5} />
          </Box>
          <Skeleton variant="text" width={80} sx={{ mb: 1 }} />
          <RowsSkeleton count={3} />
        </>
      ) : (
        <>
          <Stack direction="row" spacing={3} sx={{ alignItems: "center", mb: 4 }}>
            <Box sx={{ width: 96, height: 120, borderRadius: 2, overflow: "hidden", bgcolor: "action.hover", flexShrink: 0 }}>
              {profile.thumbnail_base64 ? (
                <Box
                  component="img"
                  src={`data:image/jpeg;base64,${profile.thumbnail_base64}`}
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
            <StatTile value={profile.total_hits} label="Total hits" />
            <StatTile value={profile.rallies_participated} label="Rallies participated" />
            <StatTile value={profile.games.length} label="Games" />
            <StatTile
              value={rallyEndingTouchRate != null ? `${(rallyEndingTouchRate * 100).toFixed(0)}%` : "-"}
              label="Rally-ending touch rate"
            />
            <StatTile
              value={totalWins + totalLosses > 0 ? `${totalWins}-${totalLosses}` : "-"}
              label="Overall win-loss"
            />
          </Grid>

          {hitsByTypeTotal > 0 && (
            <Box sx={{ mb: 4 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Action mix
              </Typography>
              <Stack direction="row" sx={{ height: 10, borderRadius: 999, overflow: "hidden" }}>
                {Object.entries(profile.hits_by_type).map(([type, count]) => (
                  <Box
                    key={type}
                    sx={{ width: `${(count / hitsByTypeTotal) * 100}%`, bgcolor: ACTION_COLORS[type] ?? "grey.500" }}
                  />
                ))}
              </Stack>
              <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }}>
                {Object.entries(profile.hits_by_type).map(([type, count]) => (
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

          <WinLossSection games={profile.games} />

          <ActionMixByGame games={profile.games} />

          <QualitySection profile={profile} />

          {profile.radar.some((p) => p.sample_size_a > 0) && (
            <Card variant="outlined" sx={{ p: 2.5, mb: 4 }}>
              <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                Win % by action
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                How often {name}'s side won the rally after they performed each action, across every video with a
                usable team split.
              </Typography>
              <RadarChart radar={profile.radar} />
            </Card>
          )}

          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            By game
          </Typography>
          {profile.games.length === 0 ? (
            <Typography color="text.secondary">Not recorded in any finished video yet.</Typography>
          ) : (
            <Stack spacing={1}>
              {profile.games.map((v) => (
                <Card
                  key={v.job_id}
                  variant="outlined"
                  sx={{ p: 1.5, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 2 }}
                >
                  <Box sx={{ minWidth: 0 }}>
                    <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                      <Typography noWrap sx={{ fontWeight: 600 }}>
                        {v.original_filename}
                      </Typography>
                      {v.result && (
                        <Chip
                          size="small"
                          label={v.result === "win" ? "Win" : "Loss"}
                          sx={{
                            bgcolor: v.result === "win" ? GOOD_COLOR : BAD_COLOR,
                            color: "#fff",
                            height: 20,
                            "& .MuiChip-label": { px: 1, fontSize: 11 },
                          }}
                        />
                      )}
                    </Stack>
                    <Typography variant="caption" color="text.secondary">
                      {v.total_hits} hits · {v.rallies_participated} rallies · {v.rally_ending_touches} rally-ending
                    </Typography>
                  </Box>
                  <Button size="small" component={RouterLink} to={`/game?job=${v.job_id}`} sx={{ flexShrink: 0 }}>
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
