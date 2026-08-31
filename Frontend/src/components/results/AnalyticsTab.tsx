import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { api } from "../../lib/api";
import type { ActionQualityCategory, ActionQualityOut, ResultsOut } from "../../lib/types";
import { LoadingSpinner } from "../LoadingSpinner";
import { CourtMap } from "./CourtMap";
import { playerInsight, type ActionCategoryKey } from "./insights";
import { recomputeCategory } from "./weights";

function formatPercent(value: number | null): string {
  return value !== null ? `${Math.round(value * 100)}%` : "-";
}

function factorLabel(key: string): string {
  return key.replace(/_/g, " ");
}

// category-wide by default; restricted to one player's own instances when
// stableId is given - the same shape either way, so ScoreBar rows don't
// need to know which case they're in.
function averageFactor(category: ActionQualityCategory, key: string, stableId: number | null): number | null {
  const instances = stableId === null ? category.instances : category.instances.filter((i) => i.player_stable_id === stableId);
  const values = instances.map((i) => i.factors[key]).filter((v): v is number => v !== null && v !== undefined);
  return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function ScoreBar({ value }: { value: number | null }) {
  return (
    <Stack direction="row" spacing={1} sx={{ alignItems: "center", flex: 1 }}>
      <Box sx={{ flex: 1, height: 6, borderRadius: 999, bgcolor: "action.hover", overflow: "hidden" }}>
        <Box sx={{ width: `${value !== null ? value * 100 : 0}%`, height: "100%", bgcolor: "primary.main" }} />
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ width: 34, textAlign: "right", flexShrink: 0 }}>
        {formatPercent(value)}
      </Typography>
    </Stack>
  );
}

interface QualitySectionProps {
  title: string;
  categoryKey: ActionCategoryKey;
  category: ActionQualityCategory;
  weights: Record<string, number>;
  onWeightChange: (factorKey: string, value: number) => void;
  onResetWeights: () => void;
  playerNames: Record<string, string | undefined>;
  playerFilter: number | null;
  onSeek: (timeS: number) => void;
}

// One card per action type (serve/receive/set/spike) - overall weighted
// score up top (recomputed live from the adjustable weights below, not the
// server's fixed defaults), the sub-factor breakdown, whoever recorded it
// (or just the one selected player), and - for Set - a court map of where
// it was actually hit from.
function QualitySection({
  title,
  categoryKey,
  category,
  weights,
  onWeightChange,
  onResetWeights,
  playerNames,
  playerFilter,
  onSeek,
}: QualitySectionProps) {
  const recomputed = useMemo(() => recomputeCategory(category, weights), [category, weights]);

  if (category.count === 0) {
    return (
      <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          No {title.toLowerCase()} actions detected.
        </Typography>
      </Card>
    );
  }

  const displayScore = playerFilter === null ? recomputed.averageScore : (recomputed.players[playerFilter]?.averageScore ?? null);

  const rankedPlayers = Object.entries(recomputed.players)
    .map(([stableId, stats]) => ({ stableId: Number(stableId), ...stats }))
    .sort((a, b) => (b.averageScore ?? -1) - (a.averageScore ?? -1));

  const shownPlayers = playerFilter === null ? rankedPlayers.slice(0, 6) : rankedPlayers.filter((p) => p.stableId === playerFilter);

  return (
    <Card variant="outlined" sx={{ p: 2.5, mb: 3 }}>
      <Stack direction="row" sx={{ alignItems: "baseline", justifyContent: "space-between", mb: 1.5 }}>
        <Typography variant="subtitle2">{title}</Typography>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          {formatPercent(displayScore)}
        </Typography>
      </Stack>

      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          Weights (drag to see scores update)
        </Typography>
        <Button size="small" onClick={onResetWeights}>
          Reset
        </Button>
      </Stack>
      <Stack spacing={1} sx={{ mb: 2 }}>
        {Object.keys(category.weights).map((key) => (
          <Stack key={key} direction="row" spacing={1.5} sx={{ alignItems: "center" }}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ width: 130, flexShrink: 0, textTransform: "capitalize" }}
            >
              {factorLabel(key)}
            </Typography>
            <Slider
              size="small"
              min={0}
              max={1}
              step={0.05}
              value={weights[key] ?? 0}
              onChange={(_, value) => onWeightChange(key, value as number)}
              sx={{ flex: 1 }}
            />
            <ScoreBar value={averageFactor(category, key, playerFilter)} />
          </Stack>
        ))}
      </Stack>

      {categoryKey === "set" && <CourtMap category={category} playerNames={playerNames} onSeek={onSeek} />}

      <Stack spacing={1}>
        {shownPlayers.map((player) => {
          const firstInstance = category.instances.find((i) => i.player_stable_id === player.stableId);
          const insight = playerInsight(category, categoryKey, player.stableId);
          return (
            <Box key={player.stableId}>
              <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between" }}>
                <Typography variant="body2">{playerNames[String(player.stableId)] ?? `Player ${player.stableId}`}</Typography>
                <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                  <Typography variant="caption" color="text.secondary">
                    {formatPercent(player.averageScore)} · {player.count}
                  </Typography>
                  {firstInstance && (
                    <Button size="small" onClick={() => onSeek(firstInstance.timestamp_s)}>
                      Jump in
                    </Button>
                  )}
                </Stack>
              </Stack>
              {insight && (
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", fontStyle: "italic" }}>
                  {insight}
                </Typography>
              )}
            </Box>
          );
        })}
        {shownPlayers.length === 0 && (
          <Typography variant="caption" color="text.secondary">
            No recorded touches for this player.
          </Typography>
        )}
      </Stack>
    </Card>
  );
}

interface AnalyticsTabProps {
  results: ResultsOut;
  onSeek: (timeS: number) => void;
}

const CATEGORY_KEYS: ActionCategoryKey[] = ["serve", "receive", "set", "spike"];

export function AnalyticsTab({ results, onSeek }: AnalyticsTabProps) {
  const [quality, setQuality] = useState<ActionQualityOut | null>(null);
  const [qualityLoading, setQualityLoading] = useState(true);
  const [weights, setWeights] = useState<Record<ActionCategoryKey, Record<string, number>> | null>(null);
  const [playerFilter, setPlayerFilter] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getActionQuality(results.job_id)
      .then((res) => {
        if (cancelled) return;
        setQuality(res);
        setWeights({
          serve: { ...res.serve.weights },
          receive: { ...res.receive.weights },
          set: { ...res.set.weights },
          spike: { ...res.spike.weights },
        });
      })
      .catch(() => undefined)
      .finally(() => !cancelled && setQualityLoading(false));
    return () => {
      cancelled = true;
    };
  }, [results.job_id]);

  const playerEntries = Object.entries(results.players);
  const playerNames = Object.fromEntries(playerEntries.map(([id, stat]) => [id, stat.name ?? undefined]));

  if (qualityLoading) {
    return (
      <Card variant="outlined">
        <LoadingSpinner minHeight={140} />
      </Card>
    );
  }

  if (!quality || !weights) {
    return <Typography color="text.secondary">Action quality data isn't available for this video.</Typography>;
  }

  return (
    <Box>
      <FormControl size="small" sx={{ mb: 2, minWidth: 200 }}>
        <InputLabel id="analytics-player-filter-label">Show</InputLabel>
        <Select
          labelId="analytics-player-filter-label"
          label="Show"
          value={playerFilter === null ? "everyone" : String(playerFilter)}
          onChange={(event) => setPlayerFilter(event.target.value === "everyone" ? null : Number(event.target.value))}
        >
          <MenuItem value="everyone">Everyone</MenuItem>
          {playerEntries.map(([id, stat]) => (
            <MenuItem key={id} value={id}>
              {stat.name ?? `Player ${id}`}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {(!quality.teams_available || !quality.height_available) && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {!quality.teams_available &&
            "Team splitting isn't available for this video, so placement/blocker factors below are skipped. "}
          {!quality.height_available &&
            "Trajectory-height factors aren't available - mark the net-top points in Court Calibration to enable them."}
        </Alert>
      )}

      {CATEGORY_KEYS.map((key) => (
        <QualitySection
          key={key}
          title={key.charAt(0).toUpperCase() + key.slice(1)}
          categoryKey={key}
          category={quality[key]}
          weights={weights[key]}
          onWeightChange={(factorKey, value) =>
            setWeights((prev) => (prev ? { ...prev, [key]: { ...prev[key], [factorKey]: value } } : prev))
          }
          onResetWeights={() => setWeights((prev) => (prev ? { ...prev, [key]: { ...quality[key].weights } } : prev))}
          playerNames={playerNames}
          playerFilter={playerFilter}
          onSeek={onSeek}
        />
      ))}
    </Box>
  );
}
