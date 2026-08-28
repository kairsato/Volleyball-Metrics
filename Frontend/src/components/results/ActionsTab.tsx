import { useEffect, useMemo, useRef, useState } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Checkbox from "@mui/material/Checkbox";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import type { SelectChangeEvent } from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import PlaylistPlayIcon from "@mui/icons-material/PlaylistPlay";
import type { ResultsOut } from "../../lib/types";
import type { FlatEvent } from "./types";
import { formatTimestamp } from "./types";

interface ActionFilterPreset {
  playerId: string;
  actionType: string;
}

interface ActionsTabProps {
  results: ResultsOut;
  flatEvents: FlatEvent[];
  currentTime: number;
  onSeek: (timeS: number) => void;
  onPlayAll: (timestamps: number[]) => void;
  presetFilter?: ActionFilterPreset | null;
}

export function ActionsTab({ results, flatEvents, currentTime, onSeek, onPlayAll, presetFilter }: ActionsTabProps) {
  const actionTypes = useMemo(() => Array.from(new Set(flatEvents.map((e) => e.action_type))).sort(), [flatEvents]);
  const rallyIndices = useMemo(
    () =>
      Array.from(new Set(flatEvents.map((e) => e.rally_index).filter((r): r is number => r !== null))).sort(
        (a, b) => a - b,
      ),
    [flatEvents],
  );

  // ActionsTab remounts fresh each time its tab is switched to (ResultsView
  // conditionally renders it), so seeding these from a preset in the
  // initializer - rather than syncing via an effect - naturally resets to
  // the unfiltered default whenever the tab is entered any other way.
  const [playerFilter, setPlayerFilter] = useState<string>(presetFilter?.playerId ?? "all");
  const [activeTypes, setActiveTypes] = useState<Set<string>>(() =>
    presetFilter ? new Set([presetFilter.actionType]) : new Set(actionTypes),
  );
  const [rallyFilter, setRallyFilter] = useState<number | "all">("all");

  function toggleType(type: string) {
    setActiveTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  }

  const filtered = flatEvents
    .filter((e) => playerFilter === "all" || e.playerId === playerFilter)
    .filter((e) => activeTypes.has(e.action_type))
    .filter((e) => rallyFilter === "all" || e.rally_index === rallyFilter);

  // The most recent visible action at or before the video's current
  // position - stays highlighted until the next one comes up, so it reads
  // as "what just happened" during playback.
  let currentEvent: FlatEvent | null = null;
  for (const event of filtered) {
    if (event.timestamp_s <= currentTime && (!currentEvent || event.timestamp_s > currentEvent.timestamp_s)) {
      currentEvent = event;
    }
  }

  const currentRowRef = useRef<HTMLTableRowElement>(null);

  useEffect(() => {
    if (currentEvent) currentRowRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [currentEvent]);

  return (
    <Box>
      <Card variant="outlined" sx={{ p: 2, mb: 2 }}>
        <Stack direction="row" spacing={1.5} sx={{ mb: 2 }}>
          <FormControl size="small" fullWidth>
            <InputLabel id="player-filter-label">Player</InputLabel>
            <Select
              labelId="player-filter-label"
              label="Player"
              value={playerFilter}
              onChange={(event: SelectChangeEvent) => setPlayerFilter(event.target.value)}
            >
              <MenuItem value="all">All players</MenuItem>
              {Object.entries(results.players).map(([id, stat]) => (
                <MenuItem key={id} value={id}>
                  {stat.name ?? `Player ${id}`}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" fullWidth>
            <InputLabel id="rally-filter-label">Rally</InputLabel>
            <Select
              labelId="rally-filter-label"
              label="Rally"
              value={rallyFilter}
              onChange={(event: SelectChangeEvent<number | "all">) =>
                setRallyFilter(event.target.value === "all" ? "all" : Number(event.target.value))
              }
            >
              <MenuItem value="all">All rallies</MenuItem>
              {rallyIndices.map((r) => (
                <MenuItem key={r} value={r}>
                  Rally {r + 1}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ textTransform: "uppercase", letterSpacing: 0.5, display: "block", mb: 0.25 }}
        >
          Action type
        </Typography>
        <Stack direction="row" sx={{ flexWrap: "wrap" }}>
          {actionTypes.map((type) => (
            <FormControlLabel
              key={type}
              control={<Checkbox size="small" checked={activeTypes.has(type)} onChange={() => toggleType(type)} />}
              label={type}
              sx={{ mr: 1.5 }}
            />
          ))}
        </Stack>
      </Card>

      <Box sx={{ mb: 1.5, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Typography variant="body2" color="text.secondary">
          {filtered.length} action{filtered.length === 1 ? "" : "s"}
        </Typography>
        {filtered.length > 0 && (
          <Button
            size="small"
            variant="contained"
            startIcon={<PlaylistPlayIcon />}
            onClick={() => onPlayAll(filtered.map((e) => e.timestamp_s))}
          >
            Play all
          </Button>
        )}
      </Box>

      <TableContainer component={Card} variant="outlined">
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>Time</TableCell>
              <TableCell>Player</TableCell>
              <TableCell>Action</TableCell>
              <TableCell>Rally</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filtered.map((event) => {
              const isCurrent = event === currentEvent;
              return (
                <TableRow
                  key={`${event.playerId}-${event.frame_idx}`}
                  ref={isCurrent ? currentRowRef : undefined}
                  hover
                  selected={isCurrent}
                  sx={{ cursor: "pointer" }}
                  onClick={() => onSeek(event.timestamp_s)}
                >
                  <TableCell>{formatTimestamp(event.timestamp_s)}</TableCell>
                  <TableCell>{event.playerName}</TableCell>
                  <TableCell>{event.action_type}</TableCell>
                  <TableCell>{event.rally_index !== null ? event.rally_index + 1 : "-"}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
