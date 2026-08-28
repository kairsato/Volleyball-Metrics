import { useEffect, useRef } from "react";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PlaylistPlayIcon from "@mui/icons-material/PlaylistPlay";
import type { Rally } from "../../lib/types";
import { formatTimestamp, type FlatEvent } from "./types";

interface RalliesTabProps {
  rallies: Rally[];
  flatEvents: FlatEvent[];
  currentTime: number;
  onSeek: (timeS: number) => void;
  onPlayAll: (timestamps: number[]) => void;
}

interface RallyCardProps {
  rally: Rally;
  events: FlatEvent[];
  isCurrent: boolean;
  onSeek: (timeS: number) => void;
  onPlayAll: (timestamps: number[]) => void;
}

function RallyCard({ rally, events, isCurrent, onSeek, onPlayAll }: RallyCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  const players = Array.from(new Set(events.map((e) => e.playerName)));

  useEffect(() => {
    if (isCurrent) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [isCurrent]);

  return (
    <Card
      ref={ref}
      variant="outlined"
      sx={{
        p: 2,
        borderColor: isCurrent ? "primary.main" : "divider",
        borderWidth: isCurrent ? 2 : 1,
        bgcolor: isCurrent ? "action.hover" : "transparent",
        transition: "background-color 0.15s ease, border-color 0.15s ease",
      }}
    >
      <Stack
        direction="row"
        spacing={1}
        sx={{ alignItems: "center", justifyContent: "space-between", mb: 1, flexWrap: "wrap" }}
      >
        <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
          <Typography sx={{ fontWeight: 600 }}>
            Rally {rally.rally_index + 1}{" "}
            <Typography component="span" variant="body2" color="text.secondary">
              {formatTimestamp(rally.start_time_s)}-{formatTimestamp(rally.end_time_s)} · {rally.duration_s.toFixed(1)}s
            </Typography>
          </Typography>
          {isCurrent && <Chip size="small" color="primary" label="Playing" />}
        </Stack>
        <Stack direction="row" spacing={1}>
          <Button size="small" startIcon={<PlayArrowIcon />} onClick={() => onSeek(rally.start_time_s)}>
            Play
          </Button>
          {events.length > 0 && (
            <Button
              size="small"
              startIcon={<PlaylistPlayIcon />}
              onClick={() => onPlayAll(events.map((e) => e.timestamp_s))}
            >
              Play all touches
            </Button>
          )}
        </Stack>
      </Stack>

      {players.length > 0 ? (
        <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap" }}>
          {players.map((name) => (
            <Chip key={name} size="small" label={name} variant="outlined" />
          ))}
        </Stack>
      ) : (
        <Typography variant="body2" color="text.secondary">
          No recorded touches.
        </Typography>
      )}
    </Card>
  );
}

export function RalliesTab({ rallies, flatEvents, currentTime, onSeek, onPlayAll }: RalliesTabProps) {
  if (rallies.length === 0) {
    return <Typography color="text.secondary">No rallies were detected in this video.</Typography>;
  }

  return (
    <Stack spacing={1.5}>
      {rallies.map((rally) => (
        <RallyCard
          key={rally.rally_index}
          rally={rally}
          events={flatEvents.filter((e) => e.rally_index === rally.rally_index)}
          isCurrent={currentTime >= rally.start_time_s && currentTime <= rally.end_time_s}
          onSeek={onSeek}
          onPlayAll={onPlayAll}
        />
      ))}
    </Stack>
  );
}
