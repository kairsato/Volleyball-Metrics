import { memo } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import type { GameStatusSegment } from "../lib/types";

// A debug/QA annotation, same spirit as VideoPlayer's "Player IDs (debug)" -
// this shows GameStatusDetection's own raw three-way call (no-play, play,
// or service) for right now, rather than anything derived from it, so a
// wrong or missed boundary is visible directly on the video instead of only
// showing up as a downstream stat discrepancy. `segments` comes from its
// own endpoint (api.getGameStatus, warmup-rebased server-side - see
// results_router.get_game_status), separate from the `rallies` prop Score
// and the scrubber's chapter dividers use: a rally is service+play merged
// into one "the ball was live" window, which can't represent the
// service/play split within it the way segments can - see
// GameStatusSegment's own doc comment.
interface GameStatusOverlayProps {
  segments: GameStatusSegment[];
  currentTimeS: number;
}

const STATE_COLOR: Record<GameStatusSegment["state"], string> = {
  service: "#38bdf8",
  play: "#22c55e",
  "no-play": "rgba(255,255,255,0.45)",
};

const STATE_LABEL: Record<GameStatusSegment["state"], string> = {
  service: "Service",
  play: "Play",
  "no-play": "Not in play",
};

// Last segment whose start_time_s is at or before `t` - same small
// binary-search-over-a-sorted-field shape as BallTrajectoryOverlay's own
// findLastAtOrBefore, not the shared findIndexAtOrBefore in lib/timeSeries.ts
// (that one is generic over a `.t` field; segments carry start_time_s
// instead, and remapping the whole array every render just to reuse it
// would cost more than this one duplicated loop).
function findSegmentAtOrBefore(segments: GameStatusSegment[], t: number): GameStatusSegment | null {
  let lo = 0;
  let hi = segments.length - 1;
  let result: GameStatusSegment | null = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segments[mid].start_time_s <= t) {
      result = segments[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return result;
}

// Top-left - Score moved to top-CENTRE (it reads as a scoreboard; see its
// own doc comment) specifically to free this corner up, and Minimap sits
// top-right, so the three don't compete.
function GameStatusOverlayImpl({ segments, currentTimeS }: GameStatusOverlayProps) {
  if (segments.length === 0) return null;

  const current = findSegmentAtOrBefore(segments, currentTimeS);
  // Segments are built as one MAXIMAL run per state with no gaps between
  // them (see gameStatusDetection.py's _build_segments), so the only two
  // ways to fall outside all of them are before the very first one starts
  // (nothing classified yet) or past the last one's end (playback running
  // slightly beyond what game_status detection covered) - hide rather than
  // show a stale/guessed state either way.
  if (current === null || currentTimeS >= current.end_time_s) return null;

  return (
    <Box
      sx={{
        position: "absolute",
        left: 8,
        top: 8,
        zIndex: 1,
        pointerEvents: "none",
        borderRadius: 1,
        boxShadow: 3,
        bgcolor: "rgba(0,0,0,0.65)",
        px: 1.25,
        py: 0.5,
        display: "flex",
        alignItems: "center",
        gap: 0.75,
      }}
    >
      <Box
        sx={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          bgcolor: STATE_COLOR[current.state],
          flexShrink: 0,
        }}
      />
      <Typography variant="caption" sx={{ color: "#fff", fontWeight: 600, lineHeight: 1 }}>
        {STATE_LABEL[current.state]}
      </Typography>
    </Box>
  );
}

export const GameStatusOverlay = memo(GameStatusOverlayImpl);
