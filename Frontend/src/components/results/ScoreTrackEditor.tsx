import { Fragment, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Stack from "@mui/material/Stack";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import type { Rally, ScoreResult } from "../../lib/types";
import { formatTimestamp } from "./types";

// Exported so GamesList (ScoreSection.tsx) can color its own team
// names/scores with the same win/loss palette used here.
export const WIN_COLOR = "#22c55e";
export const LOSS_COLOR = "#ef4444";
const UNDECIDED_COLOR = "rgba(148, 163, 184, 0.35)";
const UNCERTAIN_STRIPE =
  "repeating-linear-gradient(45deg, rgba(234,179,8,0.6) 0 4px, rgba(234,179,8,0.25) 4px 8px)";

// Games alternate between these two tints just to make adjacent groups
// visually distinguishable - the color itself carries no other meaning.
const GAME_TINTS = ["rgba(148, 163, 184, 0.18)", "rgba(148, 163, 184, 0.32)"];

const PX_PER_SECOND = 14;
const MIN_COLUMN_PX = 30;
const BOUNDARY_GAP_PX = 14;
const ROW_HEIGHT_PX = 32;
const LABEL_WIDTH_PX = 76;
const LEGEND_WIDTH_PX = 120;
// Fixed, not fill-the-page - the board and legend cards both stretch to
// match this exactly (see the grid's default stretch alignment below).
const TRACK_HEIGHT_PX = 200;

// Seeking a rally/game jumps most of the way through it rather than to the
// very start - the decisive moment (the point actually being won or lost)
// is near the end, not the serve.
const SEEK_FRACTION = 0.9;

interface ScoreTrackEditorProps {
  rallies: Rally[];
  result: ScoreResult;
  team1Name: string;
  team2Name: string;
  currentTime: number;
  onSeek: (timeS: number) => void;
  onToggleGameBoundary: (rallyIndex: number, split: boolean) => void;
  onSetWinner: (rallyIndex: number, winner: "x" | "y" | null) => void;
}

function columnWidth(rally: Rally): number {
  return Math.max(MIN_COLUMN_PX, rally.duration_s * PX_PER_SECOND);
}

interface ColumnLayout {
  rally: Rally;
  x: number;
  width: number;
}

// Rally columns sit edge to edge (plus a small fixed gap for the boundary
// control between them) regardless of how much real dead time separates
// them in the actual video - so mapping a video timestamp onto a track
// x-position (and back) has to walk rally-by-rally rather than using a
// single linear scale.
function buildColumnLayout(rallies: Rally[]): { columns: ColumnLayout[]; totalWidth: number } {
  const columns: ColumnLayout[] = [];
  let cursor = 0;
  for (let i = 0; i < rallies.length; i++) {
    const width = columnWidth(rallies[i]);
    columns.push({ rally: rallies[i], x: cursor, width });
    cursor += width + (i < rallies.length - 1 ? BOUNDARY_GAP_PX : 0);
  }
  return { columns, totalWidth: cursor };
}

// The real dead time between two rallies (players resetting, a whistle,
// walking back to serve...) is very often *longer* than the rallies
// themselves, but only gets a small fixed-width gap column on screen (see
// BOUNDARY_GAP_PX) - it's still real elapsed video time though, so the
// playhead has to keep moving through it (compressed into that small
// width) rather than sitting frozen at the next rally's start until
// playback finally reaches it. Both functions below walk rally-by-rally
// *and* gap-by-gap for exactly that reason.
function timeToX(columns: ColumnLayout[], totalWidth: number, timeS: number): number {
  if (columns.length === 0) return 0;
  if (timeS <= columns[0].rally.start_time_s) return columns[0].x;

  for (let i = 0; i < columns.length; i++) {
    const col = columns[i];
    if (timeS <= col.rally.end_time_s) {
      const frac = col.rally.duration_s > 0 ? (timeS - col.rally.start_time_s) / col.rally.duration_s : 0;
      return col.x + frac * col.width;
    }

    const next = columns[i + 1];
    if (!next) break;
    if (timeS <= next.rally.start_time_s) {
      const gapStart = col.x + col.width;
      const deadDuration = next.rally.start_time_s - col.rally.end_time_s;
      const frac = deadDuration > 0 ? (timeS - col.rally.end_time_s) / deadDuration : 0;
      return gapStart + frac * (next.x - gapStart);
    }
  }
  return totalWidth;
}

// Inverse of timeToX, for dragging the playhead.
function xToTime(columns: ColumnLayout[], x: number): number {
  if (columns.length === 0) return 0;
  if (x <= columns[0].x) return columns[0].rally.start_time_s;

  for (let i = 0; i < columns.length; i++) {
    const col = columns[i];
    const colEnd = col.x + col.width;
    if (x <= colEnd) {
      const frac = col.width > 0 ? Math.max(0, Math.min(1, (x - col.x) / col.width)) : 0;
      return col.rally.start_time_s + frac * col.rally.duration_s;
    }

    const next = columns[i + 1];
    if (!next) return col.rally.end_time_s;
    if (x <= next.x) {
      const gapWidth = next.x - colEnd;
      const frac = gapWidth > 0 ? Math.max(0, Math.min(1, (x - colEnd) / gapWidth)) : 0;
      const deadDuration = next.rally.start_time_s - col.rally.end_time_s;
      return col.rally.end_time_s + frac * deadDuration;
    }
  }
  return columns[columns.length - 1].rally.end_time_s;
}

function LegendSwatch({ swatch, label }: { swatch: ReactNode; label: string }) {
  return (
    <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
      {swatch}
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Stack>
  );
}

// Four tracks, top to bottom: which game each rally belongs to (the only
// row where boundaries can be moved, and where its own click seeks 90% of
// the way through that whole game), the rally itself (fixed - click seeks
// 90% of the way through it), Team 1's outcome, and Team 2's - the last
// two are just mirror images of each other (there are only two teams, so
// one team's win is necessarily the other's loss) but both get their own
// row since that's the natural way to read "how did *this* team do" for
// either side without mentally flipping colors.
//
// Layout is a CSS grid: a "board" (the row labels + the scrollable track
// columns, boxed together as one card) plus a separate legend card in its
// own column. The legend only actually describes the Team 1/Team 2 rows
// (the win/loss colors), so its card is offset down and sized to match
// just those two rows' height rather than the full four-row board.
export function ScoreTrackEditor({
  rallies,
  result,
  team1Name,
  team2Name,
  currentTime,
  onSeek,
  onToggleGameBoundary,
  onSetWinner,
}: ScoreTrackEditorProps) {
  const [dragTime, setDragTime] = useState<number | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  const { columns, totalWidth } = buildColumnLayout(rallies);

  function seekFromClientX(clientX: number) {
    const el = contentRef.current;
    if (!el || totalWidth <= 0) return;
    const rect = el.getBoundingClientRect();
    // Rendered column widths can be larger than columnWidth() computes -
    // rally columns grow (flexGrow) to fill any leftover space in the
    // board, so the DOM's actual width and buildColumnLayout's "natural"
    // width can differ. Converting through a 0-1 fraction of the rendered
    // width, rather than using raw pixels, keeps xToTime's math correct
    // regardless of how much (if any) scaling actually happened.
    const fraction = (clientX - rect.left) / rect.width;
    const time = xToTime(columns, fraction * totalWidth);
    setDragTime(time);
    onSeek(time);
  }

  // The mousemove/mouseup listeners are attached synchronously right here,
  // not via a `dragging`-triggered useEffect - a useEffect only runs after
  // this handler returns, and a real mouseup (even from a quick click, not
  // an actual drag) can beat it there. When that race is lost, handleUp
  // never fires, dragTime never gets cleared, and the playhead - which
  // prefers dragTime over the live currentTime prop whenever it's set -
  // freezes at that one click's position forever. Attaching inline here
  // makes that race impossible: the listeners are live before this
  // synchronous handler even returns, so no subsequent mouseup can slip
  // through unheard.
  function handlePlayheadMouseDown(event: ReactMouseEvent) {
    event.preventDefault();
    seekFromClientX(event.clientX);

    function handleMove(moveEvent: MouseEvent) {
      seekFromClientX(moveEvent.clientX);
    }
    function handleUp() {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
      setDragTime(null);
    }

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
  }

  if (rallies.length === 0) {
    return <Typography color="text.secondary">No rallies to score yet.</Typography>;
  }

  const recordByRally = new Map(result.rallies.map((r) => [r.rally_index, r]));
  // A fraction (0-1) of the timeline, not a raw pixel offset - the same
  // reasoning as seekFromClientX above: rendered column widths can differ
  // from buildColumnLayout's "natural" widths once flexGrow has scaled
  // them to fill the board, so positioning the playhead by percentage of
  // the actual rendered width is what keeps it lined up with the video.
  const playheadFraction = totalWidth > 0 ? timeToX(columns, totalWidth, dragTime ?? currentTime) / totalWidth : 0;
  // eslint-disable-next-line no-console
  console.debug("[ScoreTrackEditor] render", { currentTime, dragTime, totalWidth, playheadFraction });

  function seekInto(startS: number, endS: number) {
    onSeek(startS + (endS - startS) * SEEK_FRACTION);
  }

  return (
    <Box
      sx={{
        height: TRACK_HEIGHT_PX,
        display: "grid",
        gridTemplateColumns: `minmax(0, 1fr) ${LEGEND_WIDTH_PX}px`,
        gridTemplateAreas: `"board legend"`,
        columnGap: 1.5,
      }}
    >
      <Card variant="outlined" sx={{ gridArea: "board", height: "100%", display: "flex", minWidth: 0, p: 1 }}>
        <Stack sx={{ width: LABEL_WIDTH_PX, height: "100%", flexShrink: 0 }}>
          {["Game", "Rally", team1Name, team2Name].map((label) => (
            <Box key={label} sx={{ flex: 1, minHeight: ROW_HEIGHT_PX, display: "flex", alignItems: "center" }}>
              <Typography variant="caption" color="text.secondary" noWrap title={label}>
                {label}
              </Typography>
            </Box>
          ))}
        </Stack>

        <Box sx={{ overflowX: "auto", flex: 1, height: "100%", minWidth: 0 }}>
        {/* No explicit width here - a block-level flex row naturally fills
            its parent's width on its own. Each rally column below grows
            proportionally to its own (clamped) width - not duration_s
            directly - to fill any leftover space: a uniform zoom of the
            whole timeline rather than padding added to one column, while
            flexShrink: 0 keeps every column at least as wide as its
            duration needs, so a rally list that needs more room than the
            board has scrolls instead of being squeezed. Growing by width
            rather than duration matters once MIN_COLUMN_PX clamps a very
            short rally - growing by its true (tiny) duration instead would
            scale it disproportionately little relative to its actual
            on-screen width, breaking the uniform-zoom assumption the
            playhead's position math (timeToX/xToTime) relies on. */}
        <Stack ref={contentRef} direction="row" sx={{ height: "100%", position: "relative" }}>
          {rallies.map((rally, i) => {
            const record = recordByRally.get(rally.rally_index);
            const isCurrent = currentTime >= rally.start_time_s && currentTime <= rally.end_time_s;
            const isLast = i === rallies.length - 1;
            const prevRecord = i > 0 ? recordByRally.get(rallies[i - 1].rally_index) : undefined;
            const nextRecord = !isLast ? recordByRally.get(rallies[i + 1].rally_index) : undefined;
            const hasBoundaryAfter = isLast || record?.game_index !== nextRecord?.game_index;
            const isFirstOfGame = i === 0 || record?.game_index !== prevRecord?.game_index;

            let team1Bg = UNDECIDED_COLOR;
            if (record?.winner === "x") team1Bg = WIN_COLOR;
            else if (record?.winner === "y") team1Bg = LOSS_COLOR;
            if (record?.confidence === "uncertain") team1Bg = UNCERTAIN_STRIPE;

            // Team 2's row is the exact opposite of Team 1's: a win for one
            // is a loss for the other, and "uncertain"/undecided mirror
            // straight across since neither team has a determined outcome.
            let team2Bg = UNDECIDED_COLOR;
            if (record?.winner === "x") team2Bg = LOSS_COLOR;
            else if (record?.winner === "y") team2Bg = WIN_COLOR;
            if (record?.confidence === "uncertain") team2Bg = UNCERTAIN_STRIPE;

            function cycleTeam1() {
              const current = record?.winner ?? null;
              onSetWinner(rally.rally_index, current === null ? "x" : current === "x" ? "y" : null);
            }

            function cycleTeam2() {
              const current = record?.winner ?? null;
              onSetWinner(rally.rally_index, current === null ? "y" : current === "y" ? "x" : null);
            }

            function seekGame() {
              const gameRallies = rallies.filter(
                (r) => recordByRally.get(r.rally_index)?.game_index === record?.game_index,
              );
              if (gameRallies.length === 0) return;
              seekInto(gameRallies[0].start_time_s, gameRallies[gameRallies.length - 1].end_time_s);
            }

            return (
              <Fragment key={rally.rally_index}>
                <Stack sx={{ flex: `${columnWidth(rally)} 0 ${columnWidth(rally)}px`, height: "100%" }}>
                  <Tooltip title={`Game ${(record?.game_index ?? 0) + 1}`}>
                    <Box
                      onClick={seekGame}
                      sx={{
                        flex: 1,
                        minHeight: ROW_HEIGHT_PX,
                        bgcolor: GAME_TINTS[(record?.game_index ?? 0) % 2],
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        "&:hover": { filter: "brightness(1.3)" },
                      }}
                    >
                      {isFirstOfGame && (
                        <Typography variant="caption" noWrap sx={{ fontSize: 10, fontWeight: 600 }}>
                          Game {(record?.game_index ?? 0) + 1}
                        </Typography>
                      )}
                    </Box>
                  </Tooltip>

                  <Tooltip title={formatTimestamp(rally.start_time_s)}>
                    <Box
                      onClick={() => seekInto(rally.start_time_s, rally.end_time_s)}
                      sx={{
                        flex: 1,
                        minHeight: ROW_HEIGHT_PX,
                        bgcolor: "action.hover",
                        border: 1,
                        borderColor: isCurrent ? "primary.main" : "divider",
                        borderRadius: 0.5,
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        "&:hover": { borderColor: "primary.main" },
                      }}
                    >
                      <Typography variant="caption" noWrap sx={{ fontSize: 10 }}>
                        Rally {rally.rally_index + 1}
                      </Typography>
                    </Box>
                  </Tooltip>

                  <Tooltip title={`Click to change ${team1Name}'s outcome for this rally`}>
                    <Box
                      onClick={cycleTeam1}
                      sx={{
                        flex: 1,
                        minHeight: ROW_HEIGHT_PX,
                        background: team1Bg,
                        borderRadius: 0.5,
                        cursor: "pointer",
                        opacity: 0.9,
                        "&:hover": { opacity: 1 },
                      }}
                    />
                  </Tooltip>

                  <Tooltip title={`Click to change ${team2Name}'s outcome for this rally`}>
                    <Box
                      onClick={cycleTeam2}
                      sx={{
                        flex: 1,
                        minHeight: ROW_HEIGHT_PX,
                        background: team2Bg,
                        borderRadius: 0.5,
                        cursor: "pointer",
                        opacity: 0.9,
                        "&:hover": { opacity: 1 },
                      }}
                    />
                  </Tooltip>
                </Stack>

                {!isLast && (
                  <Tooltip title={hasBoundaryAfter ? "Merge with next game" : "Move this game's end here"}>
                    <Stack sx={{ width: BOUNDARY_GAP_PX, flexShrink: 0, flexGrow: 0, height: "100%" }}>
                      <Box
                        onClick={() => onToggleGameBoundary(rally.rally_index, !hasBoundaryAfter)}
                        sx={{
                          flex: 1,
                          minHeight: ROW_HEIGHT_PX,
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          "&:hover .boundary-line": { bgcolor: "primary.main" },
                        }}
                      >
                        <Box
                          className="boundary-line"
                          sx={{
                            width: hasBoundaryAfter ? 3 : 1,
                            height: "100%",
                            bgcolor: hasBoundaryAfter ? "warning.main" : "divider",
                          }}
                        />
                      </Box>
                      <Box sx={{ flex: 1, minHeight: ROW_HEIGHT_PX }} />
                      <Box sx={{ flex: 1, minHeight: ROW_HEIGHT_PX }} />
                      <Box sx={{ flex: 1, minHeight: ROW_HEIGHT_PX }} />
                    </Stack>
                  </Tooltip>
                )}
              </Fragment>
            );
          })}

          {/* The video's current playback position, mapped onto this
              rally-by-rally timeline - draggable to scrub the video
              directly from here, same as dragging a normal video
              scrubber. Wider than its visible 2px line so it's easy to
              grab. */}
          <Box
            onMouseDown={handlePlayheadMouseDown}
            sx={{
              position: "absolute",
              top: 0,
              left: `calc(${playheadFraction * 100}% - 6px)`,
              width: 12,
              height: "100%",
              cursor: "ew-resize",
              display: "flex",
              justifyContent: "center",
              zIndex: 2,
            }}
          >
            <Box sx={{ width: 2, height: "100%", bgcolor: "primary.main" }} />
            <Box
              sx={{
                position: "absolute",
                top: -4,
                width: 12,
                height: 8,
                borderRadius: "2px",
                bgcolor: "primary.main",
              }}
            />
          </Box>
        </Stack>
        </Box>
      </Card>

      {/* No explicit height - CSS grid's default stretch alignment matches
          this to the board card's height (they share the same grid row),
          so the legend spans the whole track, not just part of it. */}
      <Card
        variant="outlined"
        sx={{
          gridArea: "legend",
          p: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-evenly",
        }}
      >
        <LegendSwatch
          swatch={<Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: WIN_COLOR }} />}
          label="Win"
        />
        <LegendSwatch
          swatch={<Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: LOSS_COLOR }} />}
          label="Loss"
        />
        <LegendSwatch
          swatch={<Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: UNDECIDED_COLOR }} />}
          label="Undecided"
        />
        <LegendSwatch
          swatch={<Box sx={{ width: 8, height: 8, borderRadius: "50%", background: UNCERTAIN_STRIPE }} />}
          label="Uncertain"
        />
      </Card>
    </Box>
  );
}
