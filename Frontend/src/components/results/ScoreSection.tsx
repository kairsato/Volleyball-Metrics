import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import Alert from "@mui/material/Alert";
import Autocomplete from "@mui/material/Autocomplete";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CircularProgress from "@mui/material/CircularProgress";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import type { SelectChangeEvent } from "@mui/material/Select";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import ClearAllIcon from "@mui/icons-material/ClearAll";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { api } from "../../lib/api";
import type { Job, OcrRegion, Rally, ScoreConfig, ScoreMethod, ScoreResult, TeamEntry } from "../../lib/types";
import { ScoreRegionPicker } from "./ScoreRegionPicker";
import { ScoreTrackEditor, WIN_COLOR, LOSS_COLOR } from "./ScoreTrackEditor";
import { formatTimestamp } from "./types";

const METHOD_LABELS: Record<ScoreMethod, string> = {
  none: "None",
  manual: "Manual",
  automatic: "Heuristic",
  ocr: "Computer Vision",
};

// Shown under each option in the Method dropdown so it's clear what each
// one actually does before picking it, not just its name.
const METHOD_DESCRIPTIONS: Record<ScoreMethod, string> = {
  none: "Scoring isn't tracked for this video.",
  manual: "You assign every rally's winner and every game boundary yourself, from scratch.",
  automatic: "Guesses each rally's winner from which side the tracked ball was last seen on before it went out - no scoreboard needed, but never certain.",
  ocr: "Reads the scoreboard region you mark using a YOLO detector plus a digit-reading model - untested on real footage, so expect to correct it.",
};

type RangeType = "match" | "games" | "rallies";

const RANGE_LABELS: Record<RangeType, string> = {
  match: "Whole match",
  games: "Games",
  rallies: "Rallies",
};

// A minimum, not a fixed height - each card grows a bit taller when it
// also needs to show its "N undecided" warning line.
const GAME_ROW_MIN_HEIGHT_PX = 70;
const GAME_ROW_GAP_PX = 8;

// The Scoring Determination + Determined Score column sits to the left of
// the video+track column at this fixed width; the right-hand column takes
// whatever width is left over.
const LEFT_COLUMN_WIDTH_PX = 300;

// Seeking a game jumps most of the way through it rather than to the very
// start - matches ScoreTrackEditor's own rally/game seek behavior.
const SEEK_FRACTION = 0.9;

// Sits over an already-rendered, already-disabled region (the Scoring
// Determination card, the track editor) once scoring is confirmed - the
// individual controls underneath are still marked disabled (so they read
// as visually locked), but a disabled element never fires onClick at all,
// which would make "locked" look simply broken rather than intentional.
// This transparent layer is what actually catches the click and opens the
// same "Redo scoring?" dialog the Redo Scoring button itself opens,
// wherever on the locked area you click.
function LockOverlay({ active, onClick }: { active: boolean; onClick: () => void }) {
  if (!active) return null;
  return (
    <Tooltip title="Redo Scoring to make changes">
      <Box onClick={onClick} sx={{ position: "absolute", inset: 0, zIndex: 2, cursor: "pointer" }} />
    </Tooltip>
  );
}

// The estimated score so far, one boxed card per identified game titled
// "Match N" with its time span and a "Team 1 (2-1) Team 2" line underneath
// - whichever team currently leads that game is colored green, the other
// red (tied stays neutral) - recomputed straight from scoreResult on every
// render, so it updates the moment a winner is assigned or corrected
// without any extra fetch. Clicking a game jumps the video into it, the
// same as clicking that game's row in the track editor. The card itself
// fills whatever vertical space is left in the left-hand column (flex: 1
// below), with only the list of match boxes scrolling internally.
//
// "Confirm Scoring" is disabled (with an explanation) only while a rally
// is marked "uncertain" - the automatic/CV methods' own admission that
// they couldn't tell who won, which needs a human's eyes before it should
// be trusted. A rally that's simply undecided (winner left null, e.g. an
// untouched Manual-method rally) does NOT block confirming - that's a
// deliberate, ordinary state, not a red flag. Each match box that has
// uncertain rallies shows its own "N uncertain" warning, not just a
// summary at the button.
//
// Confirming doesn't touch any score data server-side - it's purely a
// local "confirmed" flag (owned by ScoreSection) that swaps this button
// for "Redo Scoring", and locks every other scoring control (see
// LockOverlay) until it's undone. Redo Scoring, once confirmed, asks for
// an explicit confirmation before flipping the flag back - unlike the
// other Redo* actions elsewhere in Setup, it doesn't reprocess or clear
// anything, it just unlocks scoring for further edits.
function GamesList({
  result,
  rallies,
  team1Name,
  team2Name,
  onSeek,
  confirmed,
  onConfirm,
  onRequestRedo,
  onResetAll,
}: {
  result: ScoreResult;
  rallies: Rally[];
  team1Name: string;
  team2Name: string;
  onSeek: (timeS: number) => void;
  confirmed: boolean;
  onConfirm: () => void;
  onRequestRedo: () => void;
  onResetAll: () => Promise<void>;
}) {
  const [resetDialogOpen, setResetDialogOpen] = useState(false);
  const [resetting, setResetting] = useState(false);

  if (result.games.length === 0) return null;

  const uncertainCount = result.rallies.filter((r) => r.confidence === "uncertain").length;
  const confirmDisabled = uncertainCount > 0;

  async function handleReset() {
    setResetting(true);
    try {
      await onResetAll();
      setResetDialogOpen(false);
    } finally {
      setResetting(false);
    }
  }

  return (
    <Card variant="outlined" sx={{ width: "100%", p: 2, flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: 1, flexShrink: 0 }}>
        <Typography variant="overline" color="text.secondary">
          Estimated Score
        </Typography>
        <Button
          size="small"
          color="warning"
          startIcon={<ClearAllIcon fontSize="small" />}
          onClick={() => (confirmed ? onRequestRedo() : setResetDialogOpen(true))}
        >
          Reset all scores
        </Button>
      </Stack>

      <Stack spacing={`${GAME_ROW_GAP_PX}px`} sx={{ flex: 1, minHeight: 0, overflowY: "auto", pr: 0.5 }}>
        {result.games.map((game) => {
          const gameRallies = result.rallies.filter((r) => r.game_index === game.game_index);
          const wins1 = gameRallies.filter((r) => r.winner === "x").length;
          const wins2 = gameRallies.filter((r) => r.winner === "y").length;
          const team1Color = wins1 === wins2 ? "text.primary" : wins1 > wins2 ? WIN_COLOR : LOSS_COLOR;
          const team2Color = wins1 === wins2 ? "text.primary" : wins2 > wins1 ? WIN_COLOR : LOSS_COLOR;
          const gameUncertain = gameRallies.filter((r) => r.confidence === "uncertain").length;

          const startRally = rallies[game.start_rally_index];
          const endRally = rallies[game.end_rally_index];

          function handleClick() {
            const span = rallies.slice(game.start_rally_index, game.end_rally_index + 1);
            if (span.length === 0) return;
            const startS = span[0].start_time_s;
            const endS = span[span.length - 1].end_time_s;
            onSeek(startS + (endS - startS) * SEEK_FRACTION);
          }

          return (
            <Card
              key={game.game_index}
              variant="outlined"
              onClick={handleClick}
              sx={{
                minHeight: GAME_ROW_MIN_HEIGHT_PX,
                flexShrink: 0,
                px: 1.5,
                py: 0.75,
                cursor: "pointer",
                "&:hover": { borderColor: "primary.main" },
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 700, borderBottom: 1, borderColor: "divider", pb: 0.25, mb: 0.5 }}>
                Match {game.game_index + 1}
              </Typography>

              {startRally && endRally && (
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.25 }}>
                  {formatTimestamp(startRally.start_time_s)} – {formatTimestamp(endRally.end_time_s)}
                </Typography>
              )}

              <Stack direction="row" spacing={0.5} sx={{ alignItems: "baseline", flexWrap: "wrap" }}>
                <Typography variant="body2" sx={{ color: team1Color, fontWeight: 600 }}>
                  {team1Name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  ({wins1}-{wins2})
                </Typography>
                <Typography variant="body2" sx={{ color: team2Color, fontWeight: 600 }}>
                  {team2Name}
                </Typography>
              </Stack>

              {gameUncertain > 0 && (
                <Stack direction="row" spacing={0.5} sx={{ alignItems: "center", mt: 0.25 }}>
                  <WarningAmberIcon sx={{ fontSize: 14, color: "warning.main" }} />
                  <Typography variant="caption" color="warning.main">
                    {gameUncertain} uncertain
                  </Typography>
                </Stack>
              )}
            </Card>
          );
        })}
      </Stack>

      {!confirmed && confirmDisabled && (
        <Typography variant="caption" color="warning.main" sx={{ mt: 1.5, flexShrink: 0 }}>
          {uncertainCount} rally{uncertainCount === 1 ? "" : "s"} marked uncertain - review and resolve them before
          scoring can be confirmed.
        </Typography>
      )}

      {confirmed ? (
        <Button
          variant="outlined"
          color="warning"
          startIcon={<RestartAltIcon />}
          fullWidth
          sx={{ mt: 1, flexShrink: 0 }}
          onClick={onRequestRedo}
        >
          Redo Scoring
        </Button>
      ) : (
        <Tooltip title={confirmDisabled ? "Resolve every uncertain rally before confirming" : ""}>
          <span style={{ display: "block" }}>
            <Button variant="contained" fullWidth sx={{ mt: 1, flexShrink: 0 }} disabled={confirmDisabled} onClick={onConfirm}>
              Confirm Scoring
            </Button>
          </span>
        </Tooltip>
      )}

      <Dialog open={resetDialogOpen} onClose={() => (resetting ? undefined : setResetDialogOpen(false))}>
        <DialogTitle>Reset all scores?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This clears every rally's winner back to undecided. Game boundaries are kept. This
            can't be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetDialogOpen(false)} disabled={resetting}>
            Cancel
          </Button>
          <Button color="warning" variant="contained" onClick={() => void handleReset()} disabled={resetting}>
            {resetting ? "Working..." : "Reset"}
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
}

// Draws/redraws are held as a local draft, not saved to score_config.json
// on every mouse-up the way the picker's own onRegionChange normally would
// be - only committed (via onConfirm) when the user hits OK, so closing
// the dialog with Cancel (or the backdrop) discards whatever was drawn.
function OcrRegionDialog({
  open,
  onClose,
  jobId,
  minTimeS,
  maxTimeS,
  initialRegion,
  reverseDirection,
  onReverseDirectionChange,
  onConfirm,
}: {
  open: boolean;
  onClose: () => void;
  jobId: string;
  minTimeS: number;
  maxTimeS: number;
  initialRegion: OcrRegion | null;
  reverseDirection: boolean;
  onReverseDirectionChange: (value: boolean) => void;
  onConfirm: (region: OcrRegion) => void;
}) {
  const [draftRegion, setDraftRegion] = useState<OcrRegion | null>(initialRegion);

  useEffect(() => {
    if (open) setDraftRegion(initialRegion);
    // Only reset the draft when the dialog is (re)opened, not every time
    // initialRegion's identity happens to change while it's already open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Scoreboard Identification Region</DialogTitle>
      <DialogContent>
        <ScoreRegionPicker
          jobId={jobId}
          minTimeS={minTimeS}
          maxTimeS={maxTimeS}
          region={draftRegion}
          onRegionChange={setDraftRegion}
        />

        <FormControlLabel
          sx={{ ml: 0, mt: 2 }}
          control={
            <Switch
              size="small"
              checked={reverseDirection}
              onChange={(e) => onReverseDirectionChange(e.target.checked)}
            />
          }
          label={<Typography variant="body2">Reverse score direction</Typography>}
        />
        <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
          Flips which side the left-read number counts for, in case the camera angle has the two
          sides mirrored from what's assumed.
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" disabled={!draftRegion} onClick={() => draftRegion && onConfirm(draftRegion)}>
          OK
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// Replaces what used to be two separate "From"/"To" number fields with
// MUI's own range slider (a Slider given an array value - see
// https://mui.com/material-ui/react-slider/#range-slider): dragging either
// thumb picks both ends of the range in one control, and disableSwap keeps
// the start from ever crossing past the end (or vice versa) the way the
// old pair of independently-typed numbers easily could. The slider's own
// min/max is the actual count of games or rallies that exist right now
// (upperBound, e.g. "2 games so far") rather than an arbitrary cap, so
// there's no way to select past what's actually there to analyze.
function RangePicker({
  rangeType,
  rangeStart,
  rangeEnd,
  upperBound,
  disabled,
  onChange,
}: {
  rangeType: Exclude<RangeType, "match">;
  rangeStart: number;
  rangeEnd: number;
  upperBound: number;
  disabled?: boolean;
  onChange: (start: number, end: number) => void;
}) {
  const unit = rangeType === "games" ? "game" : "rally";
  const unitPlural = rangeType === "games" ? "games" : "rallies";
  const sliderMax = Math.max(upperBound, 1);

  return (
    <Box sx={{ px: 1 }}>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
        {rangeType === "games" ? "Game range" : "Rally range"}:{" "}
        {rangeStart === rangeEnd ? `${unit} ${rangeStart}` : `${unitPlural} ${rangeStart}–${rangeEnd}`} ({upperBound}{" "}
        {upperBound === 1 ? unit : unitPlural} so far)
      </Typography>
      {sliderMax > 1 ? (
        <Slider
          value={[rangeStart, rangeEnd]}
          onChange={(_, value) => {
            const [start, end] = value as number[];
            onChange(start, end);
          }}
          min={1}
          max={sliderMax}
          step={1}
          marks={sliderMax <= 20}
          valueLabelDisplay="auto"
          disableSwap
          disabled={disabled}
        />
      ) : (
        <Typography variant="caption" color="text.secondary">
          No {unitPlural} identified yet - run over the whole match first, then narrow down.
        </Typography>
      )}
    </Box>
  );
}

interface ScoreSectionProps {
  job: Job;
  rallies: Rally[];
  currentTime: number;
  onSeek: (timeS: number) => void;
  videoElement: ReactNode;
}

// Who won each rally, and how rallies group into games/sets - a named
// roster team (see team_roster.py / TeamsPage) is identified for one or
// both sides once per video; which physical side each team is on is then
// worked out automatically per game rather than assumed fixed, since teams
// swap sides between sets. See Backend/API/score.py's module docstring for
// the full design.
export function ScoreSection({ job, rallies, currentTime, onSeek, videoElement }: ScoreSectionProps) {
  const [teams, setTeams] = useState<TeamEntry[]>([]);
  const [scoreConfig, setScoreConfig] = useState<ScoreConfig | null>(null);
  const [scoreResult, setScoreResult] = useState<ScoreResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rangeType, setRangeType] = useState<RangeType>("match");
  const [rangeStart, setRangeStart] = useState(1);
  const [rangeEnd, setRangeEnd] = useState(1);
  const [ocrDialogOpen, setOcrDialogOpen] = useState(false);
  // Lifted up from GamesList (rather than each locked control owning its
  // own copy) so every locked surface - the Scoring Determination card,
  // the track editor, and the "Redo Scoring" button itself - all open the
  // exact same confirmation dialog instead of three separate ones.
  const [redoDialogOpen, setRedoDialogOpen] = useState(false);
  const [redoing, setRedoing] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.getTeams(), api.getScore(job.id)])
      .then(([teamsRes, scoreRes]) => {
        if (cancelled) return;
        setTeams(teamsRes.teams);
        setScoreConfig(scoreRes.config);
        setScoreResult(scoreRes.result);
        // A compute kicked off before this mount (a previous visit to this
        // tab, or another browser tab) can still be running - without this,
        // the Analyze button just reads as permanently disabled until
        // something else happens to call startPolling again, since this
        // mount never itself triggered the compute call that would
        // normally start polling.
        if (scoreRes.config.compute_status === "computing") startPolling();
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : String(err)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  useEffect(() => {
    return () => {
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, []);

  // Whenever the range picker switches into "games" or "rallies", default
  // its two thumbs to 0% (the very first one) and 50% of however many
  // exist right now, rather than leaving both stuck at the same starting
  // value (1) - two coincident thumbs plus disableSwap (see RangePicker)
  // meant neither could actually be dragged, since disableSwap treats
  // "already at the other thumb's value" as a hard wall in both
  // directions. Only re-runs when rangeType itself changes, not on every
  // gameCount/rallies update, matching how the rest of this file (e.g.
  // OcrRegionDialog) only resets a draft on its own explicit trigger.
  useEffect(() => {
    if (rangeType === "match") return;
    const upperBound = rangeType === "games" ? (scoreResult?.games.length ?? 0) : rallies.length;
    const sliderMax = Math.max(upperBound, 1);
    setRangeStart(1);
    setRangeEnd(Math.max(1, Math.round((1 + sliderMax) / 2)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rangeType]);

  function startPolling() {
    if (pollTimer.current) clearInterval(pollTimer.current);
    pollTimer.current = setInterval(() => {
      api
        .getScore(job.id)
        .then((res) => {
          setScoreConfig(res.config);
          setScoreResult(res.result);
          if (res.config.compute_status !== "computing" && pollTimer.current) {
            clearInterval(pollTimer.current);
            pollTimer.current = null;
          }
        })
        .catch(() => undefined);
    }, 2000);
  }

  async function saveConfig(update: Partial<ScoreConfig>) {
    if (!scoreConfig) return;
    const merged = { ...scoreConfig, ...update };
    setScoreConfig(merged);
    try {
      const saved = await api.saveScoreConfig(
        job.id,
        merged.method,
        merged.team_x_id,
        merged.team_y_id,
        merged.ocr_region,
        merged.cv_reverse_direction,
      );
      setScoreConfig(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleAnalyze() {
    setError(null);
    try {
      const cfg = await api.computeScore(
        job.id,
        rangeType,
        rangeType === "match" ? undefined : rangeStart - 1,
        rangeType === "match" ? undefined : rangeEnd - 1,
      );
      setScoreConfig(cfg);
      startPolling();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleToggleBoundary(rallyIndex: number, split: boolean) {
    try {
      const result = await api.setGameBoundary(job.id, rallyIndex, split);
      setScoreResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSetWinner(rallyIndex: number, winner: "x" | "y" | null) {
    try {
      const result = await api.setRallyWinner(job.id, rallyIndex, winner);
      setScoreResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleResetScores() {
    try {
      const result = await api.resetScores(job.id);
      setScoreResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function persistConfirmed(value: boolean) {
    if (!scoreConfig) return;
    const merged = { ...scoreConfig, confirmed: value };
    setScoreConfig(merged);
    try {
      const saved = await api.setScoreConfirmed(job.id, value);
      setScoreConfig(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleConfirmScoring() {
    void persistConfirmed(true);
  }

  async function handleConfirmRedo() {
    setRedoing(true);
    try {
      await persistConfirmed(false);
      setRedoDialogOpen(false);
    } finally {
      setRedoing(false);
    }
  }

  if (loading || !scoreConfig) return null;

  const teamX = teams.find((t) => t.id === scoreConfig.team_x_id) ?? null;
  const teamY = teams.find((t) => t.id === scoreConfig.team_y_id) ?? null;
  const team1Name = teamX?.name ?? "Team 1";
  const team2Name = teamY?.name ?? "Team 2";
  const computing = scoreConfig.compute_status === "computing";
  const maxTimeS = rallies.length > 0 ? rallies[rallies.length - 1].end_time_s : 0;
  const gameCount = scoreResult?.games.length ?? 0;
  // A completed run that couldn't determine a single rally's winner - a
  // near-silent failure otherwise (the track editor would just show every
  // rally as gray/undecided, easy to mistake for "hasn't been analyzed
  // yet" rather than "was analyzed and found nothing"). Recomputed on
  // every render, not a one-off flag, so it clears itself the moment any
  // winner exists (an earlier successful run, or a manual correction).
  const foundNothing =
    !computing &&
    scoreConfig.compute_status !== "idle" &&
    scoreResult !== null &&
    scoreResult.rallies.length > 0 &&
    scoreResult.rallies.every((r) => r.winner === null);

  // The OCR region picker's scrub range follows whatever range is
  // currently selected in the Tools panel (Match/Games/Rallies), rather
  // than always spanning the whole video - if you're about to run OCR
  // over just "Rallies 5-8", the scoreboard you mark should be found by
  // scrubbing within that window, not the entire match.
  function rangeTimeBounds(): { minTimeS: number; maxTimeS: number } {
    const fallback = { minTimeS: 0, maxTimeS };
    if (rangeType === "match") return fallback;

    if (rangeType === "rallies") {
      const startRally = rallies[rangeStart - 1];
      const endRally = rallies[rangeEnd - 1];
      if (!startRally || !endRally) return fallback;
      return { minTimeS: startRally.start_time_s, maxTimeS: endRally.end_time_s };
    }

    const games = scoreResult?.games ?? [];
    const startGame = games.find((g) => g.game_index === rangeStart - 1);
    const endGame = games.find((g) => g.game_index === rangeEnd - 1);
    const startRally = startGame ? rallies[startGame.start_rally_index] : undefined;
    const endRally = endGame ? rallies[endGame.end_rally_index] : undefined;
    if (!startRally || !endRally) return fallback;
    return { minTimeS: startRally.start_time_s, maxTimeS: endRally.end_time_s };
  }

  const ocrRange = rangeTimeBounds();

  return (
    <Stack spacing={2} sx={{ height: "100%" }}>
      {error && <Alert severity="error">{error}</Alert>}

      {/* Two side-by-side elements filling whatever's left of the page
          (flex: 1, minHeight: 0 below, against the root Stack's height:
          100% above): the Scoring Determination + Estimated Score column
          on the left, and the video + track stacked together as one unit
          on the right. Within that right-hand column, the track keeps its
          own fixed height (TRACK_HEIGHT_PX, set in ScoreTrackEditor) and
          the video is what flexes to fill whatever vertical space is left
          over above it. */}
      <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} sx={{ flex: 1, minHeight: 0 }}>
        {/* Stretches (via the row's default alignItems) to match the
            right-hand column's height, same as it does. The Scoring
            Determination card keeps its own natural height (flexShrink: 0)
            and the Estimated Score card below it (see GamesList) absorbs
            whatever vertical space is left over. */}
        <Stack spacing={2} sx={{ width: LEFT_COLUMN_WIDTH_PX, flexShrink: 0, maxWidth: "100%", minHeight: 0 }}>
        <Card variant="outlined" sx={{ p: 2, flexShrink: 0, position: "relative" }}>
          <LockOverlay active={scoreConfig.confirmed} onClick={() => setRedoDialogOpen(true)} />
          <Typography variant="overline" color="text.secondary" sx={{ display: "block", mb: 1 }}>
            Scoring Determination
          </Typography>

          <Stack spacing={2}>
            <FormControl size="small" fullWidth disabled={computing || scoreConfig.confirmed}>
              <InputLabel id="score-method-label">Method</InputLabel>
              <Select
                labelId="score-method-label"
                label="Method"
                value={scoreConfig.method}
                renderValue={(value) => METHOD_LABELS[value as ScoreMethod]}
                onChange={(e: SelectChangeEvent) => {
                  const nextMethod = e.target.value as ScoreMethod;
                  void saveConfig({ method: nextMethod });
                  if (nextMethod === "ocr") setOcrDialogOpen(true);
                }}
              >
                {(Object.keys(METHOD_LABELS) as ScoreMethod[]).map((method) => (
                  <MenuItem key={method} value={method} sx={{ flexDirection: "column", alignItems: "flex-start", py: 1 }}>
                    <Typography variant="body2">{METHOD_LABELS[method]}</Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: "normal" }}>
                      {METHOD_DESCRIPTIONS[method]}
                    </Typography>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {scoreConfig.method !== "none" && (
              <>
                <Autocomplete
                  size="small"
                  disabled={computing || scoreConfig.confirmed}
                  options={teams}
                  value={teamX}
                  getOptionLabel={(t) => t.name}
                  isOptionEqualToValue={(a, b) => a.id === b.id}
                  onChange={(_, value) => void saveConfig({ team_x_id: value?.id ?? null })}
                  renderInput={(params) => <TextField {...params} label="Team 1" />}
                />
                <Autocomplete
                  size="small"
                  disabled={computing || scoreConfig.confirmed}
                  options={teams}
                  value={teamY}
                  getOptionLabel={(t) => t.name}
                  isOptionEqualToValue={(a, b) => a.id === b.id}
                  onChange={(_, value) => void saveConfig({ team_y_id: value?.id ?? null })}
                  renderInput={(params) => <TextField {...params} label="Team 2 (optional)" />}
                />
              </>
            )}

            {scoreConfig.method === "ocr" && (
              <Stack spacing={0.5}>
                <Button
                  variant="outlined"
                  fullWidth
                  disabled={computing || scoreConfig.confirmed}
                  sx={{ py: 1.25 }}
                  onClick={() => setOcrDialogOpen(true)}
                >
                  {scoreConfig.ocr_region ? "Change scoreboard region" : "Set scoreboard region"}
                </Button>
                {!scoreConfig.ocr_region && (
                  <Typography variant="caption" color="text.secondary">
                    Required before Computer Vision can run.
                  </Typography>
                )}
              </Stack>
            )}

            {(scoreConfig.method === "automatic" || scoreConfig.method === "ocr") && (
              <>
                <FormControl size="small" fullWidth disabled={computing || scoreConfig.confirmed}>
                  <InputLabel id="score-range-label">Range</InputLabel>
                  <Select
                    labelId="score-range-label"
                    label="Range"
                    value={rangeType}
                    onChange={(e: SelectChangeEvent) => setRangeType(e.target.value as RangeType)}
                  >
                    {(Object.keys(RANGE_LABELS) as RangeType[]).map((type) => (
                      <MenuItem key={type} value={type}>
                        {RANGE_LABELS[type]}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                {rangeType !== "match" && (
                  <RangePicker
                    rangeType={rangeType}
                    rangeStart={rangeStart}
                    rangeEnd={rangeEnd}
                    upperBound={rangeType === "games" ? gameCount : rallies.length}
                    disabled={computing || scoreConfig.confirmed}
                    onChange={(start, end) => {
                      setRangeStart(start);
                      setRangeEnd(end);
                    }}
                  />
                )}

                <Button
                  variant="contained"
                  startIcon={computing ? <CircularProgress size={16} color="inherit" /> : <AnalyticsIcon />}
                  disabled={
                    computing || scoreConfig.confirmed || (scoreConfig.method === "ocr" && !scoreConfig.ocr_region)
                  }
                  onClick={() => void handleAnalyze()}
                >
                  {computing ? "Analyzing..." : "Analyze"}
                </Button>
                {scoreConfig.compute_status === "error" && (
                  <Typography variant="body2" color="error">
                    {scoreConfig.compute_error}
                  </Typography>
                )}
                {foundNothing && (
                  <Alert severity="warning">
                    No winners could be determined for any rally.{" "}
                    {scoreConfig.method === "ocr"
                      ? "Double-check the marked scoreboard region - it may be misaligned, or the scoreboard may not be readable there."
                      : "Ball tracking may not have produced usable data for this video."}
                  </Alert>
                )}
              </>
            )}
          </Stack>
        </Card>

        {scoreResult && (
          <GamesList
            result={scoreResult}
            rallies={rallies}
            team1Name={team1Name}
            team2Name={team2Name}
            onSeek={onSeek}
            confirmed={scoreConfig.confirmed}
            onConfirm={handleConfirmScoring}
            onRequestRedo={() => setRedoDialogOpen(true)}
            onResetAll={handleResetScores}
          />
        )}
        </Stack>

        {/* The video + track stacked as one right-hand column, sharing the
            same left edge (no separate spacer needed - the track just sits
            directly under the video in the same flex column now). The
            video is flex: 1 so it absorbs whatever vertical space the
            fixed-height track (see TRACK_HEIGHT_PX) doesn't use. */}
        <Stack sx={{ flex: 1, minWidth: 0, height: "100%" }}>
          <Box sx={{ flex: 1, minHeight: 0, overflow: "hidden" }}>{videoElement}</Box>

          {scoreConfig.method === "none" ? (
            <Typography color="text.secondary" sx={{ pt: 1 }}>
              Pick a method above to start scoring this video.
            </Typography>
          ) : scoreResult ? (
            <Box sx={{ position: "relative", flexShrink: 0 }}>
              <LockOverlay active={scoreConfig.confirmed} onClick={() => setRedoDialogOpen(true)} />
              <ScoreTrackEditor
                rallies={rallies}
                result={scoreResult}
                team1Name={team1Name}
                team2Name={team2Name}
                currentTime={currentTime}
                onSeek={onSeek}
                onToggleGameBoundary={handleToggleBoundary}
                onSetWinner={handleSetWinner}
              />
            </Box>
          ) : (
            <Typography color="text.secondary" sx={{ pt: 1 }}>
              Run detection (or switch to Manual) to start scoring.
            </Typography>
          )}
        </Stack>
      </Stack>

      <OcrRegionDialog
        open={ocrDialogOpen}
        onClose={() => setOcrDialogOpen(false)}
        jobId={job.id}
        minTimeS={ocrRange.minTimeS}
        maxTimeS={ocrRange.maxTimeS}
        initialRegion={scoreConfig.ocr_region}
        reverseDirection={scoreConfig.cv_reverse_direction}
        onReverseDirectionChange={(value) => void saveConfig({ cv_reverse_direction: value })}
        onConfirm={(region) => {
          void saveConfig({ ocr_region: region });
          setOcrDialogOpen(false);
        }}
      />

      {/* Shared by the "Redo Scoring" button (GamesList) and every
          LockOverlay - whichever locked control the user clicks, it's this
          same dialog that opens, not a separate one per trigger. */}
      <Dialog open={redoDialogOpen} onClose={() => (redoing ? undefined : setRedoDialogOpen(false))}>
        <DialogTitle>Redo scoring?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This won't change any rally winners or game boundaries - it just unlocks the score for
            editing again until you confirm it once more.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRedoDialogOpen(false)} disabled={redoing}>
            Cancel
          </Button>
          <Button color="warning" variant="contained" onClick={() => void handleConfirmRedo()} disabled={redoing}>
            {redoing ? "Working..." : "Confirm"}
          </Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
