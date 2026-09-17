import { useEffect, useMemo, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import DeleteIcon from "@mui/icons-material/Delete";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import { useNavigate, useSearchParams } from "react-router-dom";
import { VideoPlayer } from "../components/VideoPlayer";
import { StatTilesSkeleton } from "../components/Skeletons";
import { api } from "../lib/api";
import type { BallTrajectory, CalibrationPointsOut, GameStatusOut, Job, PlayerTrajectory } from "../lib/types";

// Same fixed-viewport-height approach as ResultsView.tsx/WarmupPeriodPage.tsx
// - the page itself never scrolls; only the right-hand tab content does.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

const TAB_NAMES = ["rallies", "ball", "players", "court"] as const;
const TAB_QUERY_PARAM = "tab";
const JOB_QUERY_PARAM = "job";

function readTabFromUrl(): number {
  const name = new URLSearchParams(window.location.search).get(TAB_QUERY_PARAM);
  const index = TAB_NAMES.indexOf(name as (typeof TAB_NAMES)[number]);
  return index === -1 ? 0 : index;
}

function writeTabToUrl(index: number) {
  const url = new URL(window.location.href);
  url.searchParams.set(TAB_QUERY_PARAM, TAB_NAMES[index]);
  window.history.replaceState({}, "", url);
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// A read-only horizontal strip of colored blocks over [0, duration] - shared
// by the rally/ball panels for "make it easy to identify what was
// detected" - clicking anywhere seeks the shared video to that point.
function Timeline({
  duration,
  blocks,
  height = 28,
  onSeek,
}: {
  duration: number;
  blocks: { startS: number; endS: number; color: string; key: string | number }[];
  height?: number;
  onSeek?: (t: number) => void;
}) {
  if (duration <= 0) return null;
  return (
    <Box
      onClick={(event) => {
        if (!onSeek) return;
        const rect = event.currentTarget.getBoundingClientRect();
        onSeek(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * duration);
      }}
      sx={{
        position: "relative",
        height,
        borderRadius: 1,
        overflow: "hidden",
        bgcolor: "action.hover",
        cursor: onSeek ? "pointer" : "default",
      }}
    >
      {blocks.map((block) => (
        <Box
          key={block.key}
          sx={{
            position: "absolute",
            left: `${(block.startS / duration) * 100}%`,
            width: `${Math.max(0.2, ((block.endS - block.startS) / duration) * 100)}%`,
            top: 0,
            bottom: 0,
            bgcolor: block.color,
          }}
        />
      ))}
    </Box>
  );
}

// --- Rally & game status panel ---------------------------------------------

interface RallyRow {
  start: number;
  end: number;
}

function RallyPanel({
  job,
  gameStatus,
  onSaved,
  onSeek,
}: {
  job: Job;
  gameStatus: GameStatusOut | null;
  onSaved: (gs: GameStatusOut) => void;
  onSeek: (t: number) => void;
}) {
  const [rows, setRows] = useState<RallyRow[]>([]);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const savedRows = useMemo(
    () => (gameStatus?.rallies ?? []).map((r) => ({ start: r.start_time_s, end: r.end_time_s })),
    [gameStatus],
  );

  // Only resets the draft when the underlying data itself changes (a fresh
  // job, or a just-saved response) - NOT on every render, so edits in
  // progress survive switching to another tab and back (this panel stays
  // mounted the whole time - see DebugReviewPage's hidden-not-unmounted
  // tab panels).
  useEffect(() => {
    setRows(savedRows);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameStatus]);

  const duration = job.duration_s ?? 0;
  const dirty = JSON.stringify(rows) !== JSON.stringify(savedRows);

  function updateRow(index: number, field: "start" | "end", value: string) {
    const parsed = parseFloat(value);
    if (Number.isNaN(parsed)) return;
    setRows((prev) => prev.map((row, i) => (i === index ? { ...row, [field]: parsed } : row)));
  }

  function addRow() {
    const last = rows[rows.length - 1];
    const start = last ? last.end + 1 : 0;
    setRows((prev) => [...prev, { start, end: Math.min(start + 5, duration || start + 5) }]);
  }

  function deleteRow(index: number) {
    setRows((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleSave() {
    setSaving(true);
    setSaveError(null);
    setMessage(null);
    try {
      const updated = await api.overrideRallies(
        job.id,
        rows.map((r) => ({ start_time_s: r.start, end_time_s: r.end })),
      );
      onSaved(updated);
      setMessage("Saved - reprocessing action detection and stats with the corrected rallies now.");
      await api.recalibrateJob(job.id);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  if (!gameStatus) return <StatTilesSkeleton count={4} />;

  return (
    <Stack spacing={3}>
      <Typography variant="body2" color="text.secondary">
        Each colored block below is what the game-status model saw frame by frame (grey = no play, green = live
        play, blue = serve). The table lists the merged RALLY windows action detection and stats actually key off -
        correct any rally that's split wrong, missing, or drawn too wide/narrow, then save.
      </Typography>

      {duration > 0 && (
        <Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Detected segments (click to seek)
          </Typography>
          <Timeline
            duration={duration}
            onSeek={onSeek}
            blocks={gameStatus.segments.map((seg, i) => ({
              key: i,
              startS: seg.start_time_s,
              endS: seg.end_time_s,
              color: seg.state === "play" ? "success.main" : seg.state === "service" ? "info.main" : "action.disabledBackground",
            }))}
          />

          <Typography variant="subtitle2" sx={{ mt: 2.5, mb: 1 }}>
            Rally windows (editable below)
          </Typography>
          <Timeline
            duration={duration}
            height={16}
            blocks={rows.map((r, i) => ({ key: i, startS: r.start, endS: r.end, color: "warning.main" }))}
          />
        </Box>
      )}

      <Box>
        <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center", mb: 1 }}>
          <Typography variant="subtitle2">Rallies ({rows.length})</Typography>
          <Button size="small" onClick={addRow}>
            Add rally
          </Button>
        </Stack>

        {rows.length === 0 ? (
          <Alert severity="info">No rallies detected - add one manually if the video does have live play.</Alert>
        ) : (
          <Stack spacing={1}>
            {rows.map((row, i) => (
              <Stack key={i} direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <Typography variant="body2" color="text.secondary" sx={{ width: 28 }}>
                  #{i + 1}
                </Typography>
                <TextField
                  size="small"
                  label="Start (s)"
                  type="number"
                  value={row.start}
                  onChange={(event) => updateRow(i, "start", event.target.value)}
                  sx={{ width: 120 }}
                  slotProps={{ htmlInput: { step: 0.1 } }}
                />
                <TextField
                  size="small"
                  label="End (s)"
                  type="number"
                  value={row.end}
                  onChange={(event) => updateRow(i, "end", event.target.value)}
                  sx={{ width: 120 }}
                  slotProps={{ htmlInput: { step: 0.1 } }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ width: 90 }}>
                  {formatTime(row.start)}–{formatTime(row.end)}
                </Typography>
                <IconButton size="small" aria-label="Play from here" onClick={() => onSeek(row.start)}>
                  <PlayArrowIcon fontSize="small" />
                </IconButton>
                <IconButton size="small" aria-label="Delete rally" onClick={() => deleteRow(i)}>
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </Stack>
            ))}
          </Stack>
        )}
      </Box>

      {saveError && <Alert severity="error">{saveError}</Alert>}
      {message && !saveError && (
        <Alert severity="success" onClose={() => setMessage(null)}>
          {message}
        </Alert>
      )}

      {dirty && (
        <Stack direction="row" spacing={1}>
          <Button onClick={() => setRows(savedRows)} disabled={saving}>
            Discard
          </Button>
          <Button variant="contained" disabled={saving} onClick={() => void handleSave()}>
            Save &amp; reprocess
          </Button>
        </Stack>
      )}
    </Stack>
  );
}

// --- Ball detection panel ---------------------------------------------------

function BallPanel({
  job,
  trajectory,
  onSeek,
}: {
  job: Job;
  trajectory: BallTrajectory | null;
  onSeek: (t: number) => void;
}) {
  const [recalibrating, setRecalibrating] = useState(false);
  const [recalibrateError, setRecalibrateError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const duration = job.duration_s ?? 0;

  // A "gap" is a stretch of the decimated trajectory with no usable point
  // (neither a court nor a pixel reading) noticeably longer than the
  // typical spacing between points - i.e. the ball was lost for a real
  // stretch of time, not just between two adjacent sample points.
  const gaps = useMemo(() => {
    if (!trajectory || trajectory.points.length < 2) return [];
    const points = trajectory.points;
    const deltas = points.slice(1).map((p, i) => p.t - points[i].t).filter((d) => d > 0);
    if (deltas.length === 0) return [];
    const sorted = [...deltas].sort((a, b) => a - b);
    const medianDelta = sorted[Math.floor(sorted.length / 2)];
    const gapThreshold = Math.max(medianDelta * 3, 0.5);

    const found: { startS: number; endS: number }[] = [];
    for (let i = 1; i < points.length; i++) {
      const gapLength = points[i].t - points[i - 1].t;
      if (gapLength > gapThreshold) found.push({ startS: points[i - 1].t, endS: points[i].t });
    }
    return found;
  }, [trajectory]);

  async function handleReselect() {
    setRecalibrating(true);
    setRecalibrateError(null);
    setMessage(null);
    try {
      await api.recalibrateJob(job.id);
      setMessage("Reselecting the ball trajectory from raw detections against the current court calibration.");
    } catch (err) {
      setRecalibrateError(err instanceof Error ? err.message : String(err));
    } finally {
      setRecalibrating(false);
    }
  }

  if (!trajectory) return <StatTilesSkeleton count={4} />;

  return (
    <Stack spacing={3}>
      <Typography variant="body2" color="text.secondary">
        The on-video marker and minimap (see the video's own Annotations menu) play back this job's chosen ball
        trajectory. A red stretch on the timeline below is a real gap in coverage (no detection at all for
        noticeably longer than usual) - worth watching that moment to see whether the ball was genuinely out of
        frame, or the detector actually missed it.
      </Typography>

      {duration > 0 && (
        <Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Coverage (click to seek)
          </Typography>
          <Timeline
            duration={duration}
            onSeek={onSeek}
            blocks={[
              { key: "base", startS: 0, endS: duration, color: "success.main" },
              ...gaps.map((g, i) => ({ key: `gap-${i}`, startS: g.startS, endS: g.endS, color: "error.main" })),
            ]}
          />
        </Box>
      )}

      <Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Detection gaps ({gaps.length})
        </Typography>
        {gaps.length === 0 ? (
          <Alert severity="success">No significant gaps found in the ball trajectory.</Alert>
        ) : (
          <Stack spacing={1}>
            {gaps.map((gap, i) => (
              <Stack key={i} direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <Typography variant="body2" sx={{ width: 160 }}>
                  {formatTime(gap.startS)} – {formatTime(gap.endS)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {(gap.endS - gap.startS).toFixed(1)}s missing
                </Typography>
                <IconButton size="small" aria-label="Play from here" onClick={() => onSeek(gap.startS)}>
                  <PlayArrowIcon fontSize="small" />
                </IconButton>
              </Stack>
            ))}
          </Stack>
        )}
      </Box>

      <Card variant="outlined" sx={{ p: 2.5 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Reselect ball trajectory
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Re-picks the trajectory from every raw detection candidate logged during Ball Detection, using the
          current court calibration - the same step that runs after saving a new calibration. Try this if the
          trajectory looks wrong and you've just fixed calibration on the Court detection tab, without re-running
          the (expensive) detector itself.
        </Typography>
        <Button variant="outlined" disabled={recalibrating} onClick={() => void handleReselect()}>
          Reselect from raw detections
        </Button>
        {recalibrateError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {recalibrateError}
          </Alert>
        )}
        {message && !recalibrateError && (
          <Alert severity="success" sx={{ mt: 2 }} onClose={() => setMessage(null)}>
            {message}
          </Alert>
        )}
      </Card>
    </Stack>
  );
}

// --- Player detection panel --------------------------------------------------

function PlayerPanel({ job, trajectory }: { job: Job; trajectory: PlayerTrajectory | null }) {
  const navigate = useNavigate();

  if (!trajectory) return <StatTilesSkeleton count={4} />;

  const lastFrame = trajectory.frames[trajectory.frames.length - 1];
  const playerCount = lastFrame?.players.length ?? 0;
  const unidentifiedCount = lastFrame?.players.filter((p) => !p.name).length ?? 0;

  return (
    <Stack spacing={3}>
      <Typography variant="body2" color="text.secondary">
        Every tracked player's box plays back on the video (see the Annotations menu), labelled with whatever name
        is currently assigned. Renaming, ignoring, merging, or resetting identities is all done from the full
        Player Identification tool, not here - this tab is for spotting whether tracking/attribution actually
        looks right.
      </Typography>

      <Card variant="outlined" sx={{ p: 2.5 }}>
        <Stack direction="row" spacing={2} sx={{ alignItems: "center", flexWrap: "wrap" }}>
          <Chip label={`${playerCount} tracked player${playerCount === 1 ? "" : "s"}`} />
          <Chip
            label={`${unidentifiedCount} unidentified`}
            color={unidentifiedCount > 0 ? "warning" : "success"}
            variant={unidentifiedCount > 0 ? "filled" : "outlined"}
          />
        </Stack>
        <Button
          variant="contained"
          sx={{ mt: 2 }}
          onClick={() => navigate(`/game/setup/player-identification?job=${job.id}`)}
        >
          Open Player Identification
        </Button>
      </Card>
    </Stack>
  );
}

// --- Court detection panel ----------------------------------------------------

function CourtPanel({
  job,
  points,
  frameUrl,
}: {
  job: Job;
  points: CalibrationPointsOut | null;
  frameUrl: string | null;
}) {
  const navigate = useNavigate();

  if (!points) return <StatTilesSkeleton count={4} />;

  return (
    <Stack spacing={3}>
      <Typography variant="body2" color="text.secondary">
        Court calibration is what every other tab's real-world positions (court coordinates, minimap, camera-pose
        height) are derived from - if the ball/player minimap looks off, this is usually why. Editing the actual
        points is done from the full Court Calibration tool, not here.
      </Typography>

      <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
        <Chip
          label={points.calibrated ? "Calibrated" : "Not calibrated"}
          color={points.calibrated ? "success" : "warning"}
        />
        <Chip
          label={points.confirmed ? "Confirmed" : "Not confirmed"}
          color={points.confirmed ? "success" : "default"}
          variant={points.confirmed ? "filled" : "outlined"}
        />
        <Chip
          label={points.net_top_calibrated ? "Net height set" : "Net height not set"}
          color={points.net_top_calibrated ? "success" : "default"}
          variant={points.net_top_calibrated ? "filled" : "outlined"}
        />
        <Chip
          label={
            points.camera_pose_available == null
              ? "No camera pose"
              : points.camera_pose_available
                ? `Camera pose solved (${points.camera_pose_reprojection_error_px?.toFixed(1) ?? "?"}px error)`
                : "Camera pose failed"
          }
          color={points.camera_pose_available ? "success" : "default"}
          variant={points.camera_pose_available ? "filled" : "outlined"}
        />
      </Stack>

      {frameUrl && (
        <Box component="img" src={frameUrl} alt="Calibration reference frame" sx={{ width: "100%", borderRadius: 1 }} />
      )}

      <Box>
        <Button variant="contained" onClick={() => navigate(`/game/setup/court-calibration?job=${job.id}`)}>
          Open Court Calibration
        </Button>
      </Box>
    </Stack>
  );
}

// --- Page --------------------------------------------------------------------

// Debug tab's "Check accuracy" destination (see DebugPanel.tsx). Mirrors
// ResultsView.tsx's own layout: one persistent video preview on the left
// with every annotation loaded at once (Annotations menu picks which
// shows), a tabbed, independently-scrolling panel on the right for the
// model-specific controls. Every tab's data is fetched up front as soon as
// a game is picked - see the loadEverything effect below - and every tab
// panel stays mounted (hidden, not unmounted) once loaded, so switching
// between them is instant and never re-fetches or loses in-progress edits
// (e.g. a half-edited rally row).
export function DebugReviewPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get(JOB_QUERY_PARAM);

  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobsError, setJobsError] = useState<string | null>(null);

  const [job, setJob] = useState<Job | null>(null);
  const [gameStatus, setGameStatus] = useState<GameStatusOut | null>(null);
  const [ballTrajectory, setBallTrajectory] = useState<BallTrajectory | null>(null);
  const [playerTrajectory, setPlayerTrajectory] = useState<PlayerTrajectory | null>(null);
  const [calibrationPoints, setCalibrationPoints] = useState<CalibrationPointsOut | null>(null);
  const [calibrationFrameUrl, setCalibrationFrameUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [tab, setTab] = useState(() => readTabFromUrl());

  useEffect(() => {
    api
      .listJobs()
      .then(setJobs)
      .catch((err) => setJobsError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    function handlePopState() {
      setTab(readTabFromUrl());
    }
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Loads everything this page could possibly show for the selected game,
  // all at once, up front - every tab below just reads from state already
  // in memory by the time someone clicks it, rather than fetching on
  // first visit to that tab. Best-effort per-endpoint (same pattern
  // ResultsView.tsx uses) - a job missing one log (no ball tracking yet,
  // say) still shows everything else instead of failing the whole page.
  useEffect(() => {
    setJob(null);
    setGameStatus(null);
    setBallTrajectory(null);
    setPlayerTrajectory(null);
    setCalibrationPoints(null);
    setCalibrationFrameUrl(null);
    setLoadError(null);
    if (!jobId) return;

    let cancelled = false;
    let objectUrl: string | null = null;

    api
      .getJob(jobId)
      .then((res) => !cancelled && setJob(res))
      .catch((err) => !cancelled && setLoadError(err instanceof Error ? err.message : String(err)));
    api.getGameStatus(jobId).then((res) => !cancelled && setGameStatus(res)).catch(() => undefined);
    api.getBallTrajectory(jobId).then((res) => !cancelled && setBallTrajectory(res)).catch(() => undefined);
    api.getPlayerTrajectory(jobId).then((res) => !cancelled && setPlayerTrajectory(res)).catch(() => undefined);
    api.getCalibrationPoints(jobId).then((res) => !cancelled && setCalibrationPoints(res)).catch(() => undefined);
    api
      .getCalibrationFrame(jobId)
      .then((url) => {
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        objectUrl = url;
        setCalibrationFrameUrl(url);
      })
      .catch(() => undefined);

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [jobId]);

  useEffect(() => {
    document.title = job ? `Accuracy review · ${job.original_filename}` : "Accuracy review";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [job]);

  function changeTab(index: number) {
    setTab(index);
    writeTabToUrl(index);
  }

  function selectJob(id: string) {
    const url = new URL(window.location.href);
    if (id) url.searchParams.set(JOB_QUERY_PARAM, id);
    else url.searchParams.delete(JOB_QUERY_PARAM);
    navigate(`${url.pathname}${url.search}`, { replace: true });
  }

  function seekTo(t: number) {
    const el = videoRef.current;
    if (el) el.currentTime = t;
  }

  const eligibleJobs = jobs.filter((j) => j.status === "complete");

  return (
    <Box
      sx={{
        height: { xs: "auto", md: PAGE_CONTENT_HEIGHT },
        display: "flex",
        flexDirection: "column",
        overflow: { xs: "visible", md: "hidden" },
      }}
    >
      <Stack
        direction="row"
        spacing={2}
        sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 1, mb: 2, flexShrink: 0 }}
      >
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate("/settings?tab=debug")}>
          Back to Debug
        </Button>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          Accuracy review
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        <TextField
          select
          size="small"
          label="Game"
          value={jobId ?? ""}
          onChange={(event) => selectJob(event.target.value)}
          sx={{ minWidth: 260 }}
          disabled={eligibleJobs.length === 0}
        >
          {eligibleJobs.map((j) => (
            <MenuItem key={j.id} value={j.id}>
              {j.original_filename}
            </MenuItem>
          ))}
        </TextField>
      </Stack>

      {jobsError && (
        <Alert severity="error" sx={{ mb: 2, flexShrink: 0 }}>
          {jobsError}
        </Alert>
      )}
      {loadError && (
        <Alert severity="error" sx={{ mb: 2, flexShrink: 0 }}>
          {loadError}
        </Alert>
      )}

      {!jobId ? (
        <Alert severity="info" sx={{ maxWidth: 600 }}>
          {eligibleJobs.length === 0
            ? "No completed games yet - finish processing one first."
            : "Pick a game above to review its detections."}
        </Alert>
      ) : !job ? (
        <Box sx={{ flex: 1, minHeight: 0, display: "flex", flexDirection: { xs: "column", md: "row" }, gap: 3 }}>
          <Box sx={{ flex: { xs: "0 0 auto", md: "3 1 600px", lg: "5 1 760px" }, minWidth: { xs: 0, md: 420, lg: 520 } }}>
            <Skeleton variant="rounded" sx={{ width: "100%", aspectRatio: "16 / 9", height: { xs: "auto", md: "100%" } }} />
          </Box>
          <Box sx={{ flex: { xs: "1 1 auto", md: "2 1 340px", lg: "2 1 380px" }, minWidth: { xs: 0, md: 300, lg: 340 } }}>
            <StatTilesSkeleton count={4} />
          </Box>
        </Box>
      ) : (
        <Box
          sx={{
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            gap: 3,
            overflow: { xs: "visible", md: "hidden" },
          }}
        >
          <Box
            sx={{
              flex: { xs: "0 0 auto", md: "3 1 600px", lg: "5 1 760px" },
              minWidth: { xs: 0, md: 420, lg: 520 },
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <Box sx={{ position: "relative", flex: { xs: "0 0 auto", md: 1 }, minHeight: 0, aspectRatio: { xs: "16 / 9", md: "auto" } }}>
              <VideoPlayer
                videoRef={videoRef}
                src={api.sourceVideoUrl(job.id)}
                gameStatusSegments={gameStatus?.segments}
                ballTrajectory={ballTrajectory?.points}
                courtLengthM={ballTrajectory?.court_length_m}
                courtWidthM={ballTrajectory?.court_width_m}
                frameW={ballTrajectory?.frame_w ?? playerTrajectory?.frame_w}
                frameH={ballTrajectory?.frame_h ?? playerTrajectory?.frame_h}
                playerTrajectory={playerTrajectory?.frames}
              />
            </Box>
          </Box>

          <Box
            sx={{
              flex: { xs: "1 1 auto", md: "2 1 340px", lg: "2 1 380px" },
              minWidth: { xs: 0, md: 300, lg: 340 },
              minHeight: 0,
              display: "flex",
              flexDirection: "column",
              overflow: { xs: "visible", md: "hidden" },
            }}
          >
            <Stack
              direction="row"
              sx={{
                alignItems: "center",
                mb: 2,
                borderBottom: 1,
                borderColor: "divider",
                flexShrink: 0,
                position: { xs: "sticky", md: "static" },
                top: APP_BAR_HEIGHT_PX,
                bgcolor: "background.default",
                zIndex: 1,
              }}
            >
              <Tabs value={tab} onChange={(_event, index) => changeTab(index)} variant="scrollable" scrollButtons="auto" sx={{ flex: 1, minWidth: 0 }}>
                <Tab label="Rally & game status" />
                <Tab label="Ball detection" />
                <Tab label="Player detection" />
                <Tab label="Court detection" />
              </Tabs>
            </Stack>

            <Box
              sx={{
                flex: 1,
                minHeight: 0,
                maxHeight: { xs: "60vh", md: "none" },
                overflowY: "auto",
                pr: { xs: 0, md: 0.5 },
              }}
            >
              {/* Every panel stays mounted once the job's data starts loading -
                  hidden via display:none rather than unmounted - so switching
                  tabs is instant and never discards an in-progress edit (e.g.
                  the rally editor's draft rows). */}
              <Box sx={{ display: tab === 0 ? "block" : "none" }}>
                <RallyPanel job={job} gameStatus={gameStatus} onSaved={setGameStatus} onSeek={seekTo} />
              </Box>
              <Box sx={{ display: tab === 1 ? "block" : "none" }}>
                <BallPanel job={job} trajectory={ballTrajectory} onSeek={seekTo} />
              </Box>
              <Box sx={{ display: tab === 2 ? "block" : "none" }}>
                <PlayerPanel job={job} trajectory={playerTrajectory} />
              </Box>
              <Box sx={{ display: tab === 3 ? "block" : "none" }}>
                <CourtPanel job={job} points={calibrationPoints} frameUrl={calibrationFrameUrl} />
              </Box>
            </Box>
          </Box>
        </Box>
      )}
    </Box>
  );
}
