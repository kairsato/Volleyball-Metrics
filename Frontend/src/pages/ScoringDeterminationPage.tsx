import { useEffect, useMemo, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import type { BallTrajectory, GameStatusSegment, Job, PlayerTrajectory, Rally } from "../lib/types";
import { ScoreSection } from "../components/results/ScoreSection";
import { ToolPageSkeleton } from "../components/Skeletons";
import type { FlatEvent } from "../components/results/types";
import { toBoundedAbsolute, VideoPlayer } from "../components/VideoPlayer";

// Same viewport-fit approach as ResultsView.tsx/PlayerIdentificationPage.tsx
// - the page itself never scrolls, only ScoreSection's own content does.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

// The other full page a Setup card (see SetupTab.tsx) links to - a "back to
// results" header plus ScoreSection filling the rest of the page. Unlike
// Player Identification, the video preview stays here: judging who won a
// rally means actually watching it, not just picking from a list. Scoring
// never changes the job's own status (only names/redoing calibration do),
// so unlike the sibling pages there's no onJobUpdated to thread through.
export function ScoringDeterminationPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");

  const [job, setJob] = useState<Job | null>(null);
  const [rallies, setRallies] = useState<Rally[]>([]);
  // Every player's touches, flattened the same way ResultsView's own
  // flatEvents does - only needed here to derive hitTimestamps below (the
  // Ball Trajectory annotation's flight-segment boundaries), so nothing
  // else about them is kept.
  const [flatEvents, setFlatEvents] = useState<FlatEvent[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Quality/annotations are a standard part of the player wherever the
  // source video plays, not a Results-page special case - see VideoPlayer's
  // own props for what each one needs. Deliberately never fetching/passing
  // scoreRallies (see the VideoPlayer usage below, and rallies/setRallies
  // above which feed ScoreSection instead): Score's whole point is showing
  // a settled score, which is exactly what a reviewer is here to determine/
  // correct, so surfacing a possibly-still-wrong computed one on top of
  // that would fight the page's own purpose. Game Status is unaffected -
  // it's driven by its own gameStatusSegments below, not rallies/
  // scoreRallies, so it stays available here regardless.
  const [qualities, setQualities] = useState<string[]>(["original"]);
  const [originalLabel, setOriginalLabel] = useState("Original");
  const [quality, setQuality] = useState("original");
  const [ballTrajectory, setBallTrajectory] = useState<BallTrajectory | null>(null);
  const [playerTrajectory, setPlayerTrajectory] = useState<PlayerTrajectory | null>(null);
  const [gameStatusSegments, setGameStatusSegments] = useState<GameStatusSegment[]>([]);

  useEffect(() => {
    if (!jobId) return;
    api
      .getJob(jobId)
      .then(setJob)
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    api
      .getResults(jobId)
      .then((res) => {
        if (cancelled) return;
        setRallies(res.rallies);
        const events: FlatEvent[] = [];
        for (const [playerId, stat] of Object.entries(res.players)) {
          for (const event of stat.events) {
            events.push({ ...event, playerId, playerName: stat.name ?? `Player ${playerId}` });
          }
        }
        setFlatEvents(events.sort((a, b) => a.frame_idx - b.frame_idx));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  // Same shape as ResultsView's own quality-fetching effect - see its
  // comment for why this auto-selects the lowest-res rendition rather than
  // defaulting to streaming the original upload.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    api
      .getQualities(jobId)
      .then((res) => {
        if (cancelled) return;
        setQualities(res.qualities);
        setOriginalLabel(res.original_label);
        const el = videoRef.current;
        const stillAtStart = !el || (el.paused && el.currentTime === 0);
        if (stillAtStart && res.qualities.length > 1) setQuality(res.qualities[res.qualities.length - 1]);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  // Best-effort, same as ResultsView - a job with no ball/player tracking
  // or game-status data yet just means those annotations don't appear, not
  // an error for the whole page.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    api
      .getBallTrajectory(jobId)
      .then((res) => !cancelled && setBallTrajectory(res))
      .catch(() => undefined);
    api
      .getPlayerTrajectory(jobId)
      .then((res) => !cancelled && setPlayerTrajectory(res))
      .catch(() => undefined);
    api
      .getGameStatus(jobId)
      .then((res) => !cancelled && setGameStatusSegments(res.segments))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const hitTimestamps = useMemo(() => flatEvents.map((e) => e.timestamp_s), [flatEvents]);

  useEffect(() => {
    document.title = job ? `${job.original_filename} · Scoring Determination` : "Scoring Determination";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [job]);

  // rallies (from /results) are already relative to the warmup period's
  // start once one is confirmed - see Backend/API/warmup.py. currentTime
  // (kept in sync via VideoPlayer's onTimeUpdate below) is relative the
  // same way, so ScoreSection's rally highlighting (which compares it
  // against rallies[i].start_time_s/end_time_s) lines up; seekTo converts
  // back the other way since the raw <video> element only deals in
  // absolute time.
  const warmupStartS = job?.warmup_confirmed ? job.warmup_start_s ?? 0 : 0;
  const warmupEndS = job?.warmup_confirmed ? job.warmup_end_s ?? Infinity : Infinity;

  function seekTo(timeS: number) {
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = toBoundedAbsolute(timeS, warmupStartS, warmupEndS);
    void el.play();
  }

  if (!jobId) return <Navigate to="/games" replace />;
  if (loadError) return <Alert severity="error">{loadError}</Alert>;
  if (!job) return <ToolPageSkeleton />;

  if (job.status !== "complete") {
    return (
      <Box sx={{ maxWidth: 560 }}>
        <Alert severity="warning" sx={{ mb: 2 }}>
          This video hasn't finished processing yet, so scoring isn't available.
        </Alert>
        <Button variant="contained" onClick={() => navigate(`/game?job=${jobId}`)}>
          Back to game
        </Button>
      </Box>
    );
  }

  // height: "100%" lets this stretch to match the Scoring Determination +
  // Estimated Score column without distorting the video - VideoPlayer
  // itself letterboxes (objectFit: "contain") to whatever box it ends up in.
  const videoElement = (
    <Box sx={{ height: "100%" }}>
      <VideoPlayer
        videoRef={videoRef}
        src={api.sourceVideoUrl(jobId, quality)}
        boundStartS={warmupStartS}
        boundEndS={warmupEndS}
        onTimeUpdate={setCurrentTime}
        qualities={qualities}
        quality={quality}
        onQualityChange={setQuality}
        originalQualityLabel={originalLabel}
        ballTrajectory={ballTrajectory?.points}
        courtLengthM={ballTrajectory?.court_length_m}
        courtWidthM={ballTrajectory?.court_width_m}
        frameW={ballTrajectory?.frame_w ?? playerTrajectory?.frame_w}
        frameH={ballTrajectory?.frame_h ?? playerTrajectory?.frame_h}
        hitTimestamps={hitTimestamps}
        playerTrajectory={playerTrajectory?.frames}
        gameStatusSegments={gameStatusSegments}
      />
    </Box>
  );

  return (
    <Box sx={{ height: PAGE_CONTENT_HEIGHT, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Button
        size="small"
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(`/game?job=${jobId}&tab=setup`)}
        sx={{ alignSelf: "flex-start", mb: 2, flexShrink: 0 }}
      >
        Back to results
      </Button>

      <Box sx={{ flex: 1, minHeight: 0 }}>
        <ScoreSection job={job} rallies={rallies} currentTime={currentTime} onSeek={seekTo} videoElement={videoElement} />
      </Box>
    </Box>
  );
}
