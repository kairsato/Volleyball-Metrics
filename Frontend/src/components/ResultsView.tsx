import { useEffect, useMemo, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Badge from "@mui/material/Badge";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Typography from "@mui/material/Typography";
import StopCircleIcon from "@mui/icons-material/StopCircle";
import { api } from "../lib/api";
import type { BallTrajectory, Job, PlayerTrajectory, ResultsOut, ScoreOut, TeamEntry } from "../lib/types";
import { ActionsTab } from "./results/ActionsTab";
import { AnalyticsTab } from "./results/AnalyticsTab";
import { RalliesTab } from "./results/RalliesTab";
import { SetupTab } from "./results/SetupTab";
import { StatsTab } from "./results/StatsTab";
import type { FlatEvent } from "./results/types";
import { StatTilesSkeleton } from "./Skeletons";
import { toBoundedAbsolute, toBoundedRelative, VideoPlayer } from "./VideoPlayer";

// How long each clip plays before a "play all" playlist jumps to the next
// timestamp - long enough to actually see the touch, short enough to stay
// a highlight reel rather than just... playing the whole video.
const CLIP_DURATION_S = 3;

// The whole page is capped to exactly the viewport height minus the real
// chrome around it (App.tsx's AppBar + the page wrapper's padding above and
// below) and laid out as a column flexbox - the video/tabs row then gets
// `flex: 1` and fills whatever's left after the (optional) warning banner,
// so nothing has to guess that banner's height. This is what actually
// guarantees no page-level scroll, rather than a hardcoded top offset that
// only happened to match when nothing sat above the video.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

// Mirrors the tab into ?tab=<name> - readable/shareable alongside ?jobId=,
// same native-URLSearchParams approach App.tsx uses for the job itself.
// Stats leads (win/loss + the graphs), Analytics (formerly "Overall") is
// just the Action Quality breakdown - see AnalyticsTab.tsx.
const TAB_NAMES = ["stats", "analytics", "rallies", "actions", "setup"] as const;
const ACTIONS_TAB_INDEX = TAB_NAMES.indexOf("actions");
const TAB_QUERY_PARAM = "tab";

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

interface ActionFilterPreset {
  playerId: string;
  actionType: string;
}

interface ResultsViewProps {
  job: Job;
}

export function ResultsView({ job }: ResultsViewProps) {
  const [results, setResults] = useState<ResultsOut | null>(null);
  const [ballTrajectory, setBallTrajectory] = useState<BallTrajectory | null>(null);
  // All three below feed VideoPlayer's Annotations: Player tracking (needs
  // just the trajectory itself), Score (needs score + team names, so both
  // score and teams), and Minimap/Ball tracking (already covered by
  // ballTrajectory above). Every one of them is best-effort, same as
  // ballTrajectory already is - a video with no player tracking data (or
  // no Score configured) just means that annotation option doesn't appear
  // in the menu, not an error state for the whole page.
  const [playerTrajectory, setPlayerTrajectory] = useState<PlayerTrajectory | null>(null);
  const [score, setScore] = useState<ScoreOut | null>(null);
  const [teams, setTeams] = useState<TeamEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState(() => readTabFromUrl());
  const [actionFilterPreset, setActionFilterPreset] = useState<ActionFilterPreset | null>(null);

  useEffect(() => {
    function handlePopState() {
      setTab(readTabFromUrl());
    }
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function changeTab(index: number) {
    setTab(index);
    writeTabToUrl(index);
    setActionFilterPreset(null);
  }

  function jumpToAction(playerId: string, actionType: string) {
    setTab(ACTIONS_TAB_INDEX);
    writeTabToUrl(ACTIONS_TAB_INDEX);
    setActionFilterPreset({ playerId, actionType });
  }

  // The <video> element's own currentTime deals in ABSOLUTE video time;
  // everything that comes from `results` (rally/action timestamps, and
  // therefore every seekTo/playAll caller) is already RELATIVE to
  // warmupStartS - rebased server-side once a warmup period is confirmed
  // (see Backend/API/warmup.py). VideoPlayer's toBoundedAbsolute/
  // toBoundedRelative are the only place that conversion happens, so the
  // player's displayed clock, its scrubber, and every "jump to this rally/
  // hit" caller here all agree on 0:00 meaning "the warmup period's
  // start", not "the start of the raw file".
  const warmupStartS = job.warmup_confirmed ? job.warmup_start_s ?? 0 : 0;
  const warmupEndS = job.warmup_confirmed ? job.warmup_end_s ?? Infinity : Infinity;

  const videoRef = useRef<HTMLVideoElement>(null);
  const playlistIndexRef = useRef(0);
  const [playlist, setPlaylist] = useState<number[] | null>(null);
  const [playlistIndex, setPlaylistIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  const [qualities, setQualities] = useState<string[]>(["original"]);
  // Defaults to "original" until getQualities resolves below, then jumps to
  // the best actually-transcoded rendition if one exists - streaming a raw
  // upload (potentially a multi-GB, high-bitrate phone recording) for a
  // whole video's worth of playback is what a browser tab running out of
  // memory during long playback traced back to. "original" only stays the
  // real default for a job with no renditions yet (still processing, or
  // predates the transcoding stage entirely) - see transcode.py.
  const [quality, setQuality] = useState("original");

  useEffect(() => {
    let cancelled = false;
    api
      .getQualities(job.id)
      .then((res) => {
        if (cancelled) return;
        setQualities(res.qualities);
        // getQualities always returns ["original", ...generated], with
        // generated in TRANSCODE_TIERS' own highest-to-lowest order - so
        // index 1, if present, is the best available transcoded tier.
        // Only auto-switches to it while still sitting at the very start
        // (the common case - this resolves quickly, well before playback
        // normally begins): swapping `src` this way skips VideoPlayer's own
        // handleQualityChange, which is what actually preserves playback
        // position/state across a quality change, so doing this once
        // someone's already mid-playback would silently reset them to 0:00.
        const el = videoRef.current;
        const stillAtStart = !el || (el.paused && el.currentTime === 0);
        if (stillAtStart && res.qualities.length > 1) setQuality(res.qualities[1]);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [job.id]);

  // The filename lives in the browser tab instead of an on-page heading -
  // one less thing eating vertical space above the video.
  useEffect(() => {
    document.title = job.original_filename;
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [job.original_filename]);

  useEffect(() => {
    let cancelled = false;

    api
      .getResults(job.id)
      .then((res) => !cancelled && setResults(res))
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : String(err)));

    // Best-effort - a video with no ball_speed.json yet (or ball tracking
    // just never found much) just means no minimap shows, not an error
    // state for the whole page.
    api
      .getBallTrajectory(job.id)
      .then((res) => !cancelled && setBallTrajectory(res))
      .catch(() => undefined);

    // Same best-effort reasoning as ballTrajectory above, for the other
    // three Annotations.
    api
      .getPlayerTrajectory(job.id)
      .then((res) => !cancelled && setPlayerTrajectory(res))
      .catch(() => undefined);
    api
      .getScore(job.id)
      .then((res) => !cancelled && setScore(res))
      .catch(() => undefined);
    api
      .getTeams()
      .then((res) => !cancelled && setTeams(res.teams))
      .catch(() => undefined);

    return () => {
      cancelled = true;
    };
  }, [job.id]);

  // currentTime (relative to warmupStartS, same as every rally/action
  // timestamp) is kept in sync via VideoPlayer's onTimeUpdate below - the
  // Rallies/Actions tabs use it to highlight whatever's currently playing.

  useEffect(() => {
    const el = videoRef.current;
    if (!el || playlist === null) return;

    function handleTimeUpdate() {
      const el = videoRef.current;
      if (!el) return;
      // playlist entries are relative (same as every other rally/action
      // timestamp) - compare against the relative position, not the raw
      // absolute currentTime.
      const clipStart = playlist![playlistIndexRef.current];
      if (toBoundedRelative(el.currentTime, warmupStartS, warmupEndS) - clipStart < CLIP_DURATION_S) return;

      const nextIndex = playlistIndexRef.current + 1;
      if (nextIndex < playlist!.length) {
        playlistIndexRef.current = nextIndex;
        setPlaylistIndex(nextIndex);
        el.currentTime = toBoundedAbsolute(playlist![nextIndex], warmupStartS, warmupEndS);
      } else {
        setPlaylist(null);
        el.pause();
      }
    }

    el.addEventListener("timeupdate", handleTimeUpdate);
    return () => el.removeEventListener("timeupdate", handleTimeUpdate);
  }, [playlist, warmupStartS, warmupEndS]);

  // Drives the number badge on the Setup tab below - the same
  // needs_player_id/needs_scoring_review flags the video grid uses for its
  // own "Setup needed" chip (see VideoGrid.tsx), computed server-side so
  // this doesn't need its own fetch just to count unnamed players.
  const setupAttentionCount = (job.needs_player_id ? 1 : 0) + (job.needs_scoring_review ? 1 : 0);

  // Same team_x_id/team_y_id -> name lookup StatsTab already does - VideoPlayer's
  // Score annotation only ever needs the resolved names, not the raw config.
  const teamXName = teams.find((t) => t.id === score?.config.team_x_id)?.name ?? "Team X";
  const teamYName = teams.find((t) => t.id === score?.config.team_y_id)?.name ?? "Team Y";

  const flatEvents = useMemo<FlatEvent[]>(() => {
    if (!results) return [];
    const events: FlatEvent[] = [];
    for (const [playerId, stat] of Object.entries(results.players)) {
      for (const event of stat.events) {
        events.push({ ...event, playerId, playerName: stat.name ?? `Player ${playerId}` });
      }
    }
    return events.sort((a, b) => a.frame_idx - b.frame_idx);
  }, [results]);

  // Every touch's timestamp, in the same sorted-by-frame_idx (so also
  // sorted by timestamp_s, for a fixed-fps video) order flatEvents is
  // already in - VideoPlayer's Ball Trajectory annotation uses these as
  // the current flight's segment boundaries (see BallTrajectoryOverlay.tsx).
  const hitTimestamps = useMemo(() => flatEvents.map((e) => e.timestamp_s), [flatEvents]);

  // `pause: true` is for jumping to a single reference frame (the player
  // preview's "jump to it") - the point there is to freeze on that exact
  // moment for a visual comparison, not to keep playing past it the way
  // seeking from Rallies/Actions ("Play", "Play all touches") should.
  function seekTo(timeS: number, pause = false) {
    setPlaylist(null);
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = toBoundedAbsolute(timeS, warmupStartS, warmupEndS);
    if (pause) el.pause();
    else void el.play();
  }

  function playAll(timestamps: number[]) {
    if (timestamps.length === 0) return;
    playlistIndexRef.current = 0;
    setPlaylistIndex(0);
    setPlaylist(timestamps);
    const el = videoRef.current;
    if (el) {
      el.currentTime = toBoundedAbsolute(timestamps[0], warmupStartS, warmupEndS);
      void el.play();
    }
  }

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!results) {
    return (
      <Box
        sx={{
          height: { xs: "auto", sm: PAGE_CONTENT_HEIGHT },
          display: "flex",
          flexDirection: { xs: "column", sm: "row" },
          gap: 3,
          overflow: { xs: "visible", sm: "hidden" },
        }}
      >
        <Box sx={{ flex: { xs: "0 0 auto", sm: "5 1 760px" }, minWidth: { xs: 0, sm: 520 } }}>
          <Skeleton variant="rounded" sx={{ width: "100%", aspectRatio: "16 / 9", height: { xs: "auto", sm: "100%" } }} />
        </Box>
        <Box sx={{ flex: { xs: "1 1 auto", sm: "2 1 380px" }, minWidth: { xs: 0, sm: 340 } }}>
          <Stack direction="row" spacing={3} sx={{ mb: 2, pb: 1.5, borderBottom: 1, borderColor: "divider" }}>
            {["Stats", "Analytics", "Rallies", "Actions", "Setup"].map((label) => (
              <Skeleton key={label} variant="text" width={50} height={28} />
            ))}
          </Stack>
          <Skeleton variant="rounded" height={90} sx={{ mb: 2 }} />
          <StatTilesSkeleton count={4} />
        </Box>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        height: { xs: "auto", sm: PAGE_CONTENT_HEIGHT },
        display: "flex",
        flexDirection: "column",
        overflow: { xs: "visible", sm: "hidden" },
      }}
    >
      <Box
        sx={{
          flex: 1,
          minHeight: 0,
          display: "flex",
          flexDirection: { xs: "column", sm: "row" },
          gap: 3,
          overflow: { xs: "visible", sm: "hidden" },
        }}
      >
        <Box
          sx={{
            flex: { xs: "0 0 auto", sm: "5 1 760px" },
            minWidth: { xs: 0, sm: 520 },
            minHeight: 0,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <Box
            sx={{
              position: "relative",
              flex: { xs: "0 0 auto", sm: 1 },
              minHeight: 0,
              aspectRatio: { xs: "16 / 9", sm: "auto" },
            }}
          >
            <VideoPlayer
              videoRef={videoRef}
              src={api.sourceVideoUrl(job.id, quality)}
              boundStartS={warmupStartS}
              boundEndS={warmupEndS}
              onTimeUpdate={setCurrentTime}
              qualities={qualities}
              quality={quality}
              onQualityChange={setQuality}
              ballTrajectory={ballTrajectory?.points}
              courtLengthM={ballTrajectory?.court_length_m}
              courtWidthM={ballTrajectory?.court_width_m}
              frameW={ballTrajectory?.frame_w ?? playerTrajectory?.frame_w}
              frameH={ballTrajectory?.frame_h ?? playerTrajectory?.frame_h}
              hitTimestamps={hitTimestamps}
              playerTrajectory={playerTrajectory?.frames}
              rallies={results.rallies}
              scoreRallies={score?.result?.rallies}
              games={score?.result?.games}
              teamXName={teamXName}
              teamYName={teamYName}
              enableArrowKeySeek
            />
          </Box>

          {playlist !== null && (
            <Stack direction="row" sx={{ mt: 1, alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
              <Typography variant="body2" color="text.secondary">
                Playing clip {playlistIndex + 1} of {playlist.length}
              </Typography>
              <Button size="small" startIcon={<StopCircleIcon />} onClick={() => setPlaylist(null)}>
                Stop
              </Button>
            </Stack>
          )}
        </Box>

        <Box
          sx={{
            flex: { xs: "1 1 auto", sm: "2 1 380px" },
            minWidth: { xs: 0, sm: 340 },
            minHeight: 0,
            overflowY: { xs: "visible", sm: "auto" },
            pr: { xs: 0, sm: 0.5 },
          }}
        >
          <Stack
            direction="row"
            sx={{
              alignItems: "center",
              mb: 2,
              borderBottom: 1,
              borderColor: "divider",
              position: "sticky",
              top: { xs: APP_BAR_HEIGHT_PX, sm: 0 },
              bgcolor: "background.default",
              zIndex: 1,
            }}
          >
            <Tabs value={tab} onChange={(_, value) => changeTab(value)} variant="scrollable" scrollButtons="auto" sx={{ flex: 1, minWidth: 0 }}>
              <Tab label="Stats" />
              <Tab label="Analytics" />
              <Tab label="Rallies" />
              <Tab label="Actions" />
              <Tab
                label={
                  <Badge badgeContent={setupAttentionCount} color="warning" sx={{ "& .MuiBadge-badge": { right: -10, top: -2 } }}>
                    Setup
                  </Badge>
                }
              />
            </Tabs>
          </Stack>

          {tab === 0 && (
            <StatsTab jobId={job.id} results={results} flatEvents={flatEvents} onSeek={seekTo} onJumpToAction={jumpToAction} />
          )}
          {tab === 1 && <AnalyticsTab results={results} onSeek={seekTo} />}
          {tab === 2 && (
            <RalliesTab
              jobId={job.id}
              rallies={results.rallies}
              flatEvents={flatEvents}
              currentTime={currentTime}
              onSeek={seekTo}
              onPlayAll={playAll}
            />
          )}
          {tab === 3 && (
            <ActionsTab
              results={results}
              flatEvents={flatEvents}
              currentTime={currentTime}
              onSeek={seekTo}
              onPlayAll={playAll}
              presetFilter={actionFilterPreset}
            />
          )}
          {tab === 4 && <SetupTab job={job} />}
        </Box>
      </Box>
    </Box>
  );
}
