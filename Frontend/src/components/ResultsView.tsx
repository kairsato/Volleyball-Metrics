import { useEffect, useMemo, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Badge from "@mui/material/Badge";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Typography from "@mui/material/Typography";
import StopCircleIcon from "@mui/icons-material/StopCircle";
import { api } from "../lib/api";
import type { Job, ResultsOut } from "../lib/types";
import { LoadingSpinner } from "./LoadingSpinner";
import { ActionsTab } from "./results/ActionsTab";
import { OverallTab } from "./results/OverallTab";
import { RalliesTab } from "./results/RalliesTab";
import { SetupTab } from "./results/SetupTab";
import { StatsTab } from "./results/StatsTab";
import type { FlatEvent } from "./results/types";

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
const TAB_NAMES = ["overall", "stats", "rallies", "actions", "setup"] as const;
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

  const videoRef = useRef<HTMLVideoElement>(null);
  const playlistIndexRef = useRef(0);
  const [playlist, setPlaylist] = useState<number[] | null>(null);
  const [playlistIndex, setPlaylistIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

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

    return () => {
      cancelled = true;
    };
  }, [job.id]);

  // Tracks playback position so the Rallies/Actions tabs can highlight
  // whatever's currently happening in the video - independent of the
  // playlist logic below, since this should work during normal playback too.
  // Depends on `results` because the <video> element itself doesn't exist
  // until the loading state clears - an empty dep array here would run
  // once against a still-null ref and never attach anything.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    function handleTimeUpdate() {
      setCurrentTime(videoRef.current?.currentTime ?? 0);
    }

    el.addEventListener("timeupdate", handleTimeUpdate);
    return () => el.removeEventListener("timeupdate", handleTimeUpdate);
  }, [results]);

  useEffect(() => {
    const el = videoRef.current;
    if (!el || playlist === null) return;

    function handleTimeUpdate() {
      const el = videoRef.current;
      if (!el) return;
      const clipStart = playlist![playlistIndexRef.current];
      if (el.currentTime - clipStart < CLIP_DURATION_S) return;

      const nextIndex = playlistIndexRef.current + 1;
      if (nextIndex < playlist!.length) {
        playlistIndexRef.current = nextIndex;
        setPlaylistIndex(nextIndex);
        el.currentTime = playlist![nextIndex];
      } else {
        setPlaylist(null);
        el.pause();
      }
    }

    el.addEventListener("timeupdate", handleTimeUpdate);
    return () => el.removeEventListener("timeupdate", handleTimeUpdate);
  }, [playlist]);

  // Drives the number badge on the Setup tab below - the same
  // needs_player_id/needs_scoring_review flags the video grid uses for its
  // own "Setup needed" chip (see VideoGrid.tsx), computed server-side so
  // this doesn't need its own fetch just to count unnamed players.
  const setupAttentionCount = (job.needs_player_id ? 1 : 0) + (job.needs_scoring_review ? 1 : 0);

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

  // `pause: true` is for jumping to a single reference frame (the player
  // preview's "jump to it") - the point there is to freeze on that exact
  // moment for a visual comparison, not to keep playing past it the way
  // seeking from Rallies/Actions ("Play", "Play all touches") should.
  function seekTo(timeS: number, pause = false) {
    setPlaylist(null);
    const el = videoRef.current;
    if (!el) return;
    el.currentTime = timeS;
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
      el.currentTime = timestamps[0];
      void el.play();
    }
  }

  if (error) return <Alert severity="error">{error}</Alert>;
  if (!results) return <LoadingSpinner />;

  return (
    <Box sx={{ height: PAGE_CONTENT_HEIGHT, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <Box sx={{ flex: 1, minHeight: 0, display: "flex", gap: 3, overflow: "hidden" }}>
        <Box sx={{ flex: "5 1 760px", minWidth: 520, minHeight: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <Box
            sx={{
              flex: 1,
              minHeight: 0,
              borderRadius: 2,
              border: 1,
              borderColor: "divider",
              bgcolor: "#000",
              overflow: "hidden",
            }}
          >
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <Box
              component="video"
              ref={videoRef}
              controls
              src={api.sourceVideoUrl(job.id)}
              sx={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
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

        <Box sx={{ flex: "2 1 380px", minWidth: 340, minHeight: 0, overflowY: "auto", pr: 0.5 }}>
          <Stack
            direction="row"
            sx={{
              alignItems: "center",
              mb: 2,
              borderBottom: 1,
              borderColor: "divider",
              position: "sticky",
              top: 0,
              bgcolor: "background.default",
              zIndex: 1,
            }}
          >
            <Tabs value={tab} onChange={(_, value) => changeTab(value)} sx={{ flex: 1, minWidth: 0 }}>
              <Tab label="Overall" />
              <Tab label="Stats" />
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

          {tab === 0 && <OverallTab results={results} flatEvents={flatEvents} onSeek={seekTo} onJumpToAction={jumpToAction} />}
          {tab === 1 && <StatsTab jobId={job.id} />}
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
