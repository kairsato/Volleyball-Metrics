import { useEffect, useMemo, useRef, useState } from "react";
import Alert from "@mui/material/Alert";
import Badge from "@mui/material/Badge";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Typography from "@mui/material/Typography";
import CastIcon from "@mui/icons-material/Cast";
import CheckIcon from "@mui/icons-material/Check";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SettingsIcon from "@mui/icons-material/Settings";
import StopCircleIcon from "@mui/icons-material/StopCircle";
import VolumeOffIcon from "@mui/icons-material/VolumeOff";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import { api } from "../lib/api";
import type { Job, ResultsOut } from "../lib/types";
import { LoadingSpinner } from "./LoadingSpinner";
import { ActionsTab } from "./results/ActionsTab";
import { AnalyticsTab } from "./results/AnalyticsTab";
import { RalliesTab } from "./results/RalliesTab";
import { SetupTab } from "./results/SetupTab";
import { StatsTab } from "./results/StatsTab";
import type { FlatEvent } from "./results/types";

const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];

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

  const [playbackRate, setPlaybackRate] = useState(1);
  const [qualities, setQualities] = useState<string[]>(["original"]);
  const [quality, setQuality] = useState("original");

  useEffect(() => {
    let cancelled = false;
    api
      .getQualities(job.id)
      .then((res) => !cancelled && setQualities(res.qualities))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [job.id]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  // Swapping `quality` changes the <video>'s src below, which reloads it
  // from scratch - capture playback position/state first and restore it
  // once the new source has loaded, so switching quality feels like a
  // seamless resolution change rather than restarting the video.
  function handleQualityChange(nextQuality: string) {
    const el = videoRef.current;
    if (!el) {
      setQuality(nextQuality);
      return;
    }
    const resumeTime = el.currentTime;
    const wasPlaying = !el.paused;
    setQuality(nextQuality);

    function handleLoaded() {
      if (!el) return;
      el.currentTime = resumeTime;
      el.playbackRate = playbackRate;
      if (wasPlaying) void el.play();
      el.removeEventListener("loadedmetadata", handleLoaded);
    }
    el.addEventListener("loadedmetadata", handleLoaded);
  }

  // Speed/quality live in a settings menu opened from a gear button in the
  // player's own custom bottom control bar (see the player markup below) -
  // right where a real player like YouTube's puts it, alongside play/pause,
  // volume, and fullscreen. The native <video controls> bar can't be
  // extended with a custom button at all, so this whole bottom bar is
  // hand-built rather than just the settings menu.
  const [settingsAnchor, setSettingsAnchor] = useState<HTMLElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  // Remote Playback API support (Chrome/Edge - Chromecast) - Firefox/Safari
  // don't implement it, so the cast button only renders when it's actually
  // going to do something real.
  const [castSupported] = useState(() => typeof window !== "undefined" && "remote" in document.createElement("video"));
  const playerContainerRef = useRef<HTMLDivElement>(null);

  function togglePlay() {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  }

  function toggleMute() {
    const el = videoRef.current;
    if (!el) return;
    el.muted = !el.muted;
    setIsMuted(el.muted);
  }

  function handleVolumeChange(value: number) {
    setVolume(value);
    setIsMuted(value === 0);
    const el = videoRef.current;
    if (!el) return;
    el.volume = value;
    el.muted = value === 0;
  }

  async function handleCast() {
    const el = videoRef.current;
    if (!el?.remote) return;
    try {
      await el.remote.prompt();
    } catch {
      // User dismissed the device picker, or none were found - nothing to do.
    }
  }

  function toggleFullscreen() {
    const container = playerContainerRef.current;
    if (!container) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void container.requestFullscreen();
  }

  function formatTime(seconds: number): string {
    if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  useEffect(() => {
    function handleFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === playerContainerRef.current);
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

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

  // Drives the custom control bar's play/pause icon and duration display -
  // same "depends on results so the ref actually exists yet" reasoning as
  // the timeupdate listener above.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    function handlePlay() {
      setIsPlaying(true);
    }
    function handlePause() {
      setIsPlaying(false);
    }
    function handleDurationChange() {
      setDuration(videoRef.current?.duration || 0);
    }

    el.addEventListener("play", handlePlay);
    el.addEventListener("pause", handlePause);
    el.addEventListener("loadedmetadata", handleDurationChange);
    el.addEventListener("durationchange", handleDurationChange);
    return () => {
      el.removeEventListener("play", handlePlay);
      el.removeEventListener("pause", handlePause);
      el.removeEventListener("loadedmetadata", handleDurationChange);
      el.removeEventListener("durationchange", handleDurationChange);
    };
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
            ref={playerContainerRef}
            sx={{
              position: "relative",
              flex: { xs: "0 0 auto", sm: 1 },
              minHeight: 0,
              aspectRatio: { xs: "16 / 9", sm: "auto" },
              borderRadius: 2,
              border: 1,
              borderColor: "divider",
              bgcolor: "#000",
              overflow: "hidden",
              "&:hover .video-controls-bar": { opacity: 1 },
            }}
          >
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <Box
              component="video"
              ref={videoRef}
              src={api.sourceVideoUrl(job.id, quality)}
              onClick={togglePlay}
              disablePictureInPicture
              sx={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
                display: "block",
                cursor: "pointer",
                // Chrome auto-adds its own cast overlay button (top-left)
                // to any video element with remote-playback capability -
                // `disableRemotePlayback` would suppress that but also
                // breaks our own Cast button's remote.prompt() call, so
                // this hides just the browser's auto overlay instead,
                // leaving the Remote Playback API itself intact.
                "&::-webkit-media-controls-overlay-cast-button": { display: "none" },
              }}
            />

            {/* Custom control bar (native <video controls> can't have a
                settings/quality button added to it) - modeled on a
                standard player like YouTube's: a scrubber above a row of
                play/pause, volume, time, settings, and fullscreen. Always
                shown while paused; fades in on hover while playing so it
                doesn't sit over the video the whole time. */}
            <Box
              className="video-controls-bar"
              sx={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: 0,
                background: "linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0))",
                opacity: isPlaying ? 0 : 1,
                transition: "opacity 0.15s ease",
                px: 1.5,
                pb: 0.5,
                pt: 3,
              }}
            >
              <Slider
                value={Math.min(currentTime, duration || 0)}
                min={0}
                max={duration || 0}
                onChange={(_, value) => {
                  const el = videoRef.current;
                  if (el) el.currentTime = value as number;
                }}
                sx={{
                  color: "primary.main",
                  height: 4,
                  display: "block",
                  padding: "8px 0",
                  "& .MuiSlider-thumb": {
                    width: 12,
                    height: 12,
                    opacity: 0,
                    transition: "opacity 0.15s ease",
                  },
                  "&:hover .MuiSlider-thumb, & .MuiSlider-thumb.Mui-active": { opacity: 1 },
                  "& .MuiSlider-rail": { opacity: 0.35, height: 4 },
                  "& .MuiSlider-track": { height: 4, border: "none" },
                }}
              />
              <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
                <IconButton aria-label={isPlaying ? "Pause" : "Play"} onClick={togglePlay} sx={{ color: "#fff" }}>
                  {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                </IconButton>
                <IconButton aria-label={isMuted || volume === 0 ? "Unmute" : "Mute"} onClick={toggleMute} sx={{ color: "#fff" }}>
                  {isMuted || volume === 0 ? <VolumeOffIcon /> : <VolumeUpIcon />}
                </IconButton>
                <Slider
                  value={isMuted ? 0 : volume}
                  min={0}
                  max={1}
                  step={0.01}
                  onChange={(_, value) => handleVolumeChange(value as number)}
                  sx={{
                    width: 72,
                    color: "#fff",
                    height: 4,
                    "& .MuiSlider-thumb": { width: 10, height: 10 },
                    "& .MuiSlider-rail": { opacity: 0.35, height: 4 },
                    "& .MuiSlider-track": { height: 4, border: "none" },
                  }}
                />
                <Typography variant="caption" sx={{ color: "#fff", minWidth: 88, pl: 0.5 }}>
                  {formatTime(currentTime)} / {formatTime(duration)}
                </Typography>
                <Box sx={{ flex: 1 }} />
                <IconButton
                  aria-label="Playback settings"
                  onClick={(event) => setSettingsAnchor(event.currentTarget)}
                  sx={{ color: "#fff" }}
                >
                  <SettingsIcon />
                </IconButton>
                {castSupported && (
                  <IconButton aria-label="Cast" onClick={() => void handleCast()} sx={{ color: "#fff" }}>
                    <CastIcon />
                  </IconButton>
                )}
                <IconButton
                  aria-label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
                  onClick={toggleFullscreen}
                  sx={{ color: "#fff" }}
                >
                  {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
                </IconButton>
              </Stack>
            </Box>

            <Menu anchorEl={settingsAnchor} open={Boolean(settingsAnchor)} onClose={() => setSettingsAnchor(null)}>
              <Typography variant="caption" color="text.secondary" sx={{ px: 2, py: 0.5, display: "block" }}>
                Speed
              </Typography>
              {PLAYBACK_RATES.map((rate) => (
                <MenuItem
                  key={rate}
                  selected={rate === playbackRate}
                  onClick={() => {
                    setPlaybackRate(rate);
                    setSettingsAnchor(null);
                  }}
                >
                  <ListItemIcon>{rate === playbackRate ? <CheckIcon fontSize="small" /> : null}</ListItemIcon>
                  {rate}x
                </MenuItem>
              ))}
              {qualities.length > 1 && [
                <Divider key="divider" />,
                <Typography key="label" variant="caption" color="text.secondary" sx={{ px: 2, py: 0.5, display: "block" }}>
                  Quality
                </Typography>,
                ...qualities.map((q) => (
                  <MenuItem
                    key={q}
                    selected={q === quality}
                    onClick={() => {
                      handleQualityChange(q);
                      setSettingsAnchor(null);
                    }}
                  >
                    <ListItemIcon>{q === quality ? <CheckIcon fontSize="small" /> : null}</ListItemIcon>
                    {q === "original" ? "Original" : q}
                  </MenuItem>
                )),
              ]}
            </Menu>
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
