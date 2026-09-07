import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, RefObject } from "react";
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import CheckIcon from "@mui/icons-material/Check";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import HighQualityIcon from "@mui/icons-material/HighQuality";
import LayersIcon from "@mui/icons-material/Layers";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import SettingsIcon from "@mui/icons-material/Settings";
import SpeedIcon from "@mui/icons-material/Speed";
import VolumeOffIcon from "@mui/icons-material/VolumeOff";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import type { BallTrajectoryPoint, Game, PlayerTrajectoryFrame, Rally, RallyWinner } from "../lib/types";
import { findIndexAtOrBefore } from "../lib/timeSeries";
import { BallMinimap } from "./BallMinimap";
import { BallTrackingOverlay } from "./BallTrackingOverlay";
import { BallTrajectoryOverlay } from "./BallTrajectoryOverlay";
import { PlayerTrackingOverlay } from "./PlayerTrackingOverlay";
import { computeCurrentScore, ScoreOverlay } from "./ScoreOverlay";

const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];
// How far the Left/Right arrow keys seek - see the keydown effect below.
const ARROW_SEEK_STEP_S = 10;

// The settings menu is two layers deep, YouTube-style: "main" lists each
// category with its current value and a chevron, clicking one drills into
// that category's own options (with a back arrow to return) instead of
// showing everything flat in one long list.
type SettingsView = "main" | "speed" | "quality" | "annotations";

interface AnnotationSettings {
  ballTracking: boolean;
  // The current hit-to-hit flight's real recorded arc (see
  // BallTrajectoryOverlay.tsx) - a curve rather than ballTracking's single
  // point, independent of it (both can be on together).
  ballArc: boolean;
  playerTracking: boolean;
  score: boolean;
  minimap: boolean;
  // Shows each player's raw bounding box plus a name+stable_id label -
  // independent of playerTracking's own simplified name-only view (both
  // can be on together). Meant for checking detection/identification
  // quality, not for watching the game.
  debug: boolean;
}

const DEFAULT_ANNOTATIONS: AnnotationSettings = {
  ballTracking: false,
  ballArc: false,
  playerTracking: false,
  score: false,
  // On by default - this one already existed (unconditionally) before the
  // other four were added, so this keeps that existing behavior rather
  // than suddenly hiding something that was always there.
  minimap: true,
  debug: false,
};

interface PersistedPlayerSettings {
  playbackRate: number;
  volume: number;
  isMuted: boolean;
  annotations: AnnotationSettings;
}

const DEFAULT_PLAYER_SETTINGS: PersistedPlayerSettings = {
  playbackRate: 1,
  volume: 1,
  isMuted: false,
  annotations: DEFAULT_ANNOTATIONS,
};

// Playback preferences - speed, volume, and which annotations are on -
// carry over between videos and reloads, same "set it once" idea as the
// theme mode toggle (see themeMode.tsx). Quality is deliberately not
// included here even though it's also a setting: it's owned by each page
// (Results/Scoring Determination/Warmup all keep their own `quality`
// state), not this component, and the available renditions differ per
// job, so there's no single "last quality" that's guaranteed valid next
// time regardless.
const PLAYER_SETTINGS_KEY = "vva-player-settings";

function readPersistedSettings(): PersistedPlayerSettings {
  try {
    const stored = window.localStorage.getItem(PLAYER_SETTINGS_KEY);
    if (!stored) return DEFAULT_PLAYER_SETTINGS;
    const parsed = JSON.parse(stored);
    return {
      playbackRate: typeof parsed.playbackRate === "number" ? parsed.playbackRate : DEFAULT_PLAYER_SETTINGS.playbackRate,
      volume: typeof parsed.volume === "number" ? parsed.volume : DEFAULT_PLAYER_SETTINGS.volume,
      isMuted: typeof parsed.isMuted === "boolean" ? parsed.isMuted : DEFAULT_PLAYER_SETTINGS.isMuted,
      annotations: { ...DEFAULT_ANNOTATIONS, ...parsed.annotations },
    };
  } catch {
    // Malformed JSON or localStorage unavailable (private browsing, etc.) -
    // fall back to defaults rather than failing to load the player.
    return DEFAULT_PLAYER_SETTINGS;
  }
}

// Every page that plays the source video (Results, Scoring Determination,
// Warmup Period) restricts it to some [boundStartS, boundEndS) window of
// the raw file - the whole video for Warmup's own unrestricted picker, or
// a confirmed warmup range everywhere else (see Backend/API/warmup.py).
// These two are the one place that conversion happens, so the player's
// scrubber, external seekTo/playAll callers, and rally/action timestamps
// all agree on what "0:00" means without each page re-deriving it.
export function toBoundedAbsolute(relativeS: number, boundStartS: number, boundEndS: number): number {
  return Math.min(Math.max(relativeS + boundStartS, boundStartS), boundEndS);
}

export function toBoundedRelative(absoluteS: number, boundStartS: number, boundEndS: number): number {
  return Math.max(0, Math.min(absoluteS, boundEndS) - boundStartS);
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

interface VideoPlayerProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  src: string;
  // The seekable/playable window, in absolute video time - defaults to the
  // whole file. Time 0:00 in this player's own UI (the scrubber, the
  // elapsed/duration readout) always means boundStartS, not the raw file's
  // start.
  boundStartS?: number;
  boundEndS?: number;
  // Fired on every timeupdate with the current position relative to
  // boundStartS - lets a caller that needs to know playback position for
  // its own purposes (e.g. ResultsView highlighting the active rally) stay
  // in sync without attaching a second listener to the same element.
  onTimeUpdate?: (relativeTimeS: number) => void;
  // Omit entirely on pages with only one rendition (Warmup, Scoring
  // Determination) - the settings menu then just offers playback speed.
  qualities?: string[];
  quality?: string;
  onQualityChange?: (quality: string) => void;
  // Omit to leave out the Minimap/Ball tracking annotations entirely (see
  // BallMinimap.tsx/BallTrackingOverlay.tsx) - only Results' main player
  // passes this today. Ball tracking only needs points/frameW&H; Minimap
  // additionally needs courtLengthM/courtWidthM, so it stays unavailable
  // (and hidden from the Annotations menu) without those even if points
  // are present.
  ballTrajectory?: BallTrajectoryPoint[];
  courtLengthM?: number;
  courtWidthM?: number;
  // The fixed resolution every annotation's coordinates are measured in
  // (see BallTrajectory.frame_w/frame_h's own doc comment for why this is
  // NOT the same thing as the <video> element's own decoded videoWidth/
  // videoHeight, which changes if the viewer picks a lower-bitrate
  // playback quality). Every overlay below sizes its SVG viewBox from
  // this, falling back to the live element's own decoded size only for the
  // brief window before it's known (or a job with no detection at all, in
  // which case the overlays needing it are hidden anyway). Callers should
  // pass whichever of ballTrajectory/playerTrajectory's own frame_w/frame_h
  // is available - they're the same source video, so either works.
  frameW?: number | null;
  frameH?: number | null;
  // Every player-touch timestamp (warmup-relative, same basis as
  // currentTime), sorted ascending - the Ball Trajectory annotation's
  // segment boundaries (see BallTrajectoryOverlay.tsx: the current
  // hit-to-hit flight is bounded by the touches immediately before/after
  // now). Omit to leave that annotation out entirely, same as ballTrajectory
  // itself being omitted leaves Ball tracking out.
  hitTimestamps?: number[];
  // Omit to leave out the Player tracking annotation - only Results' main
  // player passes this today.
  playerTrajectory?: PlayerTrajectoryFrame[];
  // Omit to leave out the Score annotation - both are required together
  // (rallies gives each rally_index's timing, scoreRallies its winner; see
  // ScoreOverlay's own doc comment for why they're kept separate rather
  // than pre-joined by the caller).
  rallies?: Rally[];
  scoreRallies?: RallyWinner[];
  // Set boundaries, for the scrubber's always-visible chapter dividers
  // (see hasRallySegments below) - a game's own start_rally_index is
  // looked up against `rallies` for its actual start_time_s, same as
  // scoreRallies is kept id-only rather than pre-joined by the caller.
  games?: Game[];
  teamXName?: string;
  teamYName?: string;
  // Left/Right arrow keys skip ARROW_SEEK_STEP_S back/forward, document-wide
  // rather than requiring the player to have focus first (see the effect
  // below). Off by default and opt-in per caller, not just "only when this
  // player is the one on screen" - VideoPlayer is also used on pages with
  // their own arrow-key-sensitive controls (Warmup's trim range, Scoring
  // Determination's rally list), where a global seek-on-arrow-key would be
  // an unwanted side effect rather than the convenience it is here. Only
  // Results' main player passes this today.
  enableArrowKeySeek?: boolean;
}

// The one video player UI used everywhere the source video is actually
// watched (Results' main player, Scoring Determination's review player,
// Warmup Period's range picker) - a custom bottom control bar (native
// <video controls> can't have a settings/quality button added to it)
// modeled on a standard player like YouTube's: a scrubber above play/
// pause, volume, time, settings, and fullscreen. Always shown while
// paused; fades in on hover while playing so it doesn't sit over the video
// the whole time.
//
// `videoRef` is owned by the caller (not created here) - callers that need
// to seek externally (jumping to a rally's timestamp, a "play all"
// playlist, Warmup's "use current time" buttons) just set
// videoRef.current.currentTime directly, exactly as before this was
// extracted into its own component; this component's own listeners pick
// that up on the next timeupdate like any other seek.
export function VideoPlayer({
  videoRef,
  src,
  boundStartS = 0,
  boundEndS = Infinity,
  onTimeUpdate,
  qualities,
  quality,
  onQualityChange,
  ballTrajectory,
  courtLengthM,
  courtWidthM,
  frameW,
  frameH,
  hitTimestamps,
  playerTrajectory,
  rallies,
  scoreRallies,
  games,
  teamXName,
  teamYName,
  enableArrowKeySeek = false,
}: VideoPlayerProps) {
  const playerContainerRef = useRef<HTMLDivElement>(null);
  const [settingsAnchor, setSettingsAnchor] = useState<HTMLElement | null>(null);
  const [settingsView, setSettingsView] = useState<SettingsView>("main");
  const [playbackRate, setPlaybackRate] = useState(() => readPersistedSettings().playbackRate);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(() => readPersistedSettings().isMuted);
  const [volume, setVolume] = useState(() => readPersistedSettings().volume);
  const [isFullscreen, setIsFullscreen] = useState(false);
  // Drives the score-preview tooltip's reveal - hidden until the timeline
  // itself is hovered (see the Slider's wrapping Box below), same "don't
  // sit on top of the video uninvited" idea as the whole controls bar
  // fading in on hover.
  const [scrubberHovered, setScrubberHovered] = useState(false);
  // Where along the timeline the pointer currently sits, in relative
  // seconds - null whenever the pointer isn't over the scrubber at all
  // (as opposed to 0, a valid hover position at the very start).
  const [hoverTimeS, setHoverTimeS] = useState<number | null>(null);
  // The <video> element's own intrinsic (decoded) resolution - populated
  // once metadata loads, alongside `duration` below. Used for the video's
  // own layout math ONLY; deliberately not what the annotation overlays
  // size themselves from (see annotationSize below) - this changes if the
  // viewer switches playback quality (see api.ts's sourceVideoUrl), which
  // has nothing to do with the fixed resolution BallDetection.ballDetection
  // actually measured every tracked coordinate in.
  const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
  // What every annotation overlay's SVG viewBox is actually sized from -
  // frameW/frameH (the fixed resolution ball_trajectory.json/
  // player_positions.json's coordinates were measured against, independent
  // of playback quality) when known, falling back to the live decoded
  // videoSize only for the brief window before a job's trajectory data has
  // loaded. Getting this wrong (using videoSize directly, as this used to)
  // is exactly what made every annotation land in the wrong place the
  // moment someone switched to a lower-bitrate quality: a 1920x1080 source
  // watched as its 720p rendition decodes at 1280x720, but the tracked
  // pixel coordinates never change, so an SVG viewBox sized from the
  // rendition's own resolution scales every point by the wrong factor.
  const annotationSize = frameW && frameH ? { width: frameW, height: frameH } : videoSize;
  const [annotations, setAnnotations] = useState<AnnotationSettings>(() => readPersistedSettings().annotations);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = playbackRate;
  }, [playbackRate, videoRef]);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        PLAYER_SETTINGS_KEY,
        JSON.stringify({ playbackRate, volume, isMuted, annotations }),
      );
    } catch {
      // localStorage unavailable (private browsing, etc.) - preferences
      // still apply for this session, they just won't carry over.
    }
  }, [playbackRate, volume, isMuted, annotations]);

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;

    function handleTimeUpdate() {
      const el = videoRef.current;
      if (!el) return;
      // A stray native seek (e.g. arrow keys while the element has focus)
      // could otherwise land outside the bound range - snap it back rather
      // than letting playback wander into excluded footage.
      if (el.currentTime > boundEndS) el.currentTime = boundEndS;
      else if (el.currentTime < boundStartS) el.currentTime = boundStartS;
      const relative = toBoundedRelative(el.currentTime, boundStartS, boundEndS);
      setCurrentTime(relative);
      onTimeUpdate?.(relative);
    }
    function handlePlay() {
      setIsPlaying(true);
    }
    function handlePause() {
      setIsPlaying(false);
    }
    function handleDurationChange() {
      const el = videoRef.current;
      if (!el) return;
      setDuration(toBoundedRelative(el.duration || 0, boundStartS, boundEndS));
      setVideoSize({ width: el.videoWidth, height: el.videoHeight });
      // Jumps playback to boundStartS the first time metadata is available,
      // so a video with a warmup period starts at its real beginning
      // instead of the raw file's frame 0.
      if (boundStartS > 0 && el.currentTime < boundStartS) el.currentTime = boundStartS;
    }

    el.addEventListener("timeupdate", handleTimeUpdate);
    el.addEventListener("play", handlePlay);
    el.addEventListener("pause", handlePause);
    el.addEventListener("loadedmetadata", handleDurationChange);
    el.addEventListener("durationchange", handleDurationChange);
    return () => {
      el.removeEventListener("timeupdate", handleTimeUpdate);
      el.removeEventListener("play", handlePlay);
      el.removeEventListener("pause", handlePause);
      el.removeEventListener("loadedmetadata", handleDurationChange);
      el.removeEventListener("durationchange", handleDurationChange);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, boundStartS, boundEndS]);

  // `timeupdate` alone caps how often `currentTime` (and so this) updates
  // at whatever rate the browser fires it - the spec allows throttling it
  // to as little as ~4/sec, which is fine for anything that just snaps to
  // the current sample (the ball dot, the minimap) but not for
  // PlayerTrackingOverlay's box-gliding: advancing currentTime in ~250ms
  // steps makes the glide itself look laggy no matter how good the
  // interpolation math is. Driving it from requestAnimationFrame instead,
  // only while playing, gives it a real per-frame update rate;
  // `timeupdate` above stays wired up for the paused case (programmatic
  // seeks - see this component's own doc comment).
  useEffect(() => {
    if (!isPlaying) return;
    let raf: number;
    function tick() {
      const el = videoRef.current;
      if (el) {
        const relative = toBoundedRelative(el.currentTime, boundStartS, boundEndS);
        setCurrentTime(relative);
        onTimeUpdate?.(relative);
      }
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, boundStartS, boundEndS, onTimeUpdate, videoRef]);

  // Left/Right arrow keys skip ARROW_SEEK_STEP_S back/forward, document-wide
  // (not just while the player itself has focus) - the same convention
  // YouTube/Netflix use, since requiring a click on the player first before
  // arrow keys do anything is an easy thing to not discover. Skipped
  // whenever the focused element would itself use arrow keys for something
  // else (typing in a field, nudging a slider, moving between tabs) so this
  // doesn't fight normal keyboard navigation elsewhere on the page. Reads
  // el.currentTime directly rather than the `currentTime` React state so
  // this effect doesn't need to re-subscribe on every rAF-driven state
  // update while playing (see the effect above).
  useEffect(() => {
    if (!enableArrowKeySeek) return;

    function isArrowKeyTarget(el: Element | null): boolean {
      if (!el) return false;
      if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return true;
      if ((el as HTMLElement).isContentEditable) return true;
      const role = el.getAttribute("role");
      return role !== null && ["slider", "tab", "menuitem", "menuitemradio", "option", "spinbutton", "combobox"].includes(role);
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (isArrowKeyTarget(document.activeElement)) return;
      const el = videoRef.current;
      if (!el) return;
      event.preventDefault();
      const delta = event.key === "ArrowLeft" ? -ARROW_SEEK_STEP_S : ARROW_SEEK_STEP_S;
      const relative = toBoundedRelative(el.currentTime, boundStartS, boundEndS);
      const nextRelative = Math.min(Math.max(relative + delta, 0), duration || 0);
      el.currentTime = toBoundedAbsolute(nextRelative, boundStartS, boundEndS);
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [enableArrowKeySeek, boundStartS, boundEndS, duration, videoRef]);

  useEffect(() => {
    function handleFullscreenChange() {
      setIsFullscreen(document.fullscreenElement === playerContainerRef.current);
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

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

  function toggleAnnotation(key: keyof AnnotationSettings) {
    setAnnotations((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function toggleFullscreen() {
    const container = playerContainerRef.current;
    if (!container) return;
    if (document.fullscreenElement) void document.exitFullscreen();
    else void container.requestFullscreen();
  }

  // Swapping `quality` changes the <video>'s src, which reloads it from
  // scratch - capture playback position/state first and restore it once
  // the new source has loaded, so switching quality feels like a seamless
  // resolution change rather than restarting the video.
  function handleQualityChange(nextQuality: string) {
    const el = videoRef.current;
    if (!el || !onQualityChange) {
      onQualityChange?.(nextQuality);
      return;
    }
    const resumeTime = el.currentTime;
    const wasPlaying = !el.paused;
    onQualityChange(nextQuality);

    function handleLoaded() {
      if (!el) return;
      el.currentTime = resumeTime;
      el.playbackRate = playbackRate;
      if (wasPlaying) void el.play();
      el.removeEventListener("loadedmetadata", handleLoaded);
    }
    el.addEventListener("loadedmetadata", handleLoaded);
  }

  const hasQualityOptions = Boolean(qualities && qualities.length > 1);
  const hasBallTracking = Boolean(ballTrajectory && ballTrajectory.length > 0);
  // Needs hit timing to find the current segment's boundaries (see
  // BallTrajectoryOverlay's own doc comment) on top of everything Ball
  // tracking already needs.
  const hasBallArc = Boolean(ballTrajectory && ballTrajectory.length > 0 && hitTimestamps && hitTimestamps.length > 0);
  const hasMinimap = Boolean(ballTrajectory && ballTrajectory.length > 0 && courtLengthM && courtWidthM);
  const hasPlayerTracking = Boolean(playerTrajectory && playerTrajectory.length > 0);
  const hasScore = Boolean(rallies && scoreRallies);
  const hasAnyAnnotation = hasBallTracking || hasBallArc || hasMinimap || hasPlayerTracking || hasScore;

  // BallMinimap's small corner court-diagram dot still just snaps to "the
  // point/frame at or before now" - passing a stable array reference
  // (useMemo, only changing when new data loads) plus a plain index lets
  // it, wrapped in React.memo, skip re-rendering entirely on ticks that
  // don't cross into a new point. BallTrackingOverlay (the on-video ball
  // marker) used to work the same way, but its samples are sparse enough
  // (BALL_TRAJECTORY_STRIDE) relative to how far the ball moves in that
  // time that snapping read as teleporting - it now interpolates between
  // samples itself, the same way PlayerTrackingOverlay does, so it takes
  // currentTime directly instead of a pre-resolved index.
  const ballPointsWithPixel = useMemo(
    () => ballTrajectory?.filter((p) => p.px !== null && p.py !== null) ?? [],
    [ballTrajectory],
  );
  const ballMinimapIndex = ballTrajectory ? Math.max(0, findIndexAtOrBefore(ballTrajectory, currentTime)) : -1;
  // Same "resolve the index here so a memo'd child can skip re-rendering"
  // reasoning as ballMinimapIndex above - the minimap draws every tracked
  // player's court position alongside the ball, and player samples are
  // sparse enough (PLAYER_TRAJECTORY_STRIDE) that this only actually
  // changes a couple of times a second.
  const minimapPlayerFrame = playerTrajectory
    ? playerTrajectory[Math.max(0, findIndexAtOrBefore(playerTrajectory, currentTime))]
    : undefined;

  // Same data as the Score annotation, reused for the timeline's hover
  // score preview - both features exist only where a video has rally
  // timing AND a determined winner per rally.
  const hasRallySegments = Boolean(rallies && rallies.length > 0 && scoreRallies && duration > 0);

  // Chapter-divider positions: one per game TRANSITION, not per game - the
  // first game has no prior game to be divided from (what comes before it
  // is just pre-match footage, already outside the bound range), so it's
  // dropped rather than marked. Position is each remaining game's
  // start_time_s, i.e. its first rally's own start_time_s (games only
  // carry rally-index ranges, not timing - see the Game type's own doc).
  // Skipping is deliberate here, not filtered later - a game whose start
  // rally isn't in `rallies` (or whose start lands at/past the very edges
  // of the timeline) just adds no divider rather than one at the wrong
  // spot.
  const gameStartTimes = (games ?? [])
    .slice()
    .sort((a, b) => a.game_index - b.game_index)
    .slice(1)
    .map((g) => rallies?.find((r) => r.rally_index === g.start_rally_index)?.start_time_s)
    .filter((t): t is number => t !== undefined && t > 0 && t < duration);

  // A YouTube-retention-graph-style line above the scrubber, but plotting
  // "how contested was this point" instead of "how much was this
  // rewatched" - rally duration is the one intensity signal every rally
  // already carries with no further computation, and a long rally
  // (extended back-and-forth, both sides fighting for the point) is a
  // reasonable stand-in for "this moment mattered more" than a rally that
  // ended in a touch or two. Normalized against this video's own longest
  // rally, with a floor so even the shortest rally still traces a visible
  // line rather than flattening to the very bottom.
  //
  // One polyline per game rather than one continuous line for the whole
  // match, so a game boundary shows up as an actual break in the line
  // itself - no divider drawn across it, which would either paint a solid
  // bar over the (largely transparent) graph area or need to match every
  // background it might sit over.
  const GRAPH_HEIGHT = 28;
  const GRAPH_MIN_HEIGHT_FRAC = 0.12;
  // Thicker than the old 4px bar - easier to grab and to actually see the
  // game dividers cut into.
  const BAR_HEIGHT = 8;
  const maxRallyDuration = rallies && rallies.length > 0 ? Math.max(...rallies.map((r) => r.duration_s)) : 0;
  const rallyGroupsByGame: Rally[][] =
    games && games.length > 0
      ? games
          .slice()
          .sort((a, b) => a.game_index - b.game_index)
          .map((g) => (rallies ?? []).filter((r) => r.rally_index >= g.start_rally_index && r.rally_index <= g.end_rally_index))
      : rallies
        ? [rallies]
        : [];
  const graphSegments =
    duration > 0 && maxRallyDuration > 0
      ? rallyGroupsByGame
          .map((group) =>
            group
              .slice()
              .sort((a, b) => a.start_time_s - b.start_time_s)
              .map((r) => {
                const importance =
                  GRAPH_MIN_HEIGHT_FRAC + (r.duration_s / maxRallyDuration) * (1 - GRAPH_MIN_HEIGHT_FRAC);
                const x = (((r.start_time_s + r.end_time_s) / 2) / duration) * 100;
                const y = GRAPH_HEIGHT * (1 - importance);
                return `${x},${y}`;
              })
              .join(" "),
          )
          .filter((points) => points.length > 0)
      : [];

  // The score as of wherever the pointer is currently hovering, for the
  // preview tooltip - same computeCurrentScore ScoreOverlay uses for "the
  // score right now" during actual playback, just fed a hovered time
  // instead of the playing one. Falls back to 0-0 in game 1 rather than
  // hiding the tooltip when hovering before any rally has concluded yet
  // (e.g. right at the very start of the video).
  const hoverScore =
    hasRallySegments && hoverTimeS !== null
      ? (computeCurrentScore(rallies!, scoreRallies!, hoverTimeS) ?? { gameIndex: 0, x: 0, y: 0 })
      : null;

  function handleScrubberMouseMove(event: ReactMouseEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const frac = rect.width > 0 ? Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)) : 0;
    setHoverTimeS(frac * (duration || 0));
  }

  return (
    <Box
      ref={playerContainerRef}
      sx={{
        position: "relative",
        width: "100%",
        height: "100%",
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
        src={src}
        onClick={togglePlay}
        disablePictureInPicture
        // Chrome auto-adds its own cast overlay button (top-left) to any
        // video element with remote-playback capability - this suppresses
        // it outright. There's no custom Cast button of our own to
        // preserve a reason not to (there was one; it's gone - not worth
        // the Remote Playback API surface for what it did).
        disableRemotePlayback
        sx={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
          display: "block",
          cursor: "pointer",
        }}
      />

      {annotations.minimap && ballTrajectory && ballTrajectory.length > 0 && courtLengthM && courtWidthM && (
        <BallMinimap
          points={ballTrajectory}
          courtLengthM={courtLengthM}
          courtWidthM={courtWidthM}
          currentIndex={ballMinimapIndex}
          playerFrame={minimapPlayerFrame}
        />
      )}
      {annotations.ballTracking && ballPointsWithPixel.length > 0 && annotationSize.width > 0 && (
        <BallTrackingOverlay
          points={ballPointsWithPixel}
          currentTimeS={currentTime}
          videoWidth={annotationSize.width}
          videoHeight={annotationSize.height}
        />
      )}
      {annotations.ballArc &&
        hitTimestamps &&
        hitTimestamps.length > 0 &&
        ballPointsWithPixel.length > 0 &&
        annotationSize.width > 0 && (
          <BallTrajectoryOverlay
            points={ballPointsWithPixel}
            hitTimestamps={hitTimestamps}
            currentTimeS={currentTime}
            videoWidth={annotationSize.width}
            videoHeight={annotationSize.height}
          />
        )}
      {(annotations.playerTracking || annotations.debug) && playerTrajectory && annotationSize.width > 0 && (
        <PlayerTrackingOverlay
          frames={playerTrajectory}
          currentTimeS={currentTime}
          videoWidth={annotationSize.width}
          videoHeight={annotationSize.height}
          showNames={annotations.playerTracking}
          showDebug={annotations.debug}
        />
      )}
      {annotations.score && rallies && scoreRallies && (
        <ScoreOverlay
          rallies={rallies}
          scoreRallies={scoreRallies}
          teamXName={teamXName ?? "Team X"}
          teamYName={teamYName ?? "Team Y"}
          currentTimeS={currentTime}
        />
      )}

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
        <Box
          sx={{ position: "relative" }}
          onMouseEnter={() => setScrubberHovered(true)}
          onMouseLeave={() => {
            setScrubberHovered(false);
            setHoverTimeS(null);
          }}
          onMouseMove={handleScrubberMouseMove}
        >
          {/* Importance graph - revealed on hover only, same as the score
              tooltip below, rather than sitting on the video uninvited the
              way YouTube's own always-on retention graph does. Plots rally
              duration, not rewatch rate - see graphSegments above for why
              that's the stand-in for "this moment mattered more". One
              polyline per game so a game boundary is an actual break in
              the line, not a divider drawn across it. */}
          {graphSegments.length > 0 && (
            <Box
              component="svg"
              viewBox={`0 0 100 ${GRAPH_HEIGHT}`}
              preserveAspectRatio="none"
              sx={{
                display: "block",
                width: "100%",
                height: GRAPH_HEIGHT,
                pointerEvents: "none",
                opacity: scrubberHovered ? 1 : 0,
                transition: "opacity 0.15s ease",
              }}
            >
              {graphSegments.map((points, i) => (
                <polyline
                  key={i}
                  points={points}
                  fill="none"
                  stroke="rgba(255,255,255,0.85)"
                  strokeWidth={1.5}
                  vectorEffect="non-scaling-stroke"
                />
              ))}
            </Box>
          )}

          {hasRallySegments && (
            <>
              {/* Game-boundary markers, always visible - YouTube's own
                  chapter-divider idiom: a thin cut of background colour
                  punched through the bar rather than a highlight drawn
                  over it. One per game TRANSITION (see gameStartTimes
                  above), not per rally - a single match can have dozens
                  of rallies, which would turn the bar into a solid dashed
                  line rather than a handful of meaningful dividers. */}
              <Box
                sx={{
                  position: "absolute",
                  left: 0,
                  right: 0,
                  top: GRAPH_HEIGHT + 8,
                  height: BAR_HEIGHT,
                  zIndex: 1,
                  pointerEvents: "none",
                }}
              >
                {gameStartTimes.map((t) => (
                  <Box
                    key={t}
                    sx={{
                      position: "absolute",
                      left: `${(t / duration) * 100}%`,
                      top: 0,
                      bottom: 0,
                      width: "3px",
                      bgcolor: "#000",
                    }}
                  />
                ))}
              </Box>

              {/* Score-at-hover tooltip, following the pointer along the
                  bar - "what was the score around here", the timeline
                  equivalent of ScoreOverlay's own corner scoreboard. */}
              {scrubberHovered && hoverScore && (
                <Box
                  sx={{
                    position: "absolute",
                    left: `${((hoverTimeS ?? 0) / (duration || 1)) * 100}%`,
                    bottom: "100%",
                    mb: 0.5,
                    transform: "translateX(-50%)",
                    bgcolor: "rgba(0,0,0,0.85)",
                    color: "#fff",
                    borderRadius: 1,
                    px: 1,
                    py: 0.5,
                    whiteSpace: "nowrap",
                    pointerEvents: "none",
                    zIndex: 2,
                  }}
                >
                  <Typography variant="caption" sx={{ display: "block", opacity: 0.75, lineHeight: 1.2 }}>
                    {formatTime(hoverTimeS ?? 0)} · Set {hoverScore.gameIndex + 1}
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600, lineHeight: 1.2 }}>
                    {teamXName ?? "Team X"} {hoverScore.x} - {hoverScore.y} {teamYName ?? "Team Y"}
                  </Typography>
                </Box>
              )}
            </>
          )}

          <Slider
            value={Math.min(currentTime, duration || 0)}
            min={0}
            max={duration || 0}
            onChange={(_, value) => {
              const el = videoRef.current;
              if (el) el.currentTime = toBoundedAbsolute(value as number, boundStartS, boundEndS);
            }}
            sx={{
              color: "primary.main",
              height: BAR_HEIGHT,
              display: "block",
              padding: "8px 0",
              "& .MuiSlider-thumb": {
                width: 14,
                height: 14,
                opacity: 0,
                transition: "opacity 0.15s ease",
              },
              "&:hover .MuiSlider-thumb, & .MuiSlider-thumb.Mui-active": { opacity: 1 },
              "& .MuiSlider-rail": { opacity: 0.35, height: BAR_HEIGHT },
              "& .MuiSlider-track": { height: BAR_HEIGHT, border: "none" },
            }}
          />
        </Box>
        <Stack direction="row" spacing={0.75} sx={{ alignItems: "center" }}>
          <IconButton aria-label={isPlaying ? "Pause" : "Play"} onClick={togglePlay} sx={{ color: "#fff" }}>
            {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
          </IconButton>
          {/* Just the sound icon by default - the volume slider only slides
              out on hover, YouTube-style, rather than permanently taking up
              bar space. Pure CSS (the sibling selector below), so the
              reveal doesn't need its own hover state. Fixed height plus
              flex-centering on both this row and the slider's own wrapper
              keeps the track lined up with the icon's centre regardless of
              the Slider's own default internal padding; the extra 5px of
              horizontal margin on the Slider itself (half the thumb's own
              diameter) stops the thumb being clipped by the wrapper's
              overflow:hidden at the very ends of the track (0 and max
              volume), where it would otherwise overflow the track by half
              its own width. */}
          <Stack
            direction="row"
            spacing={0}
            sx={{ alignItems: "center", height: 40, "&:hover .volume-slider": { width: 82, opacity: 1 } }}
          >
            <IconButton aria-label={isMuted || volume === 0 ? "Unmute" : "Mute"} onClick={toggleMute} sx={{ color: "#fff" }}>
              {isMuted || volume === 0 ? <VolumeOffIcon /> : <VolumeUpIcon />}
            </IconButton>
            <Box
              className="volume-slider"
              sx={{
                width: 0,
                opacity: 0,
                overflow: "hidden",
                display: "flex",
                alignItems: "center",
                height: "100%",
                transition: "width 0.15s ease, opacity 0.15s ease",
              }}
            >
              <Slider
                value={isMuted ? 0 : volume}
                min={0}
                max={1}
                step={0.01}
                onChange={(_, value) => handleVolumeChange(value as number)}
                sx={{
                  width: 72,
                  flexShrink: 0,
                  mx: "5px",
                  color: "#fff",
                  height: 4,
                  "& .MuiSlider-thumb": { width: 10, height: 10 },
                  "& .MuiSlider-rail": { opacity: 0.35, height: 4 },
                  "& .MuiSlider-track": { height: 4, border: "none" },
                }}
              />
            </Box>
          </Stack>
          <Typography variant="caption" sx={{ color: "#fff", minWidth: 88, pl: 0.5 }}>
            {formatTime(currentTime)} / {formatTime(duration)}
          </Typography>
          <Box sx={{ flex: 1 }} />
          <IconButton
            aria-label="Playback settings"
            onClick={(event) => {
              setSettingsView("main");
              setSettingsAnchor(event.currentTarget);
            }}
            sx={{ color: "#fff" }}
          >
            <SettingsIcon />
          </IconButton>
          <IconButton
            aria-label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            onClick={toggleFullscreen}
            sx={{ color: "#fff" }}
          >
            {isFullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
          </IconButton>
        </Stack>
      </Box>

      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
        anchorOrigin={{ vertical: "top", horizontal: "right" }}
        transformOrigin={{ vertical: "bottom", horizontal: "right" }}
        slotProps={{
          paper: {
            sx: {
              bgcolor: "rgba(24,24,24,0.92)",
              backdropFilter: "blur(8px)",
              color: "#fff",
              minWidth: 240,
              borderRadius: 2,
              "& .MuiMenuItem-root": {
                "&:hover": { bgcolor: "rgba(255,255,255,0.08)" },
                "&.Mui-selected": { bgcolor: "rgba(255,255,255,0.12)" },
                "&.Mui-selected:hover": { bgcolor: "rgba(255,255,255,0.16)" },
              },
              "& .MuiListItemIcon-root": { color: "inherit", minWidth: 36 },
              "& .MuiDivider-root": { borderColor: "rgba(255,255,255,0.12)" },
              // Belt-and-suspenders on top of this Paper's own `color: "#fff"`
              // above - this menu is meant to look the same regardless of
              // the app's light/dark theme (it's an overlay ON the video,
              // not part of the surrounding page), so nothing in it should
              // fall back to the theme's own (possibly dark, in light mode)
              // text colour.
              "& .MuiTypography-root, & .MuiMenuItem-root": { color: "inherit" },
            },
          },
        }}
      >
        {settingsView === "main" && [
          <MenuItem key="speed" onClick={() => setSettingsView("speed")}>
            <ListItemIcon>
              <SpeedIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText>Playback speed</ListItemText>
            <Typography variant="body2" sx={{ ml: 3, color: "rgba(255,255,255,0.6)" }}>
              {playbackRate === 1 ? "Normal" : `${playbackRate}x`}
            </Typography>
            <ChevronRightIcon fontSize="small" sx={{ ml: 0.5, color: "rgba(255,255,255,0.6)" }} />
          </MenuItem>,
          hasQualityOptions && (
            <MenuItem key="quality" onClick={() => setSettingsView("quality")}>
              <ListItemIcon>
                <HighQualityIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Quality</ListItemText>
              <Typography variant="body2" sx={{ ml: 3, color: "rgba(255,255,255,0.6)" }}>
                {quality === "original" ? "Original" : quality}
              </Typography>
              <ChevronRightIcon fontSize="small" sx={{ ml: 0.5, color: "rgba(255,255,255,0.6)" }} />
            </MenuItem>
          ),
          hasAnyAnnotation && (
            <MenuItem key="annotations" onClick={() => setSettingsView("annotations")}>
              <ListItemIcon>
                <LayersIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText>Annotations</ListItemText>
              <ChevronRightIcon fontSize="small" sx={{ ml: "auto", color: "rgba(255,255,255,0.6)" }} />
            </MenuItem>
          ),
        ]}

        {settingsView === "speed" && [
          <MenuItem key="back" onClick={() => setSettingsView("main")}>
            <ListItemIcon>
              <ArrowBackIcon fontSize="small" />
            </ListItemIcon>
            <Typography variant="subtitle2">Playback speed</Typography>
          </MenuItem>,
          <Divider key="divider" />,
          ...PLAYBACK_RATES.map((rate) => (
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
          )),
        ]}

        {settingsView === "quality" && qualities && qualities.length > 1 && [
          <MenuItem key="back" onClick={() => setSettingsView("main")}>
            <ListItemIcon>
              <ArrowBackIcon fontSize="small" />
            </ListItemIcon>
            <Typography variant="subtitle2">Quality</Typography>
          </MenuItem>,
          <Divider key="divider" />,
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

        {settingsView === "annotations" && [
          <MenuItem key="back" onClick={() => setSettingsView("main")}>
            <ListItemIcon>
              <ArrowBackIcon fontSize="small" />
            </ListItemIcon>
            <Typography variant="subtitle2">Annotations</Typography>
          </MenuItem>,
          <Divider key="divider" />,
          hasBallTracking && (
            <MenuItem key="ballTracking" onClick={() => toggleAnnotation("ballTracking")}>
              <Checkbox checked={annotations.ballTracking} size="small" sx={{ p: 0, mr: 1.5 }} />
              Ball tracking
            </MenuItem>
          ),
          hasBallArc && (
            <MenuItem key="ballArc" onClick={() => toggleAnnotation("ballArc")}>
              <Checkbox checked={annotations.ballArc} size="small" sx={{ p: 0, mr: 1.5 }} />
              Ball trajectory
            </MenuItem>
          ),
          hasPlayerTracking && (
            <MenuItem key="playerTracking" onClick={() => toggleAnnotation("playerTracking")}>
              <Checkbox checked={annotations.playerTracking} size="small" sx={{ p: 0, mr: 1.5 }} />
              Player tracking
            </MenuItem>
          ),
          hasScore && (
            <MenuItem key="score" onClick={() => toggleAnnotation("score")}>
              <Checkbox checked={annotations.score} size="small" sx={{ p: 0, mr: 1.5 }} />
              Score
            </MenuItem>
          ),
          hasMinimap && (
            <MenuItem key="minimap" onClick={() => toggleAnnotation("minimap")}>
              <Checkbox checked={annotations.minimap} size="small" sx={{ p: 0, mr: 1.5 }} />
              Minimap
            </MenuItem>
          ),
          hasPlayerTracking && (
            <MenuItem key="debug" onClick={() => toggleAnnotation("debug")}>
              <Checkbox checked={annotations.debug} size="small" sx={{ p: 0, mr: 1.5 }} />
              Player IDs (debug)
            </MenuItem>
          ),
        ]}
      </Menu>
    </Box>
  );
}
