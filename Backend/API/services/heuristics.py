"""Registry + persisted profiles for the pipeline's tunable heuristic
constants, backing the frontend's Configuration page (see App.tsx's gear
menu -> Configuration).

This intentionally does NOT import any Analysis module itself. Each stage's
real module-level constants (e.g. PlayerDetection.tracker.APPEARANCE_WEIGHT)
stay exactly where they are - a Python function looks its globals up from
its own module's __dict__ at call time, not at def time, so overwriting an
attribute on an already-imported module is enough to change what the next
call to that stage actually uses. apply_overrides() below does exactly that,
generically, from whichever module the caller (stage_runner.py for the
Preprocessing/rendering/transcoding stages, results_router.py/team_stats.py
for the on-demand action-quality scoring "consolidating" maps to) already
has imported - see those call sites for the wiring itself.

A param's key is either a plain module attribute name ("APPEARANCE_WEIGHT")
or "DICT_NAME.subkey" for a module-level dict constant that should have one
entry overridden (e.g. "SERVE_WEIGHTS.speed" for
API.services.action_quality.SERVE_WEIGHTS["speed"]).
"""
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from .. import config

PROFILES_FILE_NAME = "heuristic_profiles.json"

DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_NAME = "Default"

ParamType = Literal["float", "int", "enum"]


@dataclass(frozen=True)
class ParamDef:
    key: str
    label: str
    description: str
    type: ParamType
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list[str]] = None
    group: Optional[str] = None  # e.g. "Serve" - lets a stage's params be sub-grouped in the UI


@dataclass(frozen=True)
class StageDef:
    key: str
    label: str
    phase: Literal["preprocessing", "postprocessing"]
    summary: str
    params: list[ParamDef] = field(default_factory=list)


# --- Registry -----------------------------------------------------------
# One entry per pipeline stage (matching Frontend/src/lib/stages.ts's
# PHASE_ONE_STAGES/PHASE_TWO_STAGES keys/labels exactly, so the
# Configuration page's flowchart lines up with what a processing job's own
# progress view already calls each stage). Only a stage's *general*
# heuristics are exposed here - thresholds and weights that shape a
# decision - not model paths, file names, or internals a value change could
# silently break (e.g. a video classifier's clip/window length).

REGISTRY: list[StageDef] = [
    StageDef(
        key="player_tracking",
        label="Player tracking",
        phase="preprocessing",
        summary="Detects and re-identifies players across the video into stable per-player identities.",
        params=[
            ParamDef("APPEARANCE_WEIGHT", "Appearance weight (with court zones)",
                     "How much a player's visual appearance counts vs. their court position once zones are "
                     "available, when matching a tracklet to an identity.",
                     "float", 0.45, 0.0, 1.0, 0.01),
            ParamDef("APPEARANCE_WEIGHT_WITHOUT_ZONES", "Appearance weight (no court zones yet)",
                     "Same as above, but before a court calibration exists to derive zones from - appearance "
                     "is trusted far more heavily since position isn't available at all.",
                     "float", 0.85, 0.0, 1.0, 0.01),
            ParamDef("ZONE_WEIGHT", "Court-zone weight",
                     "How much a player's court zone counts toward the re-identification cost.",
                     "float", 0.40, 0.0, 1.0, 0.01),
            ParamDef("CANDIDATE_MAX_APPEARANCE_DISTANCE", "Candidate appearance distance limit",
                     "Maximum appearance distance for two tracklets to even be considered the same candidate "
                     "identity. Lower is stricter (fewer false merges, more fragmented identities).",
                     "float", 0.34, 0.0, 1.0, 0.01),
            ParamDef("APPEARANCE_MAX_DISTANCE", "Confirmed match appearance distance limit",
                     "Maximum appearance distance allowed for a confirmed identity match.",
                     "float", 0.22, 0.0, 1.0, 0.01),
            ParamDef("SIDE_MISMATCH_PENALTY", "Net-side mismatch penalty",
                     "Extra cost added when a candidate match would put a player on the opposite side of the "
                     "net from where they were last seen.",
                     "float", 0.35, 0.0, 1.0, 0.01),
            ParamDef("SIDE_MIN_CONFIDENCE", "Net-side confidence threshold",
                     "Minimum confidence required before a player's net side is trusted enough to apply the "
                     "mismatch penalty above.",
                     "float", 0.7, 0.0, 1.0, 0.01),
            ParamDef("MERGE_MAX_COST", "Merge cost limit",
                     "Maximum combined cost allowed to merge two tracklets into a single identity.",
                     "float", 0.45, 0.0, 2.0, 0.01),
            ParamDef("MAX_PLAYERS_PER_SIDE", "Max players per side",
                     "Roster cap per side of the net - identities beyond this many on one side are treated as "
                     "non-players. Matches this pipeline's court-gating: identity counts are sanity-checked "
                     "against the roster.",
                     "int", 6, 4, 8, 1),
            ParamDef("MIN_TRACKLET_FRAMES", "Minimum tracklet length (frames)",
                     "Tracklets shorter than this are discarded before identity matching even begins.",
                     "int", 2, 1, 30, 1),
            ParamDef("MIN_IN_PLAY_FRACTION", "Minimum in-play fraction",
                     "Minimum fraction of a tracklet's frames that must fall during live play for it to count.",
                     "float", 0.5, 0.0, 1.0, 0.01),
        ],
    ),
    StageDef(
        key="ball_detection",
        label="Ball detection",
        phase="preprocessing",
        summary="Detects the ball each frame and picks out its real flight trajectory from the raw candidates.",
        params=[
            ParamDef("COLLECTION_CONF_THRESHOLD", "Candidate collection confidence",
                     "Minimum model confidence for a detection to even be logged as a ball candidate.",
                     "float", 0.05, 0.0, 1.0, 0.01),
            ParamDef("MIN_SELECT_CONFIDENCE", "Trajectory selection confidence",
                     "Minimum confidence for a candidate to be selected into the chosen ball trajectory.",
                     "float", 0.60, 0.0, 1.0, 0.01),
            ParamDef("MIN_BALL_DIAGONAL_PX", "Minimum ball size (px)",
                     "Detections with a bounding-box diagonal smaller than this many pixels are discarded as "
                     "implausible.",
                     "float", 6.0, 1.0, 30.0, 0.5),
            ParamDef("MAX_PLAUSIBLE_REPORTED_SPEED_MS", "Maximum plausible speed (m/s)",
                     "Ball speed readings above this are discarded as sensor/detection noise rather than a "
                     "real hit.",
                     "float", 45.0, 10.0, 80.0, 1),
            ParamDef("MIN_FLIGHT_SEGMENT_FRAMES", "Minimum flight segment (frames)",
                     "Shortest run of frames that counts as a real ball-in-flight segment.",
                     "int", 8, 2, 30, 1),
            ParamDef("MAX_INTERPOLATE_GAP_FRAMES", "Maximum interpolation gap (frames)",
                     "Largest gap between two detections the tracker will still interpolate the ball through.",
                     "int", 20, 1, 60, 1),
        ],
    ),
    StageDef(
        key="game_status",
        label="Game status detection",
        phase="preprocessing",
        summary="Classifies play/no-play/service throughout the video and segments it into rallies.",
        params=[
            ParamDef("MAX_GAP_SECONDS_WITHIN_RALLY", "Max gap within a rally (s)",
                     "Longest gap in live play still counted as part of the same rally rather than splitting "
                     "it into two.",
                     "float", 1.0, 0.2, 5.0, 0.1),
            ParamDef("MIN_RALLY_DURATION_SECONDS", "Minimum rally duration (s)",
                     "Shortest span of live play counted as a real rally.",
                     "float", 1.5, 0.2, 5.0, 0.1),
            ParamDef("SMOOTHING_WINDOW_CHUNKS", "Smoothing window (chunks)",
                     "Number of classified chunks averaged together to smooth out flicker in the live/dead "
                     "status signal.",
                     "int", 5, 1, 15, 1),
        ],
    ),
    StageDef(
        key="action_detection",
        label="Action detection",
        phase="preprocessing",
        summary="Finds ball contacts and classifies/attributes each one (serve, set, spike, dig, block).",
        params=[
            ParamDef("HIT_ANGLE_THRESHOLD_DEG", "Hit angle threshold (deg)",
                     "Trajectory angle change large enough to count as a hit.",
                     "float", 40.0, 10.0, 80.0, 1),
            ParamDef("HIT_SPEED_RATIO_THRESHOLD", "Hit speed-ratio threshold",
                     "Speed ratio change (before vs. after) large enough to count as a hit.",
                     "float", 1.6, 1.0, 3.0, 0.05),
            ParamDef("MIN_HIT_SEPARATION_SECONDS", "Minimum hit separation (s)",
                     "Shortest allowed time between two detected hits - closer detections are merged.",
                     "float", 0.3, 0.05, 1.0, 0.05),
            ParamDef("MAX_PLAYER_ATTRIBUTION_DISTANCE_M", "Max attribution distance (m)",
                     "Furthest a player can be from a contact point and still have that touch attributed to "
                     "them.",
                     "float", 2.5, 0.5, 6.0, 0.1),
            ParamDef("NET_PROXIMITY_M", "Net proximity (m)",
                     "Distance from the net counted as \"at the net\" when identifying blocks.",
                     "float", 2.5, 0.5, 6.0, 0.1),
            ParamDef("FAST_INCOMING_SPEED_MS", "Fast incoming speed (m/s)",
                     "Incoming ball speed above which a touch qualifies as a dig rather than a routine pass.",
                     "float", 8.0, 2.0, 20.0, 0.5),
            ParamDef("ACTION_DETECTOR_CONFIDENCE_THRESHOLD", "Action classifier confidence",
                     "Minimum confidence for the action-type classifier's own detection to be kept.",
                     "float", 0.6, 0.0, 1.0, 0.01),
        ],
    ),
    StageDef(
        key="consolidating",
        label="Consolidating stats",
        phase="postprocessing",
        summary="Rolls up detected actions into per-player stats, including the action-quality scores shown "
                "on the Stats/Players pages (computed on demand, not stored, so these apply immediately).",
        params=[
            ParamDef("SERVE_WEIGHTS.speed", "Serve: speed weight", "", "float", 0.35, 0.0, 1.0, 0.01, group="Serve"),
            ParamDef("SERVE_WEIGHTS.placement", "Serve: placement weight", "", "float", 0.40, 0.0, 1.0, 0.01, group="Serve"),
            ParamDef("SERVE_WEIGHTS.trajectory_height", "Serve: trajectory height weight", "", "float", 0.25, 0.0, 1.0, 0.01, group="Serve"),
            ParamDef("RECEIVE_WEIGHTS.placement", "Receive: placement weight", "", "float", 0.35, 0.0, 1.0, 0.01, group="Receive"),
            ParamDef("RECEIVE_WEIGHTS.reaction", "Receive: reaction weight", "", "float", 0.30, 0.0, 1.0, 0.01, group="Receive"),
            ParamDef("RECEIVE_WEIGHTS.handling_speed", "Receive: handling-speed weight", "", "float", 0.35, 0.0, 1.0, 0.01, group="Receive"),
            ParamDef("SET_WEIGHTS.placement", "Set: placement weight", "", "float", 0.40, 0.0, 1.0, 0.01, group="Set"),
            ParamDef("SET_WEIGHTS.height", "Set: height weight", "", "float", 0.25, 0.0, 1.0, 0.01, group="Set"),
            ParamDef("SET_WEIGHTS.time_given", "Set: time-given weight", "", "float", 0.35, 0.0, 1.0, 0.01, group="Set"),
            ParamDef("SPIKE_WEIGHTS.positioning", "Spike: positioning weight", "", "float", 0.25, 0.0, 1.0, 0.01, group="Spike"),
            ParamDef("SPIKE_WEIGHTS.speed", "Spike: speed weight", "", "float", 0.30, 0.0, 1.0, 0.01, group="Spike"),
            ParamDef("SPIKE_WEIGHTS.placement", "Spike: placement weight", "", "float", 0.25, 0.0, 1.0, 0.01, group="Spike"),
            ParamDef("SPIKE_WEIGHTS.blockers", "Spike: blockers weight", "", "float", 0.20, 0.0, 1.0, 0.01, group="Spike"),
            ParamDef("BLOCK_WEIGHTS.positioning", "Block: positioning weight", "", "float", 0.35, 0.0, 1.0, 0.01, group="Block"),
            ParamDef("BLOCK_WEIGHTS.reach", "Block: reach weight", "", "float", 0.35, 0.0, 1.0, 0.01, group="Block"),
            ParamDef("BLOCK_WEIGHTS.timing", "Block: timing weight", "", "float", 0.30, 0.0, 1.0, 0.01, group="Block"),
        ],
    ),
    StageDef(
        key="dashboard",
        label="Generating dashboard",
        phase="postprocessing",
        summary="Renders the static per-job stats dashboard from already-computed logs - nothing here is a "
                "tunable heuristic.",
        params=[],
    ),
    StageDef(
        key="rendering",
        label="Rendering annotated video",
        phase="postprocessing",
        summary="Draws the tracking overlays (ball trail, player boxes, action labels) onto a copy of the video.",
        params=[
            ParamDef("TRAIL_LENGTH", "Ball trail length (frames)",
                     "How many past frames of ball position are drawn as a trail.",
                     "int", 15, 1, 60, 1),
            ParamDef("BALL_MARKER_RADIUS", "Ball marker radius (px)",
                     "Radius of the marker drawn at the ball's current position.",
                     "int", 6, 2, 20, 1),
            ParamDef("ACTION_DISPLAY_SECONDS", "Action label duration (s)",
                     "How long a detected action's label stays on screen after it happens.",
                     "float", 1.5, 0.2, 5.0, 0.1),
        ],
    ),
    StageDef(
        key="transcoding",
        label="Generating quality renditions",
        phase="postprocessing",
        summary="Encodes the downscaled 1080p/720p/480p video renditions the Results page's quality selector offers.",
        params=[
            ParamDef("PRESET", "Encode preset",
                     "libx264 speed/efficiency tradeoff - slower presets compress better at the same quality "
                     "but take longer to encode.",
                     "enum", "veryfast", options=[
                         "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow",
                     ]),
            ParamDef("CRF", "Quality (CRF)",
                     "Constant rate factor - lower means higher quality and larger files.",
                     "int", 23, 15, 35, 1),
        ],
    ),
]

STAGE_BY_KEY: dict[str, StageDef] = {stage.key: stage for stage in REGISTRY}


def _profiles_file() -> Path:
    return config.DATA_DIR / PROFILES_FILE_NAME


def _default_values() -> dict[str, dict[str, Any]]:
    return {stage.key: {param.key: param.default for param in stage.params} for stage in REGISTRY}


def _fresh_store() -> dict[str, Any]:
    return {
        "active_profile_id": DEFAULT_PROFILE_ID,
        "profiles": {
            DEFAULT_PROFILE_ID: {"name": DEFAULT_PROFILE_NAME, "values": _default_values()},
        },
    }


def _load_store() -> dict[str, Any]:
    if not _profiles_file().exists():
        return _fresh_store()
    try:
        store = json.loads(_profiles_file().read_text())
    except json.JSONDecodeError:
        return _fresh_store()

    profiles = store.get("profiles", {})
    # The Default profile always reflects the registry's own current
    # defaults - never persisted-then-drifted - so it stays a reliable
    # baseline to duplicate from even if the registry itself changes
    # between runs.
    profiles[DEFAULT_PROFILE_ID] = {"name": DEFAULT_PROFILE_NAME, "values": _default_values()}
    if store.get("active_profile_id") not in profiles:
        store["active_profile_id"] = DEFAULT_PROFILE_ID
    store["profiles"] = profiles
    return store


def _save_store(store: dict[str, Any]) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _profiles_file().write_text(json.dumps(store, indent=2))


def _resolve_values(stored_values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merges a profile's stored values over the registry's current
    defaults, so a param added to the registry after a profile was created
    still shows up (at its default) instead of being silently missing."""
    resolved: dict[str, dict[str, Any]] = {}
    for stage in REGISTRY:
        stage_values = dict(stored_values.get(stage.key, {}))
        resolved[stage.key] = {param.key: stage_values.get(param.key, param.default) for param in stage.params}
    return resolved


def _profile_out(profile_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": profile_id,
        "name": profile["name"],
        "is_default": profile_id == DEFAULT_PROFILE_ID,
        "values": _resolve_values(profile.get("values", {})),
    }


def get_registry() -> list[StageDef]:
    return REGISTRY


def list_state() -> dict[str, Any]:
    store = _load_store()
    profiles = [_profile_out(pid, p) for pid, p in store["profiles"].items()]
    profiles.sort(key=lambda p: (not p["is_default"], p["name"].lower()))
    return {"active_profile_id": store["active_profile_id"], "profiles": profiles}


def get_active_profile() -> dict[str, Any]:
    store = _load_store()
    active_id = store["active_profile_id"]
    return _profile_out(active_id, store["profiles"][active_id])


class HeuristicsError(ValueError):
    pass


def _validate_param(param: ParamDef, value: Any) -> Any:
    if param.type == "enum":
        if value not in (param.options or []):
            raise HeuristicsError(f"{param.key}: must be one of {param.options}")
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HeuristicsError(f"{param.key}: must be a number")
    if param.min is not None and value < param.min:
        raise HeuristicsError(f"{param.key}: must be >= {param.min}")
    if param.max is not None and value > param.max:
        raise HeuristicsError(f"{param.key}: must be <= {param.max}")
    return int(value) if param.type == "int" else float(value)


def _validate_values(values: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for stage_key, stage_values in values.items():
        stage = STAGE_BY_KEY.get(stage_key)
        if stage is None:
            raise HeuristicsError(f"Unknown stage: {stage_key}")
        params_by_key = {param.key: param for param in stage.params}
        validated[stage_key] = {}
        for param_key, value in stage_values.items():
            param = params_by_key.get(param_key)
            if param is None:
                raise HeuristicsError(f"Unknown parameter: {stage_key}.{param_key}")
            validated[stage_key][param_key] = _validate_param(param, value)
    return validated


def create_profile(name: str, base_profile_id: Optional[str] = None) -> dict[str, Any]:
    store = _load_store()
    base_id = base_profile_id if base_profile_id in store["profiles"] else DEFAULT_PROFILE_ID
    base_values = store["profiles"][base_id]["values"]
    profile_id = uuid.uuid4().hex
    store["profiles"][profile_id] = {"name": name, "values": json.loads(json.dumps(base_values))}
    _save_store(store)
    return _profile_out(profile_id, store["profiles"][profile_id])


def update_profile(profile_id: str, name: Optional[str] = None,
                    values: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, Any]:
    if profile_id == DEFAULT_PROFILE_ID:
        raise HeuristicsError("The Default profile can't be edited - duplicate it to customize.")
    store = _load_store()
    profile = store["profiles"].get(profile_id)
    if profile is None:
        raise HeuristicsError(f"No such profile: {profile_id}")

    if name is not None:
        profile["name"] = name
    if values is not None:
        validated = _validate_values(values)
        merged = dict(profile.get("values", {}))
        for stage_key, stage_values in validated.items():
            merged[stage_key] = {**merged.get(stage_key, {}), **stage_values}
        profile["values"] = merged

    _save_store(store)
    return _profile_out(profile_id, profile)


def delete_profile(profile_id: str) -> dict[str, Any]:
    if profile_id == DEFAULT_PROFILE_ID:
        raise HeuristicsError("The Default profile can't be deleted.")
    store = _load_store()
    if profile_id not in store["profiles"]:
        raise HeuristicsError(f"No such profile: {profile_id}")
    del store["profiles"][profile_id]
    if store["active_profile_id"] == profile_id:
        store["active_profile_id"] = DEFAULT_PROFILE_ID
    _save_store(store)
    return list_state()


def set_active_profile(profile_id: str) -> dict[str, Any]:
    store = _load_store()
    if profile_id not in store["profiles"]:
        raise HeuristicsError(f"No such profile: {profile_id}")
    store["active_profile_id"] = profile_id
    _save_store(store)
    return list_state()


def apply_overrides(stage_key: str, module: Any) -> None:
    """Pushes the active profile's values for one stage onto the given
    already-imported module's globals, so the very next call into that
    module picks them up. Safe to call unconditionally (e.g. once per
    stage_runner.py invocation) even for a stage whose module doesn't
    define a given constant - those are skipped rather than raising, so a
    registry entry can never crash a pipeline run over a stale key."""
    stage = STAGE_BY_KEY.get(stage_key)
    if stage is None or not stage.params:
        return
    values = get_active_profile()["values"].get(stage_key, {})
    for key, value in values.items():
        if "." in key:
            container_name, subkey = key.split(".", 1)
            container = getattr(module, container_name, None)
            if isinstance(container, dict) and subkey in container:
                container[subkey] = value
        elif hasattr(module, key):
            setattr(module, key, value)
