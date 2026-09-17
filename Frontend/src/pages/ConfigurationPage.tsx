import { Fragment, useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Divider from "@mui/material/Divider";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { HeuristicParam, HeuristicsRegistry, HeuristicStage, HeuristicsState, HeuristicValues } from "../lib/types";
import { ToolPageSkeleton } from "../components/Skeletons";

function cloneValues(values: HeuristicValues): HeuristicValues {
  return JSON.parse(JSON.stringify(values));
}

// Groups a stage's params by their optional `group` (e.g. "Serve"/"Spike"
// for the consolidating stage's action-quality weights), preserving each
// group's first-seen order - "" (ungrouped) sorts first since every other
// stage's params have no group at all.
function groupParams(params: HeuristicParam[]): [string, HeuristicParam[]][] {
  const order: string[] = [];
  const byGroup = new Map<string, HeuristicParam[]>();
  for (const param of params) {
    const key = param.group ?? "";
    if (!byGroup.has(key)) {
      order.push(key);
      byGroup.set(key, []);
    }
    byGroup.get(key)!.push(param);
  }
  return order.map((key) => [key, byGroup.get(key)!]);
}

interface StageFlowProps {
  stages: HeuristicStage[];
  selectedKey: string;
  onSelect: (key: string) => void;
}

// A horizontal row of the stages in this phase, connected by arrows -
// deliberately reusing each stage's own key/label from the registry (which
// mirrors Frontend/src/lib/stages.ts's STAGE_LABELS) so this lines up with
// the exact stage names a processing job's own progress view already uses.
function StageFlow({ stages, selectedKey, onSelect }: StageFlowProps) {
  return (
    <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 1.5 }}>
      {stages.map((stage, index) => (
        <Fragment key={stage.key}>
          {index > 0 && <ArrowForwardIcon fontSize="small" sx={{ color: "text.disabled" }} />}
          <Chip
            label={stage.label}
            onClick={() => onSelect(stage.key)}
            color={selectedKey === stage.key ? "primary" : "default"}
            variant={selectedKey === stage.key ? "filled" : "outlined"}
            sx={{ fontWeight: selectedKey === stage.key ? 600 : 400, cursor: "pointer" }}
          />
        </Fragment>
      ))}
    </Stack>
  );
}

interface ParamEditorProps {
  param: HeuristicParam;
  value: number | string;
  disabled: boolean;
  onChange: (value: number | string) => void;
}

function ParamEditor({ param, value, disabled, onChange }: ParamEditorProps) {
  if (param.type === "enum") {
    return (
      <TextField
        select
        size="small"
        label={param.label}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        helperText={param.description}
        sx={{ maxWidth: 320 }}
      >
        {(param.options ?? []).map((option) => (
          <MenuItem key={option} value={option}>
            {option}
          </MenuItem>
        ))}
      </TextField>
    );
  }

  const numericValue = typeof value === "number" ? value : Number(value) || 0;

  return (
    <Box>
      <Stack direction="row" spacing={2} sx={{ alignItems: "center", justifyContent: "space-between" }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          {param.label}
        </Typography>
        <TextField
          type="number"
          size="small"
          value={numericValue}
          disabled={disabled}
          onChange={(event) => {
            const raw = event.target.value;
            if (raw === "") return;
            const parsed = param.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
            if (!Number.isNaN(parsed)) onChange(parsed);
          }}
          slotProps={{
            htmlInput: {
              min: param.min ?? undefined,
              max: param.max ?? undefined,
              step: param.step ?? (param.type === "int" ? 1 : 0.01),
            },
          }}
          sx={{ width: 110 }}
        />
      </Stack>
      {param.min != null && param.max != null && (
        <Slider
          size="small"
          value={numericValue}
          min={param.min}
          max={param.max}
          step={param.step ?? undefined}
          disabled={disabled}
          onChange={(_event, newValue) => onChange(newValue as number)}
          sx={{ mt: 0.5 }}
        />
      )}
      {param.description && (
        <Typography variant="caption" color="text.secondary">
          {param.description}
        </Typography>
      )}
    </Box>
  );
}

interface StagePanelProps {
  stage: HeuristicStage;
  values: Record<string, number | string>;
  disabled: boolean;
  onChange: (paramKey: string, value: number | string) => void;
}

function StagePanel({ stage, values, disabled, onChange }: StagePanelProps) {
  const groups = useMemo(() => groupParams(stage.params), [stage]);

  return (
    <Card variant="outlined" sx={{ p: 3 }}>
      <Chip
        size="small"
        label={stage.phase === "preprocessing" ? "Preprocessing" : "Post-processing"}
        sx={{ mb: 1 }}
      />
      <Typography variant="h6" sx={{ fontWeight: 700 }}>
        {stage.label}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2.5 }}>
        {stage.summary}
      </Typography>

      {stage.params.length === 0 ? (
        <Alert severity="info">No adjustable heuristics for this stage.</Alert>
      ) : (
        <Stack spacing={3}>
          {groups.map(([group, params]) => (
            <Box key={group || "_ungrouped"}>
              {group && (
                <Typography variant="overline" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                  {group}
                </Typography>
              )}
              <Stack spacing={2.5}>
                {params.map((param) => (
                  <ParamEditor
                    key={param.key}
                    param={param}
                    value={values[param.key] ?? param.default}
                    disabled={disabled}
                    onChange={(value) => onChange(param.key, value)}
                  />
                ))}
              </Stack>
            </Box>
          ))}
        </Stack>
      )}
    </Card>
  );
}

export function ConfigurationPage() {
  const navigate = useNavigate();

  const [registry, setRegistry] = useState<HeuristicsRegistry | null>(null);
  const [heuristicsState, setHeuristicsState] = useState<HeuristicsState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [viewingProfileId, setViewingProfileId] = useState<string>("default");
  const [draftValues, setDraftValues] = useState<HeuristicValues>({});
  const [selectedStageKey, setSelectedStageKey] = useState<string>("player_tracking");

  const [saving, setSaving] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [newProfileOpen, setNewProfileOpen] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);

  useEffect(() => {
    document.title = "Configuration";
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, []);

  function loadState(preferredProfileId?: string) {
    return api
      .getHeuristicsState()
      .then((state) => {
        setHeuristicsState(state);
        const nextId =
          preferredProfileId && state.profiles.some((p) => p.id === preferredProfileId)
            ? preferredProfileId
            : state.profiles.some((p) => p.id === viewingProfileId)
              ? viewingProfileId
              : state.active_profile_id;
        setViewingProfileId(nextId);
        const profile = state.profiles.find((p) => p.id === nextId);
        if (profile) setDraftValues(cloneValues(profile.values));
        return state;
      });
  }

  useEffect(() => {
    Promise.all([api.getHeuristicsRegistry(), loadState()])
      .then(([reg]) => setRegistry(reg))
      .catch((err) => setLoadError(err instanceof Error ? err.message : String(err)));
    // Only on mount - loadState reads viewingProfileId itself to decide which
    // profile to keep viewing, so it can't be a dependency without looping.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const viewingProfile = heuristicsState?.profiles.find((p) => p.id === viewingProfileId) ?? null;
  const isActive = heuristicsState?.active_profile_id === viewingProfileId;
  const isDefault = viewingProfile?.is_default ?? false;

  const dirty = useMemo(() => {
    if (!viewingProfile) return false;
    return JSON.stringify(draftValues) !== JSON.stringify(viewingProfile.values);
  }, [draftValues, viewingProfile]);

  function switchViewingProfile(profileId: string) {
    if (dirty && !window.confirm("Discard unsaved changes to this profile?")) return;
    setViewingProfileId(profileId);
    const profile = heuristicsState?.profiles.find((p) => p.id === profileId);
    if (profile) setDraftValues(cloneValues(profile.values));
    setActionError(null);
    setMessage(null);
  }

  function handleParamChange(stageKey: string, paramKey: string, value: number | string) {
    setDraftValues((prev) => ({ ...prev, [stageKey]: { ...prev[stageKey], [paramKey]: value } }));
    setMessage(null);
  }

  function handleDiscard() {
    if (viewingProfile) setDraftValues(cloneValues(viewingProfile.values));
    setActionError(null);
  }

  async function handleSave() {
    if (!viewingProfile) return;
    setSaving(true);
    setActionError(null);
    try {
      await api.updateHeuristicProfile(viewingProfile.id, { values: draftValues });
      await loadState(viewingProfile.id);
      setMessage("Saved.");
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleSetActive() {
    if (!viewingProfile) return;
    setSaving(true);
    setActionError(null);
    try {
      await api.activateHeuristicProfile(viewingProfile.id);
      await loadState(viewingProfile.id);
      setMessage(`"${viewingProfile.name}" is now the active profile.`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleCreateProfile() {
    const name = newProfileName.trim();
    if (!name) return;
    setSaving(true);
    setActionError(null);
    try {
      const created = await api.createHeuristicProfile(name, viewingProfileId);
      await loadState(created.id);
      setNewProfileOpen(false);
      setNewProfileName("");
      setMessage(`Created "${name}" from "${viewingProfile?.name ?? "Default"}".`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleDeleteProfile() {
    if (!viewingProfile) return;
    setSaving(true);
    setActionError(null);
    try {
      await api.deleteHeuristicProfile(viewingProfile.id);
      await loadState();
      setDeleteConfirmOpen(false);
      setMessage(`Deleted "${viewingProfile.name}".`);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  if (loadError) return <Alert severity="error">{loadError}</Alert>;
  if (!registry || !heuristicsState || !viewingProfile) return <ToolPageSkeleton />;

  const preprocessingStages = registry.stages.filter((s) => s.phase === "preprocessing");
  const postprocessingStages = registry.stages.filter((s) => s.phase === "postprocessing");
  const selectedStage = registry.stages.find((s) => s.key === selectedStageKey) ?? preprocessingStages[0];
  const selectedValues = draftValues[selectedStage.key] ?? {};

  return (
    <Box sx={{ maxWidth: 1100 }}>
      <Button startIcon={<ArrowBackIcon />} onClick={() => navigate("/home")} sx={{ alignSelf: "flex-start", mb: 2 }}>
        Back to home
      </Button>

      <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
        Configuration
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 3 }}>
        How a video moves through this pipeline, stage by stage, and the general heuristics each stage uses to make
        its decisions. Changes apply to every video processed while a profile is active - past results aren't
        recomputed automatically (recalibrate/redo a job to apply new tuning to it).
      </Typography>

      <Card variant="outlined" sx={{ p: 2, mb: 3 }}>
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ alignItems: { sm: "center" }, justifyContent: "space-between" }}>
          <Stack direction="row" spacing={1.5} sx={{ alignItems: "center", flexWrap: "wrap", rowGap: 1 }}>
            <TextField
              select
              size="small"
              label="Profile"
              value={viewingProfileId}
              onChange={(event) => switchViewingProfile(event.target.value)}
              sx={{ minWidth: 220 }}
            >
              {heuristicsState.profiles.map((profile) => (
                <MenuItem key={profile.id} value={profile.id}>
                  {profile.name}
                </MenuItem>
              ))}
            </TextField>
            {isActive && <Chip size="small" color="success" label="Active" />}
            {isDefault && <Chip size="small" variant="outlined" label="Read-only" />}
          </Stack>

          <Stack direction="row" spacing={1}>
            <Button size="small" onClick={() => setNewProfileOpen(true)} disabled={saving}>
              Duplicate as new…
            </Button>
            {!isActive && (
              <Button size="small" variant="outlined" onClick={() => void handleSetActive()} disabled={saving}>
                Set active
              </Button>
            )}
            {!isDefault && (
              <Button size="small" color="error" onClick={() => setDeleteConfirmOpen(true)} disabled={saving}>
                Delete
              </Button>
            )}
          </Stack>
        </Stack>

        {isDefault && (
          <Alert severity="info" sx={{ mt: 2 }}>
            The Default profile mirrors this app's built-in tuning and can't be edited directly - duplicate it
            above to start customizing.
          </Alert>
        )}
      </Card>

      {actionError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {actionError}
        </Alert>
      )}
      {message && !actionError && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setMessage(null)}>
          {message}
        </Alert>
      )}

      <Typography variant="h5" sx={{ fontWeight: 700, mb: 1.5 }}>
        Preprocessing
      </Typography>
      <Card variant="outlined" sx={{ p: 2, mb: 3, overflowX: "auto" }}>
        <StageFlow stages={preprocessingStages} selectedKey={selectedStageKey} onSelect={setSelectedStageKey} />
      </Card>

      <Typography variant="h5" sx={{ fontWeight: 700, mb: 1.5 }}>
        Post-processing
      </Typography>
      <Card variant="outlined" sx={{ p: 2, mb: 3, overflowX: "auto" }}>
        <StageFlow stages={postprocessingStages} selectedKey={selectedStageKey} onSelect={setSelectedStageKey} />
      </Card>

      <Divider sx={{ mb: 3 }} />

      <StagePanel
        stage={selectedStage}
        values={selectedValues}
        disabled={isDefault}
        onChange={(paramKey, value) => handleParamChange(selectedStage.key, paramKey, value)}
      />

      {/* Bottom spacer so the sticky save bar never overlaps the last field. */}
      <Box sx={{ height: dirty ? 88 : 24 }} />

      {dirty && (
        <Paper
          elevation={4}
          sx={{
            position: "sticky",
            bottom: 16,
            p: 2,
            display: "flex",
            flexWrap: "wrap",
            gap: 2,
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Typography variant="body2">Unsaved changes to "{viewingProfile.name}"</Typography>
          <Stack direction="row" spacing={1}>
            <Button onClick={handleDiscard} disabled={saving}>
              Discard
            </Button>
            <Button variant="contained" onClick={() => void handleSave()} disabled={saving}>
              Save changes
            </Button>
          </Stack>
        </Paper>
      )}

      <Dialog open={newProfileOpen} onClose={() => setNewProfileOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>New profile</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Starts as a copy of "{viewingProfile.name}".
          </Typography>
          <TextField
            autoFocus
            fullWidth
            label="Profile name"
            value={newProfileName}
            onChange={(event) => setNewProfileName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void handleCreateProfile();
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewProfileOpen(false)}>Cancel</Button>
          <Button variant="contained" disabled={!newProfileName.trim() || saving} onClick={() => void handleCreateProfile()}>
            Create
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteConfirmOpen} onClose={() => setDeleteConfirmOpen(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Delete "{viewingProfile.name}"?</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            {isActive
              ? "This is the active profile - deleting it switches processing back to Default."
              : "This can't be undone."}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(false)}>Cancel</Button>
          <Button color="error" onClick={() => void handleDeleteProfile()} disabled={saving}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
