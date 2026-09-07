import { useCallback, useEffect, useMemo, useState } from "react";
import AppBar from "@mui/material/AppBar";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CssBaseline from "@mui/material/CssBaseline";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Drawer from "@mui/material/Drawer";
import FormControlLabel from "@mui/material/FormControlLabel";
import IconButton from "@mui/material/IconButton";
import Link from "@mui/material/Link";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import TextField from "@mui/material/TextField";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import { ThemeProvider } from "@mui/material/styles";
import CheckIcon from "@mui/icons-material/Check";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import LightModeIcon from "@mui/icons-material/LightMode";
import LogoutIcon from "@mui/icons-material/Logout";
import MenuIcon from "@mui/icons-material/Menu";
import SettingsIcon from "@mui/icons-material/Settings";
import ShareIcon from "@mui/icons-material/Share";
import ShuffleIcon from "@mui/icons-material/Shuffle";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import InputAdornment from "@mui/material/InputAdornment";
import {
  BrowserRouter,
  Link as RouterLink,
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import { AboutPage } from "./pages/AboutPage";
import { CourtCalibrationPage } from "./pages/CourtCalibrationPage";
import { HomePage } from "./pages/HomePage";
import { PlayerIdentificationPage } from "./pages/PlayerIdentificationPage";
import { PlayersPage } from "./pages/PlayersPage";
import { PlayerStatsPage } from "./pages/PlayerStatsPage";
import { ScoringDeterminationPage } from "./pages/ScoringDeterminationPage";
import { TeamsPage } from "./pages/TeamsPage";
import { TeamStatsPage } from "./pages/TeamStatsPage";
import { VideoPage } from "./pages/VideoPage";
import { VideosPage } from "./pages/VideosPage";
import { WarmupPeriodPage } from "./pages/WarmupPeriodPage";
import { LoginGate } from "./components/LoginGate";
import { UploadPanel } from "./components/UploadPanel";
import { api } from "./lib/api";
import { ThemeModeProvider, useThemeMode } from "./lib/themeMode";
import type { AuthStatus, Job, ShareStatus } from "./lib/types";
import { createAppTheme } from "./theme";

// Everything needed to make this app reachable outside the local network:
// a login password (typed or generated, off by default - see
// Backend/API/auth.py's module docstring), an optional UPnP toggle to have
// the router open the needed ports automatically, and a copyable share URL.
// The login gate itself lives in LoginGate.tsx, which picks up this
// dialog's changes on its very next status check.
function formatExpiryTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function formatRemaining(iso: string): string {
  const totalMinutes = Math.max(0, Math.floor((new Date(iso).getTime() - Date.now()) / 60_000));
  const days = Math.floor(totalMinutes / (60 * 24));
  const hours = Math.floor((totalMinutes % (60 * 24)) / 60);
  if (days > 0) return `${days} day${days === 1 ? "" : "s"} ${hours} hr${hours === 1 ? "" : "s"}`;
  const minutes = totalMinutes % 60;
  return `${hours} hr${hours === 1 ? "" : "s"} ${minutes} min${minutes === 1 ? "" : "s"}`;
}

function ShareSettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [password, setPassword] = useState("");
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [share, setShare] = useState<ShareStatus | null>(null);
  const [upnpSaving, setUpnpSaving] = useState(false);
  const [urlRevealed, setUrlRevealed] = useState(false);
  const [hostnameInput, setHostnameInput] = useState("");
  const [hostnameSaving, setHostnameSaving] = useState(false);
  // Turning ports off is a real router change, not just a UI flip - confirm
  // before actually doing it rather than closing them on the first click.
  const [confirmUpnpOff, setConfirmUpnpOff] = useState(false);
  // How long Share stays on before auto-disabling itself once switched on -
  // null means "Forever" (never auto-expires). Only read at the moment the
  // switch turns on (see handleToggleEnabled) - not persisted/restored, so
  // this just resets to the default each time the dialog reopens.
  const [durationDays, setDurationDays] = useState<number | null>(7);

  useEffect(() => {
    if (!open) return;
    setPassword("");
    setPasswordVisible(false);
    setError(null);
    setMessage(null);
    setUrlRevealed(false);
    setDurationDays(7);
    api.authStatus().then(setStatus).catch(() => undefined);
    api
      .shareStatus()
      .then((s) => {
        setShare(s);
        setHostnameInput(s.hostname ?? "");
      })
      .catch(() => undefined);
  }, [open]);

  async function handleSetPassword() {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const res = await api.authSetPassword(password);
      setStatus(res);
      setPassword("");
      setPasswordVisible(false);
      setMessage("Password updated.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleSuggestPassword() {
    setError(null);
    try {
      const res = await api.authSuggestPassword();
      setPassword(res.password);
      setPasswordVisible(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleToggleEnabled(enabled: boolean) {
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      setStatus(await api.authSetEnabled(enabled, durationDays));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleToggleUpnp(enabled: boolean) {
    setUpnpSaving(true);
    setError(null);
    try {
      setShare(enabled ? await api.enableUpnp() : await api.disableUpnp());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUpnpSaving(false);
    }
  }

  async function handleSetHostname() {
    setHostnameSaving(true);
    setError(null);
    setMessage(null);
    try {
      const res = await api.setHostname(hostnameInput.trim() || null);
      setShare(res);
      setHostnameInput(res.hostname ?? "");
      setMessage(res.hostname ? "Hostname saved." : "Hostname cleared.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setHostnameSaving(false);
    }
  }

  function copyToClipboard(text: string) {
    navigator.clipboard?.writeText(text).catch(() => undefined);
  }

  return (
    <>
      <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
        <DialogTitle>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="h6" component="span">
              Share
            </Typography>
            <FormControlLabel
              labelPlacement="start"
              sx={{ mr: 0 }}
              control={
                <Switch
                  checked={status?.enabled ?? false}
                  disabled={saving || !status?.password_set}
                  onChange={(event) => void handleToggleEnabled(event.target.checked)}
                />
              }
              label="Enable Share"
            />
          </Stack>
        </DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 1 }}>
            <Alert severity="warning">
              Enabling Share makes this app reachable from the internet. Your password is the only thing standing
              between a stranger and your videos - use a strong one (ideally generated). Share settings themselves
              (this dialog, and changing the password) stay locked to your local network either way - only the rest
              of the app is reachable remotely.
            </Alert>

            <TextField
              select
              size="small"
              label="Share stays on for"
              value={durationDays === null ? "forever" : String(durationDays)}
              onChange={(event) => setDurationDays(event.target.value === "forever" ? null : Number(event.target.value))}
              disabled={saving}
              sx={{ maxWidth: 220 }}
            >
              <MenuItem value="1">1 day</MenuItem>
              <MenuItem value="7">7 days</MenuItem>
              <MenuItem value="forever">Forever</MenuItem>
            </TextField>

            <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
              <Stack spacing={2}>
                <Typography variant="subtitle2">Password</Typography>
                <Typography variant="body2" color="text.secondary">
                  Off by default. Allows outside users to access this site, but requires authentication (a password) to
                  do so.
                </Typography>

                <TextField
                  type={passwordVisible ? "text" : "password"}
                  label={status?.password_set ? "New password" : "Set a password"}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  helperText="At least 16 characters, with at least 3 of: lowercase, uppercase, digits, symbols."
                  fullWidth
                  slotProps={{
                    input: {
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            size="small"
                            aria-label="Generate a strong password"
                            onClick={() => void handleSuggestPassword()}
                          >
                            <ShuffleIcon fontSize="small" />
                          </IconButton>
                          <IconButton
                            size="small"
                            aria-label={passwordVisible ? "Hide password" : "Show password"}
                            onClick={() => setPasswordVisible((v) => !v)}
                          >
                            {passwordVisible ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                          </IconButton>
                          <IconButton
                            size="small"
                            aria-label="Copy password"
                            disabled={!password}
                            onClick={() => copyToClipboard(password)}
                          >
                            <ContentCopyIcon fontSize="small" />
                          </IconButton>
                        </InputAdornment>
                      ),
                    },
                  }}
                />
                <Button variant="outlined" disabled={saving || !password} onClick={() => void handleSetPassword()}>
                  {status?.password_set ? "Update password" : "Set password"}
                </Button>

                {!status?.password_set && (
                  <Typography variant="caption" color="text.secondary">
                    Set a password above before turning this on.
                  </Typography>
                )}
                {status?.enabled && (
                  <Typography variant="caption" color="text.secondary">
                    {status.expires_at
                      ? `Expires ${formatExpiryTimestamp(status.expires_at)} (${formatRemaining(status.expires_at)} remaining).`
                      : "Set to never expire automatically - turn it off yourself when you're done."}
                  </Typography>
                )}
              </Stack>
            </Paper>

            <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
              <Stack spacing={1.5}>
                <Stack direction="row" alignItems="center" spacing={1}>
                  <Typography variant="subtitle2">Port forwarding</Typography>
                  <Chip label="Required" size="small" color="warning" variant="outlined" />
                </Stack>

                <Alert severity="info" sx={{ py: 0.5 }}>
                  Some ISPs use CGNAT by default, which silently stops port forwarding from working at all - visit your
                  ISP's website to check if this applies to you.
                </Alert>

                {share && (
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {share.ports.map((p) => (
                      <Chip
                        key={p.port}
                        label={`Port ${p.port} · TCP${p.status === "open" ? " · Open" : p.status === "error" ? " · Error" : ""}`}
                        size="small"
                        color={p.status === "open" ? "success" : "error"}
                        variant={p.status === "open" ? "filled" : "outlined"}
                      />
                    ))}
                  </Stack>
                )}

                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  Methods
                </Typography>

                <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                  <Stack spacing={1} sx={{ flex: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      1. Automatic (UPnP)
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Your router must support UPnP and have it enabled.
                    </Typography>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={share?.upnp_enabled ?? false}
                          disabled={upnpSaving || !status?.enabled}
                          onChange={(event) =>
                            event.target.checked ? void handleToggleUpnp(true) : setConfirmUpnpOff(true)
                          }
                        />
                      }
                      label="Automatically open ports"
                    />
                    {!status?.enabled && (
                      <Typography variant="caption" color="text.secondary">
                        Turn on Enable Share above first.
                      </Typography>
                    )}
                    {share?.last_error && (
                      <Typography variant="caption" color="error">
                        {share.last_error}
                      </Typography>
                    )}
                  </Stack>

                  <Stack spacing={1} sx={{ flex: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      2. Manual
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Forward the ports above yourself in your router's settings.{" "}
                      <Link href="https://portforward.com" target="_blank" rel="noopener noreferrer">
                        portforward.com
                      </Link>{" "}
                      has router-specific guides if you're not sure how.
                    </Typography>
                  </Stack>
                </Stack>
              </Stack>
            </Paper>

            <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
              <Stack spacing={1}>
                <Typography variant="subtitle2">Custom domain (optional)</Typography>
                <Typography variant="body2" color="text.secondary">
                  Point a domain at this machine - a free dynamic-DNS one (e.g. via DuckDNS) works fine if your IP
                  isn't static - and set it here. The Share URL below uses it instead of a raw IP, and if you're
                  running this behind the bundled Caddy reverse proxy (see ../Caddyfile), Caddy picks it up
                  automatically next time it starts, so it can get a real, browser-trusted HTTPS certificate for it.
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  <TextField
                    size="small"
                    fullWidth
                    placeholder="example.duckdns.org"
                    value={hostnameInput}
                    onChange={(event) => setHostnameInput(event.target.value)}
                    disabled={hostnameSaving}
                  />
                  <Button
                    variant="outlined"
                    disabled={hostnameSaving || hostnameInput.trim() === (share?.hostname ?? "")}
                    onClick={() => void handleSetHostname()}
                  >
                    Save
                  </Button>
                </Stack>
              </Stack>
            </Paper>

            <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
              <Stack spacing={1}>
                <Typography variant="subtitle2">Share URL</Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  <TextField
                    size="small"
                    fullWidth
                    value={share ? (urlRevealed ? share.share_url : "•".repeat(share.share_url.length)) : ""}
                    slotProps={{ input: { readOnly: true } }}
                  />
                  <IconButton
                    aria-label={urlRevealed ? "Hide share URL" : "Show share URL"}
                    onClick={() => setUrlRevealed((v) => !v)}
                  >
                    {urlRevealed ? <VisibilityOffIcon fontSize="small" /> : <VisibilityIcon fontSize="small" />}
                  </IconButton>
                  <IconButton aria-label="Copy share URL" onClick={() => share && copyToClipboard(share.share_url)}>
                    <ContentCopyIcon fontSize="small" />
                  </IconButton>
                </Stack>
                <Typography variant="caption" color="text.secondary">
                  Hidden by default since it includes this device's network address.
                </Typography>
              </Stack>
            </Paper>

            {message && <Alert severity="success">{message}</Alert>}
            {error && <Alert severity="error">{error}</Alert>}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Close</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={confirmUpnpOff} onClose={() => setConfirmUpnpOff(false)} maxWidth="xs" fullWidth>
        <DialogTitle>Turn off automatic port forwarding?</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            This will close {share?.ports.map((p) => p.port).join(", ") ?? "these"} on your router. You can turn it
            back on anytime.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmUpnpOff(false)}>Cancel</Button>
          <Button
            color="warning"
            onClick={() => {
              setConfirmUpnpOff(false);
              void handleToggleUpnp(false);
            }}
          >
            Turn off
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

function SettingsMenu() {
  const { mode, setMode } = useThemeMode();
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  // Defaults to true (shown) while this is still loading - briefly showing
  // the item is harmless (every /api/share/* call it could make is already
  // blocked server-side for a remote visitor regardless, see
  // main.py's LAN_ONLY_PATH_PREFIXES), a flash of hidden-then-shown isn't.
  const [isLan, setIsLan] = useState(true);

  useEffect(() => {
    api
      .authStatus()
      .then((status) => setIsLan(status.is_lan))
      .catch(() => undefined);
  }, []);

  return (
    <>
      <IconButton aria-label="Settings" onClick={(e) => setAnchorEl(e.currentTarget)}>
        <SettingsIcon />
      </IconButton>
      <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={() => setAnchorEl(null)}>
        <MenuItem
          selected={mode === "dark"}
          onClick={() => {
            setMode("dark");
            setAnchorEl(null);
          }}
        >
          <ListItemIcon>{mode === "dark" ? <CheckIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}</ListItemIcon>
          Dark theme
        </MenuItem>
        <MenuItem
          selected={mode === "light"}
          onClick={() => {
            setMode("light");
            setAnchorEl(null);
          }}
        >
          <ListItemIcon>{mode === "light" ? <CheckIcon fontSize="small" /> : <LightModeIcon fontSize="small" />}</ListItemIcon>
          Light theme
        </MenuItem>
        {isLan && (
          <MenuItem
            onClick={() => {
              setShareDialogOpen(true);
              setAnchorEl(null);
            }}
          >
            <ListItemIcon>
              <ShareIcon fontSize="small" />
            </ListItemIcon>
            Share
          </MenuItem>
        )}
        <MenuItem
          onClick={() => {
            setAnchorEl(null);
            void api.authLogout();
          }}
        >
          <ListItemIcon>
            <LogoutIcon fontSize="small" />
          </ListItemIcon>
          Log out
        </MenuItem>
      </Menu>
      <ShareSettingsDialog open={shareDialogOpen} onClose={() => setShareDialogOpen(false)} />
    </>
  );
}

const NAV_ITEMS = [
  { label: "Home", to: "/home" },
  { label: "Videos", to: "/videos" },
  { label: "Players", to: "/players" },
  { label: "Teams", to: "/teams" },
  { label: "About", to: "/about" },
];

// The Setup page used to be a standalone /video/setup?job=&tab=<name> route
// with its own Court/Players/Score tabs; Setup is a tab on the results page
// again now (see ResultsView.tsx's SetupTab), with Court living in a dialog
// there and Players/Score each getting their own page. This keeps any old
// /video/setup links (bookmarks, the Players page's per-video shortcuts
// elsewhere) working by mapping their ?tab= value onto the new destination.
function LegacySetupRedirect() {
  const [searchParams] = useSearchParams();
  const jobId = searchParams.get("job");
  const tab = searchParams.get("tab");

  if (!jobId) return <Navigate to="/videos" replace />;
  if (tab === "players") return <Navigate to={`/video/setup/player-identification?job=${jobId}`} replace />;
  if (tab === "score") return <Navigate to={`/video/setup/scoring-determination?job=${jobId}`} replace />;
  return <Navigate to={`/video?job=${jobId}&tab=setup`} replace />;
}

function TopNav() {
  const location = useLocation();
  const activeTo = NAV_ITEMS.find((item) => item.to === location.pathname)?.to ?? false;
  const [navOpen, setNavOpen] = useState(false);

  return (
    <AppBar position="sticky" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
      <Toolbar sx={{ gap: { xs: 1, sm: 2 } }}>
        <IconButton
          aria-label="Open navigation menu"
          onClick={() => setNavOpen(true)}
          sx={{ display: { xs: "inline-flex", md: "none" } }}
        >
          <MenuIcon />
        </IconButton>

        <Typography
          variant="h6"
          component={RouterLink}
          to="/home"
          sx={{ fontWeight: 600, textDecoration: "none", color: "inherit", whiteSpace: "nowrap", flexShrink: 0 }}
        >
          🏐 Volleyball Metrics
        </Typography>

        <Tabs value={activeTo} sx={{ flexGrow: 1, minHeight: 0, display: { xs: "none", md: "flex" } }}>
          {NAV_ITEMS.map((item) => (
            <Tab key={item.to} label={item.label} value={item.to} component={RouterLink} to={item.to} sx={{ minHeight: 0 }} />
          ))}
        </Tabs>

        <Box sx={{ flexGrow: { xs: 1, md: 0 } }} />

        <SettingsMenu />
      </Toolbar>

      <Drawer anchor="left" open={navOpen} onClose={() => setNavOpen(false)}>
        <Box sx={{ width: 240 }} role="presentation" onClick={() => setNavOpen(false)}>
          <List>
            {NAV_ITEMS.map((item) => (
              <ListItemButton key={item.to} component={RouterLink} to={item.to} selected={activeTo === item.to}>
                <ListItemText primary={item.label} />
              </ListItemButton>
            ))}
          </List>
        </Box>
      </Drawer>
    </AppBar>
  );
}

function AppContent() {
  const navigate = useNavigate();
  const location = useLocation();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listJobs()
      .then(setJobs)
      .catch((err) => setListError(err instanceof Error ? err.message : String(err)));
  }, []);

  // Keeps each video's status chip live everywhere even when its
  // JobWorkspace isn't mounted to poll it itself - e.g. right after
  // "Start processing" navigates back to the videos list.
  useEffect(() => {
    const hasRunningJob = jobs.some((j) => j.status === "processing" || j.status === "finalizing");
    if (!hasRunningJob) return;

    const timer = setInterval(() => {
      api.listJobs().then(setJobs).catch(() => undefined);
    }, 3000);

    return () => clearInterval(timer);
  }, [jobs]);

  const handleJobUpdated = useCallback((job: Job) => {
    setJobs((prev) =>
      prev.some((j) => j.id === job.id) ? prev.map((j) => (j.id === job.id ? job : j)) : [job, ...prev],
    );
  }, []);

  function handleUploaded(job: Job) {
    handleJobUpdated(job);
    setUploadOpen(false);
    navigate(`/video?job=${job.id}`);
  }

  function handleDeleteJob(jobId: string) {
    setJobs((prev) => prev.filter((j) => j.id !== jobId));
  }

  function selectJob(jobId: string) {
    navigate(`/video?job=${jobId}`);
  }

  return (
    <Box sx={{ minHeight: "100vh" }}>
      <TopNav />

      <Box sx={{ px: "clamp(16px, 4vw, 40px)", py: 5 }}>
        {listError && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {listError}
          </Alert>
        )}

        <Routes>
          <Route path="/" element={<Navigate to="/home" replace />} />
          <Route
            path="/home"
            element={
              <HomePage
                jobs={jobs}
                onSelectJob={selectJob}
                onViewVideos={() => navigate("/videos")}
              />
            }
          />
          <Route
            path="/videos"
            element={
              <VideosPage
                jobs={jobs}
                onSelectJob={selectJob}
                onAddVideo={() => setUploadOpen(true)}
                onDeleteJob={handleDeleteJob}
              />
            }
          />
          <Route path="/players" element={<PlayersPage jobs={jobs} />} />
          <Route path="/player" element={<PlayerStatsPage jobs={jobs} />} />
          <Route path="/teams" element={<TeamsPage jobs={jobs} />} />
          <Route path="/team" element={<TeamStatsPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/video" element={<VideoPage onJobUpdated={handleJobUpdated} />} />
          <Route path="/video/setup" element={<LegacySetupRedirect />} />
          <Route
            path="/video/setup/court-calibration"
            element={<CourtCalibrationPage onJobUpdated={handleJobUpdated} />}
          />
          <Route
            path="/video/setup/player-identification"
            element={<PlayerIdentificationPage onJobUpdated={handleJobUpdated} />}
          />
          <Route path="/video/setup/scoring-determination" element={<ScoringDeterminationPage />} />
          <Route
            path="/video/setup/warmup-period"
            element={<WarmupPeriodPage onJobUpdated={handleJobUpdated} />}
          />
          {/* Old routes from before Playlist/Results were renamed to Videos, Stats to Players, and Credits to About - redirect rather than 404 in case anything still links here. */}
          <Route path="/playlist" element={<Navigate to="/videos" replace />} />
          <Route path="/results" element={<Navigate to={`/video${location.search}`} replace />} />
          <Route path="/stats" element={<Navigate to="/players" replace />} />
          <Route path="/credits" element={<Navigate to="/about" replace />} />
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
      </Box>

      <Dialog open={uploadOpen} onClose={() => setUploadOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add a new video</DialogTitle>
        <DialogContent>
          <UploadPanel onUploaded={handleUploaded} embedded />
        </DialogContent>
      </Dialog>
    </Box>
  );
}

function ThemedApp() {
  const { mode } = useThemeMode();
  const theme = useMemo(() => createAppTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <LoginGate>
        <AppContent />
      </LoginGate>
    </ThemeProvider>
  );
}

function App() {
  return (
    <BrowserRouter>
      <ThemeModeProvider>
        <ThemedApp />
      </ThemeModeProvider>
    </BrowserRouter>
  );
}

export default App;
