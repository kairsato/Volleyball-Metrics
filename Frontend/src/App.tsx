import { useCallback, useEffect, useMemo, useState } from "react";
import AppBar from "@mui/material/AppBar";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import { ThemeProvider } from "@mui/material/styles";
import CheckIcon from "@mui/icons-material/Check";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import LightModeIcon from "@mui/icons-material/LightMode";
import LogoutIcon from "@mui/icons-material/Logout";
import MenuIcon from "@mui/icons-material/Menu";
import SettingsIcon from "@mui/icons-material/Settings";
import TuneIcon from "@mui/icons-material/Tune";
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
import { DebugReviewPage } from "./pages/DebugReviewPage";
import { HomePage } from "./pages/HomePage";
import { PlayerIdentificationPage } from "./pages/PlayerIdentificationPage";
import { PlayersPage } from "./pages/PlayersPage";
import { PlayerStatsPage } from "./pages/PlayerStatsPage";
import { ScoringDeterminationPage } from "./pages/ScoringDeterminationPage";
import { SettingsPage } from "./pages/SettingsPage";
import { TeamsPage } from "./pages/TeamsPage";
import { TeamStatsPage } from "./pages/TeamStatsPage";
import { GamePage } from "./pages/GamePage";
import { GamesPage } from "./pages/GamesPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { WarmupPeriodPage } from "./pages/WarmupPeriodPage";
import { LoginGate } from "./components/LoginGate";
import { UploadPanel } from "./components/UploadPanel";
import { api } from "./lib/api";
import { ThemeModeProvider, useThemeMode } from "./lib/themeMode";
import type { Job } from "./lib/types";
import { createAppTheme } from "./theme";

function SettingsMenu() {
  const { mode, setMode } = useThemeMode();
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  // Gates the Settings item - it opens SettingsPage's Share/Configuration/
  // Debug tabs, all of which are LAN-only server-side too (see main.py's
  // LAN_ONLY_PATH_PREFIXES). Defaults to true (shown) while this is still
  // loading - briefly showing the item is harmless (every call those tabs
  // could make is already blocked server-side for a remote visitor
  // regardless), a flash of hidden-then-shown isn't.
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
              setAnchorEl(null);
              navigate("/settings");
            }}
          >
            <ListItemIcon>
              <TuneIcon fontSize="small" />
            </ListItemIcon>
            Settings
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
    </>
  );
}

const NAV_ITEMS = [
  { label: "Home", to: "/home" },
  { label: "Games", to: "/games" },
  { label: "Players", to: "/players" },
  { label: "Teams", to: "/teams" },
  { label: "About", to: "/about" },
];

// The Setup page used to be a standalone /game/setup?job=&tab=<name> route
// with its own Court/Players/Score tabs; Setup is a tab on the results page
// again now (see ResultsView.tsx's SetupTab), with Court living in a dialog
// there and Players/Score each getting their own page. This keeps any old
// /game/setup links (bookmarks, the Players page's per-game shortcuts
// elsewhere) working by mapping their ?tab= value onto the new destination.
function LegacySetupRedirect() {
  const [searchParams] = useSearchParams();
  const jobId = searchParams.get("job");
  const tab = searchParams.get("tab");

  if (!jobId) return <Navigate to="/games" replace />;
  if (tab === "players") return <Navigate to={`/game/setup/player-identification?job=${jobId}`} replace />;
  if (tab === "score") return <Navigate to={`/game/setup/scoring-determination?job=${jobId}`} replace />;
  return <Navigate to={`/game?job=${jobId}&tab=setup`} replace />;
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
    navigate(`/game?job=${job.id}`);
  }

  function handleDeleteJob(jobId: string) {
    setJobs((prev) => prev.filter((j) => j.id !== jobId));
  }

  function selectJob(jobId: string) {
    navigate(`/game?job=${jobId}`);
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
                onViewGames={() => navigate("/games")}
              />
            }
          />
          <Route
            path="/games"
            element={
              <GamesPage
                jobs={jobs}
                onSelectJob={selectJob}
                onAddVideo={() => setUploadOpen(true)}
                onDeleteJob={handleDeleteJob}
              />
            }
          />
          <Route path="/players" element={<PlayersPage />} />
          <Route path="/player" element={<PlayerStatsPage />} />
          <Route path="/teams" element={<TeamsPage jobs={jobs} />} />
          <Route path="/team" element={<TeamStatsPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/debug/review" element={<DebugReviewPage />} />
          <Route path="/game" element={<GamePage onJobUpdated={handleJobUpdated} />} />
          <Route path="/game/setup" element={<LegacySetupRedirect />} />
          <Route
            path="/game/setup/court-calibration"
            element={<CourtCalibrationPage onJobUpdated={handleJobUpdated} />}
          />
          <Route
            path="/game/setup/player-identification"
            element={<PlayerIdentificationPage onJobUpdated={handleJobUpdated} />}
          />
          <Route path="/game/setup/scoring-determination" element={<ScoringDeterminationPage />} />
          <Route
            path="/game/setup/warmup-period"
            element={<WarmupPeriodPage onJobUpdated={handleJobUpdated} />}
          />
          {/* Old routes from before Playlist/Results were renamed to Videos (now Games), Stats to Players, and Credits to About - redirect rather than 404 in case anything still links here. */}
          <Route path="/playlist" element={<Navigate to="/games" replace />} />
          <Route path="/results" element={<Navigate to={`/game${location.search}`} replace />} />
          <Route path="/stats" element={<Navigate to="/players" replace />} />
          <Route path="/credits" element={<Navigate to="/about" replace />} />
          {/* Configuration used to be its own page off the gear menu - it's the Configuration tab of Settings now (see SettingsPage.tsx), alongside Share and Debug. */}
          <Route path="/configuration" element={<Navigate to="/settings?tab=configuration" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Box>

      <Dialog open={uploadOpen} onClose={() => setUploadOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add a new game</DialogTitle>
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
