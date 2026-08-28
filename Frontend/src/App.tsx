import { useCallback, useEffect, useMemo, useState } from "react";
import AppBar from "@mui/material/AppBar";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import IconButton from "@mui/material/IconButton";
import ListItemIcon from "@mui/material/ListItemIcon";
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
import SettingsIcon from "@mui/icons-material/Settings";
import { BrowserRouter, Link as RouterLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { CreditsPage } from "./pages/CreditsPage";
import { HomePage } from "./pages/HomePage";
import { PlaylistPage } from "./pages/PlaylistPage";
import { ResultsPage } from "./pages/ResultsPage";
import { StatsPage } from "./pages/StatsPage";
import { VideoPage } from "./pages/VideoPage";
import { UploadPanel } from "./components/UploadPanel";
import { api } from "./lib/api";
import { ThemeModeProvider, useThemeMode } from "./lib/themeMode";
import type { Job } from "./lib/types";
import { createAppTheme } from "./theme";

function SettingsMenu() {
  const { mode, setMode } = useThemeMode();
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

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
      </Menu>
    </>
  );
}

function TopNav() {
  const location = useLocation();
  // The video for whatever job is currently in view (via ?job= on /results
  // or /video itself) - when present, a "Video" tab is inserted so there's
  // always a one-click jump to the big video view from the persistent nav,
  // not just a button buried in the results page.
  const jobId = new URLSearchParams(location.search).get("job");

  const navItems = [
    { label: "Home", to: "/home" },
    { label: "Playlist", to: "/playlist" },
    { label: "Stats", to: "/stats" },
    ...(jobId ? [{ label: "Video", to: `/video?job=${jobId}` }] : []),
    { label: "Credits", to: "/credits" },
  ];

  const activeTo = navItems.find((item) => item.to.split("?")[0] === location.pathname)?.to ?? false;

  return (
    <AppBar position="sticky" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
      <Toolbar sx={{ gap: 2 }}>
        <Typography
          variant="h6"
          component={RouterLink}
          to="/home"
          sx={{ fontWeight: 600, textDecoration: "none", color: "inherit", whiteSpace: "nowrap" }}
        >
          🏐 Volleyball Analytics
        </Typography>

        <Tabs value={activeTo} sx={{ flexGrow: 1, minHeight: 0 }}>
          {navItems.map((item) => (
            <Tab key={item.to} label={item.label} value={item.to} component={RouterLink} to={item.to} sx={{ minHeight: 0 }} />
          ))}
        </Tabs>

        <SettingsMenu />
      </Toolbar>
    </AppBar>
  );
}

function AppContent() {
  const navigate = useNavigate();
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
  // "Start processing" navigates back to the playlist.
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
    navigate(`/results?job=${job.id}`);
  }

  function handleDeleteJob(jobId: string) {
    setJobs((prev) => prev.filter((j) => j.id !== jobId));
  }

  function selectJob(jobId: string) {
    navigate(`/results?job=${jobId}`);
  }

  return (
    <Box sx={{ minHeight: "100vh" }}>
      <TopNav />

      <Box sx={{ p: 5 }}>
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
                onAddVideo={() => setUploadOpen(true)}
                onViewPlaylist={() => navigate("/playlist")}
              />
            }
          />
          <Route
            path="/playlist"
            element={
              <PlaylistPage
                jobs={jobs}
                onSelectJob={selectJob}
                onAddVideo={() => setUploadOpen(true)}
                onDeleteJob={handleDeleteJob}
              />
            }
          />
          <Route path="/stats" element={<StatsPage jobs={jobs} />} />
          <Route path="/credits" element={<CreditsPage />} />
          <Route path="/results" element={<ResultsPage onJobUpdated={handleJobUpdated} />} />
          <Route path="/video" element={<VideoPage />} />
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
      <AppContent />
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
