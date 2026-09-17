import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Typography from "@mui/material/Typography";
import { ConfigurationPanel } from "../components/ConfigurationPanel";
import { DebugPanel } from "../components/DebugPanel";
import { ShareSettingsPanel } from "../components/ShareSettingsPanel";

// Mirrors ResultsView's own ?tab=<name> pattern - readable/shareable, and
// what lets App.tsx's legacy /configuration redirect land straight on the
// right tab (?tab=configuration) instead of just /settings.
const TAB_NAMES = ["share", "configuration", "debug"] as const;
const TAB_LABELS: Record<(typeof TAB_NAMES)[number], string> = {
  share: "Share",
  configuration: "Configuration",
  debug: "Debug",
};
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

// The gear menu's one Settings destination - Share (making the app
// reachable off the LAN), Configuration (pipeline heuristic profiles), and
// Debug (reprocessing/accuracy tools) used to be a dialog and a standalone
// page respectively; both are tabs here now so they live in one place. All
// three are LAN-only, same as the gear menu item that links here (see
// App.tsx's SettingsMenu).
export function SettingsPage() {
  const [tab, setTab] = useState(() => readTabFromUrl());

  useEffect(() => {
    function handlePopState() {
      setTab(readTabFromUrl());
    }
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  useEffect(() => {
    document.title = `Settings · ${TAB_LABELS[TAB_NAMES[tab]]}`;
    return () => {
      document.title = "Volleyball Metrics";
    };
  }, [tab]);

  function changeTab(index: number) {
    setTab(index);
    writeTabToUrl(index);
  }

  return (
    <Box>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 2 }}>
        Settings
      </Typography>

      <Tabs value={tab} onChange={(_event, index) => changeTab(index)} sx={{ mb: 3, borderBottom: 1, borderColor: "divider" }}>
        <Tab label="Share" />
        <Tab label="Configuration" />
        <Tab label="Debug" />
      </Tabs>

      {tab === 0 && <ShareSettingsPanel />}
      {tab === 1 && <ConfigurationPanel />}
      {tab === 2 && <DebugPanel />}
    </Box>
  );
}
