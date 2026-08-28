import { createTheme } from "@mui/material/styles";
import type { PaletteMode } from "@mui/material";

// A "court" inspired palette - teal accent (net/court line color) against a
// warm amber for highlights, distinct from the previous purple scheme.
// Dark mode is the default (see lib/themeMode.tsx) so it gets first-class
// tuning: a soft near-black rather than pure black, with a lighter paper
// tone so cards read as distinctly "raised" without needing shadows.
export function createAppTheme(mode: PaletteMode) {
  const isDark = mode === "dark";

  return createTheme({
    palette: {
      mode,
      primary: {
        main: isDark ? "#2dd4bf" : "#0f766e",
      },
      secondary: {
        main: isDark ? "#fb923c" : "#c2410c",
      },
      background: {
        default: isDark ? "#0f1115" : "#f5f6f8",
        paper: isDark ? "#181b21" : "#ffffff",
      },
      divider: isDark ? "rgba(255,255,255,0.09)" : "rgba(15,23,42,0.09)",
    },
    shape: {
      borderRadius: 10,
    },
    typography: {
      fontFamily: "system-ui, 'Segoe UI', Roboto, sans-serif",
    },
    components: {
      MuiButton: {
        defaultProps: { disableElevation: true },
        styleOverrides: {
          root: { textTransform: "none", borderRadius: 8 },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: isDark ? "#14161b" : "#ffffff",
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            backgroundImage: "none",
          },
        },
      },
    },
  });
}
