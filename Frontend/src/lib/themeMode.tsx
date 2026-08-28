import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import type { PaletteMode } from "@mui/material";

const STORAGE_KEY = "vva-theme-mode";

interface ThemeModeContextValue {
  mode: PaletteMode;
  setMode: (mode: PaletteMode) => void;
}

const ThemeModeContext = createContext<ThemeModeContextValue | null>(null);

// Defaults to dark regardless of OS preference - a deliberate product
// choice, not a fallback for a missing media query.
function readStoredMode(): PaletteMode {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function ThemeModeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<PaletteMode>(() => readStoredMode());

  const value = useMemo<ThemeModeContextValue>(
    () => ({
      mode,
      setMode: (next: PaletteMode) => {
        setModeState(next);
        try {
          window.localStorage.setItem(STORAGE_KEY, next);
        } catch {
          // localStorage unavailable (private browsing, etc.) - mode still
          // applies for this session, it just won't persist.
        }
      },
    }),
    [mode],
  );

  return <ThemeModeContext.Provider value={value}>{children}</ThemeModeContext.Provider>;
}

export function useThemeMode(): ThemeModeContextValue {
  const ctx = useContext(ThemeModeContext);
  if (!ctx) throw new Error("useThemeMode must be used within a ThemeModeProvider");
  return ctx;
}
