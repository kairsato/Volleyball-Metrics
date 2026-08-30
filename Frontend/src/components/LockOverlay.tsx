import Box from "@mui/material/Box";
import Tooltip from "@mui/material/Tooltip";

// Sits over an already-rendered, already-disabled region (e.g. the Scoring
// Determination card, the Player Identification groups, Court Calibration)
// once that section is confirmed/set - the individual controls underneath
// are still marked disabled (so they read as visually locked), but a
// disabled element never fires onClick at all, which would make "locked"
// look simply broken rather than intentional. This transparent layer is
// what actually catches the click and opens the same "Redo ...?"/"Set
// ...?" dialog the section's own button opens, wherever on the locked area
// you click. Shared by ScoreSection.tsx, UnidentifiedPlayersSection.tsx,
// and CalibrationPanel.tsx - all three use the exact same
// confirmed/locked/redo pattern.
export function LockOverlay({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  if (!active) return null;
  return (
    <Tooltip title={label}>
      <Box onClick={onClick} sx={{ position: "absolute", inset: 0, zIndex: 2, cursor: "pointer" }} />
    </Tooltip>
  );
}
