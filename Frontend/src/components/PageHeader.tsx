import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import AddIcon from "@mui/icons-material/Add";

interface PageHeaderProps {
  title: string;
  addLabel?: string;
  onAdd?: () => void;
  /** A search/filter row rendered below the title, if any - shared layout
   * (used identically by Videos/Players/Teams) so each page only owns its
   * own filter controls, not the row's spacing/alignment. */
  children?: ReactNode;
}

// Shared by the Videos/Players/Teams pages: a title on the left, the page's
// one primary "add" action on the right - previously each page put its add
// action as a dashed tile inside the grid itself; pulling it up here keeps
// it in a consistent spot as each list grows past a single screen.
export function PageHeader({ title, addLabel, onAdd, children }: PageHeaderProps) {
  return (
    <Box sx={{ mb: 3 }}>
      <Stack direction="row" sx={{ alignItems: "center", justifyContent: "space-between", mb: children ? 2 : 0 }}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
        {onAdd && (
          <Button variant="contained" startIcon={<AddIcon />} onClick={onAdd}>
            {addLabel ?? "Add"}
          </Button>
        )}
      </Stack>
      {children}
    </Box>
  );
}
