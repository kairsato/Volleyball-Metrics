import type { ComponentProps } from "react";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import Skeleton from "@mui/material/Skeleton";
import Stack from "@mui/material/Stack";

// Shared placeholder shapes used while a page/section's real data is still
// loading - each mirrors the size/layout of what it's standing in for (a
// StatTile card, a video/team/player card, a list row) so the page doesn't
// visibly jump once the real content swaps in.

type GridResponsiveSize = ComponentProps<typeof Grid>["size"];

export function StatTilesSkeleton({ count, size = { xs: 6, sm: 4 } }: { count: number; size?: GridResponsiveSize }) {
  return (
    <Grid container spacing={2}>
      {Array.from({ length: count }).map((_, i) => (
        <Grid key={i} size={size}>
          <Skeleton variant="rounded" height={70} />
        </Grid>
      ))}
    </Grid>
  );
}

export function CardTilesSkeleton({
  width,
  maxWidth,
  height,
  aspectRatio,
  count,
}: {
  width: number | string | Partial<Record<"xs" | "sm" | "md" | "lg", number | string>>;
  /** Caps `width` where it's fluid (e.g. "100%") - mirrors a real card's own maxWidth so the
   * loading placeholder doesn't grow past what the real content will. */
  maxWidth?: number;
  /** Fixed height - use this or `aspectRatio`, not both, matching whichever the real card uses. */
  height?: number;
  /** e.g. "900 / 280" - lets height scale with a fluid width instead of staying fixed, matching
   * a real card that keeps its own ratio while resizing. */
  aspectRatio?: string;
  count: number;
}) {
  return (
    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} variant="rounded" sx={{ width, maxWidth, height, aspectRatio }} />
      ))}
    </Box>
  );
}

export function RowsSkeleton({ count, height = 60 }: { count: number; height?: number }) {
  return (
    <Stack spacing={1}>
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} variant="rounded" height={height} />
      ))}
    </Stack>
  );
}

// Same viewport-fit height every "Setup" tool page (Court Calibration,
// Player Identification, Scoring Determination) computes for itself - kept
// here too so this skeleton fills exactly as much space as the real content
// will, rather than guessing a fixed height and letting the page jump once
// the job loads.
const APP_BAR_HEIGHT_PX = 64;
const PAGE_PADDING_PX = 40; // matches AppContent's `p: 5` (5 * 8px) in App.tsx
const PAGE_CONTENT_HEIGHT = `calc(100vh - ${APP_BAR_HEIGHT_PX + PAGE_PADDING_PX * 2}px)`;

// A "back to results" button plus the tool's full-height content area -
// what all three Setup tool pages show once their job has loaded, so this
// is what they render first while it's still in flight.
export function ToolPageSkeleton() {
  return (
    <Box sx={{ height: PAGE_CONTENT_HEIGHT, display: "flex", flexDirection: "column" }}>
      <Skeleton variant="rounded" width={130} height={30} sx={{ mb: 2, flexShrink: 0 }} />
      <Skeleton variant="rounded" sx={{ flex: 1, minHeight: 0 }} />
    </Box>
  );
}

// JobWorkspace's own loading state, before it even knows the job's status -
// approximates the simple title/description/button shape its "uploaded",
// "error", and "cancelled" states all share (the more elaborate states -
// StageProgress, ResultsView - render their own content once the status is
// known, so there's nothing further to guess at here).
export function JobWorkspaceSkeleton() {
  return (
    <Box sx={{ maxWidth: 560 }}>
      <Skeleton variant="text" width="55%" height={40} sx={{ mb: 1 }} />
      <Skeleton variant="text" width="90%" />
      <Skeleton variant="text" width="70%" sx={{ mb: 2 }} />
      <Skeleton variant="rounded" width={160} height={36} />
    </Box>
  );
}
