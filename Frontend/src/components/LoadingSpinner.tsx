import Box from "@mui/material/Box";
import CircularProgress from "@mui/material/CircularProgress";

interface LoadingSpinnerProps {
  minHeight?: number;
}

export function LoadingSpinner({ minHeight = 160 }: LoadingSpinnerProps) {
  return (
    <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight, py: 4 }}>
      <CircularProgress />
    </Box>
  );
}
