import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import { Link as RouterLink } from "react-router-dom";

// Wildcard route target (see App.tsx's <Route path="*">) for any URL that
// doesn't match a known page - previously this silently redirected to
// /home, which made typoed/stale links look like they worked.
export function NotFoundPage() {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        py: 10,
      }}
    >
      <Typography variant="h1" sx={{ fontWeight: 700, fontSize: { xs: 72, sm: 96 }, color: "text.secondary" }}>
        404
      </Typography>
      <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
        Page not found
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 4, maxWidth: 420 }}>
        The page you're looking for doesn't exist or may have moved.
      </Typography>
      <Button variant="contained" component={RouterLink} to="/home">
        Back to home
      </Button>
    </Box>
  );
}
