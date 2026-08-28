import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../lib/api";

// Matches the AppBar height (64px) plus the shared page padding (p: 5 = 40px
// top and bottom) that App.tsx wraps every routed page in - without
// accounting for that chrome, a height derived purely from "100vh" runs
// past the actual visible area and forces the page to scroll.
const CHROME_PX = 144;

export function VideoPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");

  if (!jobId) return <Navigate to="/playlist" replace />;

  return (
    <Box sx={{ display: "flex", justifyContent: "center" }}>
      <Box sx={{ position: "relative" }}>
        <IconButton
          aria-label="Back to results"
          onClick={() => navigate(`/results?job=${jobId}`)}
          sx={{
            position: "absolute",
            top: 8,
            left: 8,
            zIndex: 1,
            color: "#fff",
            bgcolor: "rgba(0,0,0,0.5)",
            "&:hover": { bgcolor: "rgba(0,0,0,0.7)" },
          }}
        >
          <ArrowBackIcon />
        </IconButton>

        <Box
          sx={{
            height: `calc(100vh - ${CHROME_PX}px)`,
            maxHeight: `calc(100vh - ${CHROME_PX}px)`,
            width: "auto",
            maxWidth: "100%",
            aspectRatio: "16 / 9",
            borderRadius: 2,
            overflow: "hidden",
            bgcolor: "#000",
          }}
        >
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <Box
            component="video"
            controls
            src={api.sourceVideoUrl(jobId)}
            sx={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
          />
        </Box>
      </Box>
    </Box>
  );
}
