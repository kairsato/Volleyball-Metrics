import { useEffect, useState } from "react";
import Box from "@mui/material/Box";
import PersonIcon from "@mui/icons-material/Person";

// How long each photo stays up before cross-fading to the next - slow on
// purpose, this is a quiet background detail on the card, not a slideshow
// competing for attention.
const PHOTO_TRANSITION_INTERVAL_MS = 5000;
const PHOTO_TRANSITION_DURATION_S = 1.5;

// Shared by the Players page's identified-player cards and the Teams
// page's lineup strips - anywhere a player has more than one thumbnail
// available (different videos, different angles), this slowly cross-fades
// between them instead of being stuck on whichever one was found first.
export function PlayerPhotoCarousel({ thumbnails, alt }: { thumbnails: string[]; alt: string }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (thumbnails.length <= 1) return;
    const timer = setInterval(() => {
      setIndex((i) => (i + 1) % thumbnails.length);
    }, PHOTO_TRANSITION_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [thumbnails.length]);

  if (thumbnails.length === 0) {
    return (
      <Box sx={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <PersonIcon sx={{ fontSize: 48, color: "text.disabled" }} />
      </Box>
    );
  }

  return (
    <Box sx={{ position: "relative", width: "100%", height: "100%" }}>
      {thumbnails.map((thumbnail, i) => (
        <Box
          key={i}
          component="img"
          src={`data:image/jpeg;base64,${thumbnail}`}
          alt={alt}
          sx={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            opacity: i === index ? 1 : 0,
            transition: `opacity ${PHOTO_TRANSITION_DURATION_S}s ease-in-out`,
          }}
        />
      ))}
    </Box>
  );
}
