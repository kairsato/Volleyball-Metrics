import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import Link from "@mui/material/Link";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

interface CreditEntry {
  title: string;
  what: string;
  source: string;
  sourceHref?: string;
  license: string;
}

const CREDITS: CreditEntry[] = [
  {
    title: "Player detection & tracking",
    what:
      "Person detection from Ultralytics YOLO, fed into a custom 5-tracker ensemble with a ResNet18-based " +
      "appearance encoder for re-identifying players who briefly leave the frame - built for this project.",
    source: "Ultralytics YOLO",
    sourceHref: "https://github.com/ultralytics/ultralytics",
    license: "AGPL-3.0 (or a commercial Ultralytics license)",
  },
  {
    title: "Ball detection",
    what:
      "A YOLO11m detector fine-tuned specifically on volleyball footage (single class: \"volleyball\"), " +
      "trained on ~15k images from the open VolleyVision / Volleyball_v2 dataset.",
    source: "VolleyVision / Volleyball_v2 dataset (Roboflow), credit: shukur-sabzaliev1",
    sourceHref: "https://universe.roboflow.com",
    license: "CC BY 4.0",
  },
  {
    title: "Game status detection (rally segmentation)",
    what:
      "Every clip of the match is classified as no-play, play, or service by a fine-tuned VideoMAE video " +
      "classifier, which is how rallies get their start/end boundaries.",
    source: "masouduut94/volleyball_analytics (fine-tuned checkpoint), base model MCG-NJU/videomae-base-finetuned-kinetics",
    sourceHref: "https://github.com/masouduut94/volleyball_analytics",
    license: "Base model: CC-BY-NC-4.0 (non-commercial use only)",
  },
  {
    title: "Action detection (serve / set / spike / dig / block)",
    what: "A classifier trained from scratch for this project on labelled clips of each action type.",
    source: "Original to this project",
    license: "N/A",
  },
  {
    title: "Court calibration & homography",
    what:
      "Pixel-to-court coordinate mapping from four user-clicked corners, plus a single-view camera pose " +
      "solved from two net-top points (for ball/action height estimation) - original to this project.",
    source: "Original to this project",
    license: "N/A",
  },
];

export function AboutPage() {
  return (
    <Box sx={{ maxWidth: 860 }}>
      <Typography variant="h4" sx={{ fontWeight: 700, mb: 2 }}>
        About
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 1.5 }}>
        Volleyball Metrics turns a raw match recording into player stats, team analytics, rally breakdowns, and an
        annotated video. Calibrate the court once, and the pipeline tracks every player and the ball, segments the
        match into rallies, classifies each touch (serve/set/spike/dig/block), and rolls all of that up into
        per-player and per-team stats and a rendered copy of the video with tracking overlays.
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 4 }}>
        Named teams (Teams page) and match Scoring (manual, automatic, or scoreboard OCR - set up from a video's
        Setup tab) are optional add-ons on top of that: configure them to unlock win/loss records and win-rate
        radars on the Teams and Players pages, not just the per-video Results page.
      </Typography>

      <Divider sx={{ mb: 4 }} />

      <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
        Credits
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 3 }}>
        This app combines several detection and classification models, some trained for this project and some
        fine-tuned or adapted from open-source work. Attribution and license terms for each below.
      </Typography>

      <Stack spacing={2}>
        {CREDITS.map((entry) => (
          <Card key={entry.title} variant="outlined" sx={{ p: 2.5 }}>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5 }}>
              {entry.title}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
              {entry.what}
            </Typography>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center", flexWrap: "wrap", gap: 1 }}>
              <Typography variant="body2">
                {entry.sourceHref ? (
                  <Link href={entry.sourceHref} target="_blank" rel="noopener noreferrer">
                    {entry.source}
                  </Link>
                ) : (
                  entry.source
                )}
              </Typography>
              <Chip size="small" label={entry.license} variant="outlined" />
            </Stack>
          </Card>
        ))}
      </Stack>
    </Box>
  );
}
