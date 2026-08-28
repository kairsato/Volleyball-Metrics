import Accordion from "@mui/material/Accordion";
import AccordionDetails from "@mui/material/AccordionDetails";
import AccordionSummary from "@mui/material/AccordionSummary";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { PlayerGroupingEditor } from "../PlayerGroupingEditor";
import { CourtSummary } from "./CourtSummary";
import type { Job } from "../../lib/types";

interface SetupTabProps {
  job: Job;
  onSaved: (job: Job) => void;
}

export function SetupTab({ job, onSaved }: SetupTabProps) {
  return (
    <>
      <Accordion variant="outlined" defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="overline" color="text.secondary">
            Players
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <PlayerGroupingEditor
            jobId={job.id}
            title="Setup players"
            description="Rename or merge players, or ignore ones that shouldn't be counted (a ref, a coach, a false detection)."
            saveButtonLabel="Save changes"
            confirmBeforeSave="Saving will reprocess the stats, dashboard, and video with these changes - it only redoes the quick consolidation step, not full tracking, but it'll take a moment."
            onSaved={onSaved}
          />
        </AccordionDetails>
      </Accordion>

      <Accordion variant="outlined" defaultExpanded sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="overline" color="text.secondary">
            Court
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <CourtSummary jobId={job.id} />
        </AccordionDetails>
      </Accordion>
    </>
  );
}
