import { useMemo, useState } from "react";
import Box from "@mui/material/Box";
import MenuItem from "@mui/material/MenuItem";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import type { Job } from "../lib/types";
import { PageHeader } from "../components/PageHeader";
import { VideoGrid } from "../components/VideoGrid";

type StatusFilter = "all" | "complete" | "in_progress" | "needs_attention" | "error";

const STATUS_FILTER_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "all", label: "All statuses" },
  { value: "complete", label: "Complete" },
  { value: "in_progress", label: "In progress" },
  { value: "needs_attention", label: "Needs attention" },
  { value: "error", label: "Error/cancelled" },
];

function matchesStatusFilter(job: Job, filter: StatusFilter): boolean {
  switch (filter) {
    case "all":
      return true;
    case "complete":
      return job.status === "complete";
    case "in_progress":
      return ["uploaded", "processing", "awaiting_player_review", "finalizing"].includes(job.status);
    case "needs_attention":
      return job.status === "complete" && (job.needs_player_id || job.needs_scoring_review);
    case "error":
      return job.status === "error" || job.status === "cancelled";
  }
}

type DateFilterMode = "any" | "exact" | "before" | "after";

const DATE_FILTER_OPTIONS: { value: DateFilterMode; label: string }[] = [
  { value: "any", label: "Any date" },
  { value: "exact", label: "On" },
  { value: "before", label: "Before" },
  { value: "after", label: "After" },
];

// Local calendar date (YYYY-MM-DD) rather than a UTC one, so "uploaded
// today" matches what the user's own clock says today is - and so this
// compares lexicographically the same way an <input type="date"> value
// does, without pulling in a date library for what's just a string compare.
function toDateKey(iso: string): string {
  const d = new Date(iso);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function matchesDateFilter(job: Job, mode: DateFilterMode, dateValue: string): boolean {
  if (mode === "any" || !dateValue) return true;
  const jobDate = toDateKey(job.created_at);
  switch (mode) {
    case "exact":
      return jobDate === dateValue;
    case "before":
      return jobDate < dateValue;
    case "after":
      return jobDate > dateValue;
  }
}

interface VideosPageProps {
  jobs: Job[];
  onSelectJob: (jobId: string) => void;
  onAddVideo: () => void;
  onDeleteJob: (jobId: string) => void;
}

export function VideosPage({ jobs, onSelectJob, onAddVideo, onDeleteJob }: VideosPageProps) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [dateFilterMode, setDateFilterMode] = useState<DateFilterMode>("any");
  const [dateFilterValue, setDateFilterValue] = useState("");

  const filteredJobs = useMemo(() => {
    const query = search.trim().toLowerCase();
    return jobs.filter(
      (job) =>
        matchesStatusFilter(job, statusFilter) &&
        matchesDateFilter(job, dateFilterMode, dateFilterValue) &&
        (query === "" || job.original_filename.toLowerCase().includes(query)),
    );
  }, [jobs, search, statusFilter, dateFilterMode, dateFilterValue]);

  return (
    <Box>
      <PageHeader title="Videos" addLabel="Add video" onAdd={onAddVideo}>
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
          <TextField
            size="small"
            placeholder="Search videos..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            sx={{ width: { xs: "100%", sm: 240 } }}
          />
          <TextField
            select
            size="small"
            label="Status"
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value as StatusFilter)}
            sx={{ width: { xs: "100%", sm: 180 } }}
          >
            {STATUS_FILTER_OPTIONS.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>
                {opt.label}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            select
            size="small"
            label="Uploaded"
            value={dateFilterMode}
            onChange={(event) => setDateFilterMode(event.target.value as DateFilterMode)}
            sx={{ width: { xs: "100%", sm: 130 } }}
          >
            {DATE_FILTER_OPTIONS.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>
                {opt.label}
              </MenuItem>
            ))}
          </TextField>
          {dateFilterMode !== "any" && (
            <TextField
              type="date"
              size="small"
              value={dateFilterValue}
              onChange={(event) => setDateFilterValue(event.target.value)}
              sx={{ width: { xs: "100%", sm: 170 } }}
            />
          )}
        </Stack>
      </PageHeader>

      {jobs.length === 0 ? (
        <Box sx={{ color: "text.secondary" }}>No videos yet - use "Add video" above to get started.</Box>
      ) : filteredJobs.length === 0 ? (
        <Box sx={{ color: "text.secondary" }}>No videos match your search/filter.</Box>
      ) : (
        <VideoGrid
          jobs={filteredJobs}
          onSelectJob={onSelectJob}
          onDeleteJob={onDeleteJob}
          onAddVideo={onAddVideo}
          showAddTile={false}
        />
      )}
    </Box>
  );
}
