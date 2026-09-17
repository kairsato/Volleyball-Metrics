import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import MenuItem from "@mui/material/MenuItem";
import Popover from "@mui/material/Popover";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import ClearIcon from "@mui/icons-material/Clear";
import { api } from "../lib/api";
import type { Job, TeamEntry } from "../lib/types";
import { PageHeader } from "../components/PageHeader";
import { GameGrid } from "../components/GameGrid";

// Local calendar date (YYYY-MM-DD) rather than a UTC one, so "uploaded
// today" matches what the user's own clock says today is - and so this
// compares lexicographically the same way an <input type="date"> value
// does, without pulling in a date library for what's just a string compare.
function toDateKey(iso: string): string {
  const d = new Date(iso);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function matchesDateFilter(job: Job, from: string, to: string): boolean {
  if (!from && !to) return true;
  const jobDate = toDateKey(job.created_at);
  if (from && jobDate < from) return false;
  if (to && jobDate > to) return false;
  return true;
}

// A YYYY-MM-DD date key formatted for display without going through
// Date parsing (new Date("YYYY-MM-DD") parses as UTC midnight, which reads
// as the previous day in any timezone behind UTC) - splitting the key
// itself sidesteps that entirely.
function formatDateKey(dateKey: string): string {
  const [year, month, day] = dateKey.split("-");
  return `${month}/${day}/${year}`;
}

function formatRangeLabel(from: string, to: string): string {
  if (from && to) return from === to ? formatDateKey(from) : `${formatDateKey(from)} – ${formatDateKey(to)}`;
  if (from) return `From ${formatDateKey(from)}`;
  if (to) return `Until ${formatDateKey(to)}`;
  return "";
}

// A single date box with an "on" operator (and Before/After alongside it)
// was more control than this ever needed - a plain From/To range covers
// every one of those (leave From blank for "before", leave To blank for
// "after", set both the same for "on") with one less control to explain.
// It renders as one field (a read-only summary of whatever's set, e.g.
// "9/1/2026 – 9/10/2026") that opens a popover with the actual From/To
// inputs on click, rather than two boxes sitting in the toolbar at once.
function UploadedFilter({
  from,
  to,
  onFromChange,
  onToChange,
}: {
  from: string;
  to: string;
  onFromChange: (value: string) => void;
  onToChange: (value: string) => void;
}) {
  const anchorRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const hasValue = Boolean(from || to);

  function clear(event: MouseEvent) {
    event.stopPropagation();
    onFromChange("");
    onToChange("");
  }

  return (
    <>
      <TextField
        ref={anchorRef}
        size="small"
        label="Uploaded"
        placeholder="Any date"
        value={formatRangeLabel(from, to)}
        onClick={() => setOpen(true)}
        sx={{ width: { xs: "100%", sm: 220 } }}
        slotProps={{
          input: {
            readOnly: true,
            sx: { cursor: "pointer" },
            endAdornment: hasValue && (
              <InputAdornment position="end">
                <IconButton size="small" edge="end" aria-label="Clear date range" onClick={clear}>
                  <ClearIcon fontSize="small" />
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
      />
      <Popover
        open={open}
        anchorEl={anchorRef.current}
        onClose={() => setOpen(false)}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Stack direction="row" spacing={1.5} sx={{ p: 2 }}>
          <TextField
            type="date"
            size="small"
            label="From"
            value={from}
            onChange={(event) => onFromChange(event.target.value)}
            sx={{ width: 170 }}
            slotProps={{ inputLabel: { shrink: true } }}
          />
          <TextField
            type="date"
            size="small"
            label="To"
            value={to}
            onChange={(event) => onToChange(event.target.value)}
            sx={{ width: 170 }}
            slotProps={{ inputLabel: { shrink: true } }}
          />
        </Stack>
      </Popover>
    </>
  );
}

// "all" (no filter), or "<teamId>:wins" / "<teamId>:losses" for a specific
// registered team's record - there's no single well-defined "wins/losses"
// without picking a team first, since any two games on this list can be a
// completely different matchup.
type TeamFilter = "all" | `${string}:wins` | `${string}:losses`;

function matchesTeamFilter(job: Job, filter: TeamFilter, teamsById: Map<string, TeamEntry>): boolean {
  if (filter === "all") return true;
  const [teamId, mode] = filter.split(":") as [string, "wins" | "losses"];
  const teamName = teamsById.get(teamId)?.name;
  if (!teamName) return false;
  return mode === "wins" ? job.winner_team_name === teamName : job.loser_team_name === teamName;
}

type SortField = "date" | "wins" | "losses" | "incomplete";
type SortBy = `${SortField}_${"asc" | "desc"}`;

const SORT_OPTIONS: { value: SortBy; label: string }[] = [
  { value: "date_desc", label: "Date (Newest)" },
  { value: "date_asc", label: "Date (Oldest)" },
  { value: "wins_desc", label: "Wins (Most)" },
  { value: "wins_asc", label: "Wins (Fewest)" },
  { value: "losses_desc", label: "Losses (Most)" },
  { value: "losses_asc", label: "Losses (Fewest)" },
  { value: "incomplete_desc", label: "Incomplete first" },
  { value: "incomplete_asc", label: "Complete first" },
];

function isIncomplete(job: Job): boolean {
  return job.status !== "complete" || job.needs_player_id || job.needs_scoring_review;
}

// Wins/Losses sort follows whatever the Team filter is currently set to -
// once a team is selected, its wins (or losses) rank first; with "All
// teams" selected there's no specific team to rank by, so it falls back to
// "has a decided winner yet" as the closest analogue. Every field has an
// asc/desc pair (e.g. "Date (Newest)"/"Date (Oldest)") - direction just
// flips which end of the ranking comes first; ties always fall back to
// newest-first, regardless of the chosen direction.
function sortJobs(jobs: Job[], sortBy: SortBy, teamFilter: TeamFilter, teamsById: Map<string, TeamEntry>): Job[] {
  const [field, direction] = sortBy.split("_") as [SortField, "asc" | "desc"];
  const sign = direction === "asc" ? -1 : 1;
  const byDateDesc = (a: Job, b: Job) => b.created_at.localeCompare(a.created_at);

  if (field === "incomplete") {
    return [...jobs].sort((a, b) => sign * (Number(isIncomplete(b)) - Number(isIncomplete(a))) || byDateDesc(a, b));
  }

  if (field === "wins" || field === "losses") {
    const teamName = teamFilter === "all" ? null : teamsById.get(teamFilter.split(":")[0])?.name ?? null;
    const rank = (job: Job): number => {
      if (teamName) {
        const isMatch = field === "wins" ? job.winner_team_name === teamName : job.loser_team_name === teamName;
        return isMatch ? 1 : 0;
      }
      const decided = job.winner_team_name != null;
      return (field === "wins") === decided ? 1 : 0;
    };
    return [...jobs].sort((a, b) => sign * (rank(b) - rank(a)) || byDateDesc(a, b));
  }

  return [...jobs].sort((a, b) => sign * byDateDesc(a, b));
}

interface GamesPageProps {
  jobs: Job[];
  onSelectJob: (jobId: string) => void;
  onAddVideo: () => void;
  onDeleteJob: (jobId: string) => void;
}

export function GamesPage({ jobs, onSelectJob, onAddVideo, onDeleteJob }: GamesPageProps) {
  const [teams, setTeams] = useState<TeamEntry[]>([]);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [teamFilter, setTeamFilter] = useState<TeamFilter>("all");
  const [sortBy, setSortBy] = useState<SortBy>("date_desc");

  useEffect(() => {
    api
      .getTeams()
      .then((res) => setTeams(res.teams))
      .catch(() => undefined);
  }, []);

  const teamsById = useMemo(() => new Map(teams.map((t) => [t.id, t])), [teams]);

  const filteredJobs = useMemo(() => {
    const matching = jobs.filter(
      (job) => matchesDateFilter(job, dateFrom, dateTo) && matchesTeamFilter(job, teamFilter, teamsById),
    );
    return sortJobs(matching, sortBy, teamFilter, teamsById);
  }, [jobs, dateFrom, dateTo, teamFilter, sortBy, teamsById]);

  return (
    <Box>
      <PageHeader title="Games" addLabel="Add game" onAdd={onAddVideo}>
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
          <UploadedFilter from={dateFrom} to={dateTo} onFromChange={setDateFrom} onToChange={setDateTo} />

          {teams.length > 0 && (
            <TextField
              select
              size="small"
              label="Team"
              value={teamFilter}
              onChange={(event) => setTeamFilter(event.target.value as TeamFilter)}
              sx={{ width: { xs: "100%", sm: 220 } }}
            >
              <MenuItem value="all">All teams</MenuItem>
              {teams.map((team) => [
                <MenuItem key={`${team.id}:wins`} value={`${team.id}:wins`}>
                  {team.name} — Wins
                </MenuItem>,
                <MenuItem key={`${team.id}:losses`} value={`${team.id}:losses`}>
                  {team.name} — Losses
                </MenuItem>,
              ])}
            </TextField>
          )}

          <TextField
            select
            size="small"
            label="Sort by"
            value={sortBy}
            onChange={(event) => setSortBy(event.target.value as SortBy)}
            sx={{ width: { xs: "100%", sm: 190 } }}
          >
            {SORT_OPTIONS.map((opt) => (
              <MenuItem key={opt.value} value={opt.value}>
                {opt.label}
              </MenuItem>
            ))}
          </TextField>
        </Stack>
      </PageHeader>

      {jobs.length === 0 ? (
        <Box sx={{ color: "text.secondary" }}>No games yet - use "Add game" above to get started.</Box>
      ) : filteredJobs.length === 0 ? (
        <Box sx={{ color: "text.secondary" }}>No games match your filters.</Box>
      ) : (
        <GameGrid
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
