import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { JobWorkspace } from "../components/JobWorkspace";
import type { Job } from "../lib/types";

interface ResultsPageProps {
  onJobUpdated: (job: Job) => void;
}

export function ResultsPage({ onJobUpdated }: ResultsPageProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");

  if (!jobId) return <Navigate to="/playlist" replace />;

  return (
    <JobWorkspace
      key={jobId}
      jobId={jobId}
      onJobUpdated={onJobUpdated}
      onBackToDashboard={() => navigate("/playlist")}
    />
  );
}
