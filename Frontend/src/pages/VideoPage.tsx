import { useEffect } from "react";
import { Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { JobWorkspace } from "../components/JobWorkspace";
import type { Job } from "../lib/types";

const SCROLLBAR_HIDDEN_CLASS = "vva-scrollbar-hidden";

// Hides (not disables) the page scrollbar while this page is mounted - the
// video page's layout is sized to fit the viewport, but sub-pixel rounding
// across browsers/zoom levels can still leave a stray pixel or two of real
// overflow. That's not worth chasing further; a barely-there scrollbar
// flickering in and out is worse than just not showing one here.
function useHiddenScrollbar() {
  useEffect(() => {
    let styleEl = document.getElementById("vva-scrollbar-hidden-style") as HTMLStyleElement | null;
    if (!styleEl) {
      styleEl = document.createElement("style");
      styleEl.id = "vva-scrollbar-hidden-style";
      styleEl.textContent = `
        .${SCROLLBAR_HIDDEN_CLASS} {
          scrollbar-width: none;
          -ms-overflow-style: none;
        }
        .${SCROLLBAR_HIDDEN_CLASS}::-webkit-scrollbar {
          display: none;
        }
      `;
      document.head.appendChild(styleEl);
    }

    document.documentElement.classList.add(SCROLLBAR_HIDDEN_CLASS);
    return () => {
      document.documentElement.classList.remove(SCROLLBAR_HIDDEN_CLASS);
    };
  }, []);
}

interface VideoPageProps {
  onJobUpdated: (job: Job) => void;
}

export function VideoPage({ onJobUpdated }: VideoPageProps) {
  useHiddenScrollbar();

  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const jobId = searchParams.get("job");

  if (!jobId) return <Navigate to="/videos" replace />;

  return (
    <JobWorkspace
      key={jobId}
      jobId={jobId}
      onJobUpdated={onJobUpdated}
      onBackToDashboard={() => navigate("/videos")}
    />
  );
}
