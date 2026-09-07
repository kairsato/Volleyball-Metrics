"""Shared plumbing used across this project's ML tooling - currently just
the Roboflow-download mechanics datasetGather.py's per-model dataset
classes (BallDatasets, ActionDatasets, CourtDatasets) all call into, so
none of them need to reimplement "pull a Roboflow Universe project down
in YOLO format". A future cross-model helper (shared augmentation,
common eval metrics, etc.) belongs here too, alongside this one.
"""

from pathlib import Path
from typing import Optional


def download_roboflow_project(
    api_key: str,
    workspace: str,
    project_slug: str,
    dest: Path,
    version: Optional[int] = None,
    model_format: str = "yolov8",
) -> Path:
    """Pulls the given Roboflow project's latest published version (or a
    pinned one) into `dest`. Skips the download entirely if `dest` already
    exists, since a Roboflow export is static per version - there's
    nothing to refresh."""
    from roboflow import Roboflow

    if dest.exists():
        print(f"[{dest.name}] already downloaded at {dest}, skipping.")
        return dest

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)

    if version is None:
        versions = project.versions()
        if not versions:
            raise RuntimeError(f"Roboflow project {workspace}/{project_slug} has no published versions.")
        version_id = versions[-1].version
        # Roboflow's SDK has returned this as either a bare int or a
        # "workspace/project/N" string across versions - handle both.
        version = int(str(version_id).rsplit("/", 1)[-1])

    dest.parent.mkdir(parents=True, exist_ok=True)
    project.version(version).download(model_format=model_format, location=str(dest))
    return dest
