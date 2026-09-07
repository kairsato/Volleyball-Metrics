"""Benchmarks candidate ball-detector architectures against this project's
merged dataset (MachineLearning/dataset_ballDetection/merged, see
datasetGather.BallDatasets)
and a fixed match clip, so the best-performing architecture can replace the
production ball detector (currently a YOLO11m fine-tune, see
BallDetection/ballDetection.py's module docstring).

Fine-tunes each of YOLO26x, YOLO11x, YOLO12x, and RT-DETR-x - the largest
stock, COCO-pretrained checkpoint of each architecture ultralytics ships -
on the merged ball dataset for up to EPOCHS epochs with PATIENCE early
stopping, then for each trained model:
  - validates against the dataset's own val split (precision/recall/mAP),
  - runs it frame-by-frame over BENCHMARK_VIDEO to measure real-world
    detection rate, tracking smoothness, and inference speed,
  - writes an annotated tracking video so the results can be watched, not
    just read as numbers.

Every architecture here (including RT-DETR) trains and evaluates through
the same ultralytics YOLO/RTDETR Model API, so one code path covers all
four - deliberately excludes RT-DETRv2, D-FINE, and RF-DETR, which would
each need a separate package, a COCO-format copy of the dataset, and their
own inference wrapper.

This is long-running and meant to be started once and left alone (possibly
for many hours - four x-scale architectures at up to 300 epochs each on a
single GPU, run sequentially): every stage is idempotent, so interrupting
the script and rerunning it skips whatever already finished (an existing
weights/best.pt skips that model's training; an existing report skips its
video pass) instead of starting over. Progress is both printed and
appended to benchmark_results/benchmark_log.txt so it can be checked on
without keeping a terminal open.

Requires a torch build with kernels for the local GPU - this project's dev
box has an RTX 5090 (sm_120/Blackwell), which the stock
`pip install torch` (cu126) does not support; that was fixed here via:
    pip install --force-reinstall --no-deps torch==2.13.0+cu130 torchvision==0.28.0+cu130 \
        --index-url https://download.pytorch.org/whl/cu130
check_cuda() below re-verifies this on every run so a future environment
regression fails fast with that fix spelled out, instead of silently
training on CPU.

Usage (from Backend/Analysis/MachineLearning/):
    python model_benchmark.py
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ML_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ML_DIR / "models"

# --- Configuration ----------------------------------------------------------
DATA_YAML = ML_DIR / "dataset_ballDetection" / "merged" / "data.yaml"
BENCHMARK_VIDEO = Path(r"C:\Users\Kai\Documents\VolleyballArea\verycut - Copy.mp4")
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_results"

EPOCHS = 300
PATIENCE = 50
TRAIN_IMGSZ = 640
DEVICE = 0

# Video pass runs at 960/0.25 to match ballDetection.py's own
# ACQUISITION_IMG_SIZE/CONFIDENCE_THRESHOLD full-frame settings, so these
# stats reflect how each model would actually behave in production rather
# than at its training resolution.
VIDEO_IMGSZ = 960
VIDEO_CONF = 0.25
TRAIL_LEN = 15  # frames of trail drawn on the annotated output video


@dataclass
class ModelSpec:
    name: str
    weights: str
    kind: str  # "yolo" or "rtdetr" - which ultralytics Model subclass loads it


MODELS = [
    ModelSpec("yolo26x", str(MODELS_DIR / "yolo26x.pt"), "yolo"),
    ModelSpec("yolo11x", str(MODELS_DIR / "yolo11x.pt"), "yolo"),
    ModelSpec("yolo12x", str(MODELS_DIR / "yolo12x.pt"), "yolo"),
    ModelSpec("rtdetr-x", str(MODELS_DIR / "rtdetr-x.pt"), "rtdetr"),
]
# -----------------------------------------------------------------------------


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "benchmark_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available - training four x-scale architectures on "
            "CPU would take far too long to be practical. See this file's "
            "module docstring for the torch reinstall that fixed this on "
            "this project's dev box."
        )
    try:
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    except RuntimeError as e:
        raise SystemExit(
            f"CUDA reports available but a GPU op failed ({e}). Your torch "
            "build likely lacks kernels for this GPU - reinstall it from "
            "the matching cuXXX wheel index at "
            "https://download.pytorch.org/whl/ (this project's dev box "
            "needed torch==2.13.0+cu130 / torchvision==0.28.0+cu130 for its "
            "RTX 5090)."
        ) from e


def _model_class(kind: str):
    from ultralytics import RTDETR, YOLO

    return RTDETR if kind == "rtdetr" else YOLO


def train_one(spec: ModelSpec) -> Path:
    """Fine-tunes spec's stock COCO-pretrained checkpoint on the merged ball
    dataset. Idempotent: reuses an already-trained best.pt from a prior run
    of this script instead of retraining."""
    run_dir = OUTPUT_DIR / "training" / spec.name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        log(f"[{spec.name}] already trained, reusing {best}")
        return best

    log(f"[{spec.name}] training from {spec.weights} for up to {EPOCHS} epochs (patience={PATIENCE})...")
    model = _model_class(spec.kind)(spec.weights)
    t0 = time.perf_counter()
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=TRAIN_IMGSZ,
        device=DEVICE,
        batch=-1,  # autobatch - use as much of the 32GB card as fits
        cache="ram",  # merged dataset is ~3.5k images, fits comfortably in RAM
        seed=0,
        project=str(OUTPUT_DIR / "training"),
        name=spec.name,
        exist_ok=True,
        plots=True,
    )
    elapsed_hours = (time.perf_counter() - t0) / 3600
    if not best.exists():
        raise RuntimeError(f"[{spec.name}] training finished but no best.pt found at {best}")
    log(f"[{spec.name}] training finished in {elapsed_hours:.2f}h -> {best}")
    return best


def validate_one(spec: ModelSpec, best: Path) -> dict:
    model = _model_class(spec.kind)(str(best))
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        device=DEVICE,
        verbose=False,
        project=str(OUTPUT_DIR / "validation"),
        name=spec.name,
        exist_ok=True,
    )
    rd = metrics.results_dict
    return {
        "precision": rd.get("metrics/precision(B)"),
        "recall": rd.get("metrics/recall(B)"),
        "mAP50": rd.get("metrics/mAP50(B)"),
        "mAP50-95": rd.get("metrics/mAP50-95(B)"),
        "fitness": rd.get("fitness"),
    }


def summarize_detections(
    detections: list[Optional[tuple[float, float, float]]],
    infer_times: list[float],
    frame_w: int,
    frame_h: int,
) -> dict:
    """Turns a per-frame list of (cx, cy, conf)-or-None into detection-rate,
    gap, jitter, and speed stats. Jitter (frame-to-frame center displacement
    between two *consecutive* detected frames, normalized by frame diagonal)
    conflates real ball motion with model noise, but since every model sees
    the same video and the same true motion, a higher relative jitter here
    means a noisier/less stable track (spurious jumps from false positives
    or from a track re-acquiring after a miss), not a faster ball."""
    total = len(detections)
    detected_idxs = [i for i, d in enumerate(detections) if d is not None]
    detection_rate = len(detected_idxs) / total if total else 0.0
    avg_conf = sum(detections[i][2] for i in detected_idxs) / len(detected_idxs) if detected_idxs else 0.0

    gaps: list[int] = []
    run = 0
    for d in detections:
        if d is None:
            run += 1
        else:
            if run:
                gaps.append(run)
            run = 0
    if run:
        gaps.append(run)

    diag = math.hypot(frame_w, frame_h)
    jitter = []
    for i in range(1, total):
        a, b = detections[i - 1], detections[i]
        if a is not None and b is not None:
            jitter.append(math.hypot(b[0] - a[0], b[1] - a[1]) / diag)
    jitter.sort()
    jitter_mean = sum(jitter) / len(jitter) if jitter else 0.0
    jitter_p95 = jitter[int(0.95 * (len(jitter) - 1))] if jitter else 0.0

    infer_fps = 1.0 / (sum(infer_times) / len(infer_times)) if infer_times else 0.0

    return {
        "total_frames": total,
        "detected_frames": len(detected_idxs),
        "detection_rate": round(detection_rate, 4),
        "avg_confidence": round(avg_conf, 4),
        "gap_count": len(gaps),
        "max_gap_frames": max(gaps) if gaps else 0,
        "jitter_mean_norm": round(jitter_mean, 5),
        "jitter_p95_norm": round(jitter_p95, 5),
        "infer_fps": round(infer_fps, 2),
    }


def run_video_benchmark(spec: ModelSpec, best: Path, out_video: Path) -> dict:
    """Runs best over BENCHMARK_VIDEO frame-by-frame (highest-confidence box
    per frame, no cross-frame smoothing/interpolation - these are the raw
    per-model numbers), writes an annotated copy with a trailing motion
    trail, and returns summarize_detections()'s stats."""
    import cv2

    model = _model_class(spec.kind)(str(best))

    cap = cv2.VideoCapture(str(BENCHMARK_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open benchmark video: {BENCHMARK_VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detections: list[Optional[tuple[float, float, float]]] = []
    infer_times: list[float] = []
    trail: deque[tuple[int, int]] = deque(maxlen=TRAIL_LEN)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.perf_counter()
            result = model.predict(frame, imgsz=VIDEO_IMGSZ, conf=VIDEO_CONF, device=DEVICE, verbose=False)[0]
            infer_times.append(time.perf_counter() - t0)

            best_det = None
            if len(result.boxes):
                confs = result.boxes.conf.tolist()
                i = max(range(len(confs)), key=lambda k: confs[k])
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                conf = confs[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                best_det = (cx, cy, conf)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{conf:.2f}", (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
                trail.append((int(cx), int(cy)))

            for j in range(1, len(trail)):
                cv2.line(frame, trail[j - 1], trail[j], (0, 165, 255), 2)

            detections.append(best_det)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return summarize_detections(detections, infer_times, w, h)


def run_model(spec: ModelSpec) -> dict:
    report_path = OUTPUT_DIR / "reports" / f"{spec.name}.json"
    if report_path.exists():
        log(f"[{spec.name}] already has a report, skipping entirely: {report_path}")
        return json.loads(report_path.read_text(encoding="utf-8"))

    report: dict = {"name": spec.name, "weights": spec.weights, "kind": spec.kind}
    try:
        t0 = time.perf_counter()
        best = train_one(spec)
        report["train_hours"] = round((time.perf_counter() - t0) / 3600, 3)
        report["best_checkpoint"] = str(best)

        log(f"[{spec.name}] validating on {DATA_YAML}...")
        report["val"] = validate_one(spec, best)

        out_video = OUTPUT_DIR / "videos" / f"{spec.name}_tracked.mp4"
        log(f"[{spec.name}] running video benchmark against {BENCHMARK_VIDEO.name}...")
        report["video"] = run_video_benchmark(spec, best, out_video)
        report["video_path"] = str(out_video)
    except Exception as e:  # noqa: BLE001 - one model's failure must not abort the rest
        log(f"[{spec.name}] FAILED: {e!r}")
        report["error"] = repr(e)

    OUTPUT_DIR.joinpath("reports").mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_comparison_table(reports: list[dict]) -> str:
    def sort_key(r: dict) -> float:
        v = r.get("val", {}).get("mAP50-95")
        return v if isinstance(v, (int, float)) else -1.0

    rows = sorted(reports, key=sort_key, reverse=True)

    headers = [
        "model", "mAP50-95", "mAP50", "precision", "recall",
        "det.rate", "gaps", "max_gap", "jitter", "infer FPS", "train h",
    ]
    lines = [" | ".join(headers), " | ".join("-" * len(h) for h in headers)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r['name']} | ERROR: {r['error']}")
            continue
        v, vid = r.get("val", {}), r.get("video", {})
        lines.append(" | ".join([
            r["name"],
            f"{v.get('mAP50-95', 0):.3f}",
            f"{v.get('mAP50', 0):.3f}",
            f"{v.get('precision', 0):.3f}",
            f"{v.get('recall', 0):.3f}",
            f"{vid.get('detection_rate', 0):.3f}",
            str(vid.get("gap_count", "-")),
            str(vid.get("max_gap_frames", "-")),
            f"{vid.get('jitter_mean_norm', 0):.4f}",
            f"{vid.get('infer_fps', 0):.1f}",
            f"{r.get('train_hours', 0):.2f}",
        ]))
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_cuda()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"Merged dataset not found at {DATA_YAML}. Run "
            "mainTrainingModels.py's ball_detector_datasets() first "
            "to download and merge the ball-detector data sources."
        )
    if not BENCHMARK_VIDEO.exists():
        raise SystemExit(f"Benchmark video not found at {BENCHMARK_VIDEO}")

    log(f"Benchmarking {len(MODELS)} models: {[m.name for m in MODELS]}")
    reports = [run_model(spec) for spec in MODELS]

    (OUTPUT_DIR / "comparison_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    table = format_comparison_table(reports)
    (OUTPUT_DIR / "comparison_report.md").write_text(table, encoding="utf-8")

    log("All models done. Comparison:\n" + table)


if __name__ == "__main__":
    main()
