"""
Fine-tunes the game-state VideoMAE classifier (gameStatusDetection.py) on
our own labeled footage from label.py's game_status_labels*.json exports.

Two phases, run separately so a crash/interrupt during training doesn't
force redoing the (slow, CPU-bound) video decode:

    python trainGameStatus.py cache
    python trainGameStatus.py train

Dataset layout expected: <dataset_root>/<video_id>/<video file> alongside a
game_status_labels*.json in the same folder - exactly what label.py writes.
`cache` decodes every video once, and for each label.py-style chunk saves a
compact JPEG (the same CLIP_NUM_FRAMES frames, at the same buffered
resolution, that a live inference call would classify) plus its
ground-truth label. `train` fine-tunes from BASE_MODEL_DIR against that
cache and saves the best checkpoint to CUSTOM_MODEL_DIR, which
gameStatusDetection.py prefers automatically once it exists.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from gameStatusDetection import (
    BASE_MODEL_DIR,
    BUFFER_SHORTEST_EDGE,
    CLIP_NUM_FRAMES,
    CUSTOM_MODEL_DIR,
    STRIDE_FRAMES,
    WINDOW_FRAMES,
    _patch_legacy_attention_bias,
    _resize_shortest_edge,
)

DEFAULT_DATASET_ROOTS = [
    Path(r"C:\Users\Kai\Documents\Volleyball Footage\VolleyballGameClassificationDatasets\PersonalVolleyballDataset"),
    Path(r"C:\Users\Kai\Documents\Volleyball Footage\VolleyballGameClassificationDatasets\VolleyballAnalyticsDatasetConverted"),
]
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

CACHE_DIR = Path(__file__).resolve().parents[1] / "MachineLearning" / "dataset_gameStatusDetection"
MANIFEST_NAME = "manifest.json"

# One training example every this many frames - a multiple of STRIDE_FRAMES
# so each example still lines up with exactly one of inference's real
# STRIDE_FRAMES-wide label chunks. Consecutive chunks overlap heavily in
# their WINDOW_FRAMES of context anyway (64 vs. 16), so caching literally
# every chunk would mostly buy redundancy, not signal.
TRAIN_SAMPLE_STRIDE = STRIDE_FRAMES * 2

# Every VAL_HOLDOUT_MODULOth video (by numeric id, per dataset) is held out
# for validation instead of being trained on - whole videos, not individual
# windows, since neighbouring windows from the same video are near-
# duplicates and would leak across a random split.
VAL_HOLDOUT_MODULO = 5
VAL_HOLDOUT_REMAINDER = 2

JPEG_QUALITY = 90

# Matches the base checkpoint's config.json image_size - frames are cached
# at BUFFER_SHORTEST_EDGE=256 (see gameStatusDetection.py), a bit larger
# than this, specifically so training can take a randomly-placed crop
# instead of always the same deterministic center crop inference uses.
CROP_SIZE = 224


def _numeric_key(video_id: str):
    try:
        return (0, int(video_id))
    except ValueError:
        return (1, video_id)


def _find_dataset_videos(dataset_root: Path):
    videos = []
    for video_dir in sorted(dataset_root.iterdir()):
        if not video_dir.is_dir():
            continue
        video_path = next((p for ext in VIDEO_EXTENSIONS for p in video_dir.glob(f"*{ext}")), None)
        labels_path = next(iter(video_dir.glob("game_status_labels*.json")), None)
        if video_path is None or labels_path is None:
            continue
        videos.append((video_dir.name, video_path, labels_path))
    return videos


def _split_train_val(dataset_roots):
    """Returns (train, val) lists of (dataset_name, video_id, video_path, labels_path)."""
    train, val = [], []
    for root in dataset_roots:
        ordered = sorted(_find_dataset_videos(root), key=lambda v: _numeric_key(v[0]))
        for i, (video_id, video_path, labels_path) in enumerate(ordered):
            entry = (root.name, video_id, video_path, labels_path)
            if i % VAL_HOLDOUT_MODULO == VAL_HOLDOUT_REMAINDER:
                val.append(entry)
            else:
                train.append(entry)
    return train, val


def _majority_state(segments, start_frame, end_frame):
    overlap = Counter()
    for seg in segments:
        lo, hi = max(seg["start_frame"], start_frame), min(seg["end_frame"], end_frame)
        if hi >= lo:
            overlap[seg["state"]] += hi - lo + 1
    if not overlap:
        return None
    return overlap.most_common(1)[0][0]


def _cache_video(dataset_name, video_id, video_path, labels_path, label2id, cache_dir: Path):
    out_dir = cache_dir / dataset_name / video_id
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        print(f"skip {dataset_name}/{video_id}: already cached")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_path) as f:
        segments = json.load(f)["segments"]

    cap = cv2.VideoCapture(str(video_path))
    buffer = []
    frame_idx = 0
    entries = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buffer.append(_resize_shortest_edge(frame, BUFFER_SHORTEST_EDGE))
        if len(buffer) > WINDOW_FRAMES:
            buffer.pop(0)
        frame_idx += 1

        if frame_idx % TRAIN_SAMPLE_STRIDE != 0:
            continue

        state = _majority_state(segments, frame_idx - STRIDE_FRAMES, frame_idx - 1)
        if state is None or state not in label2id:
            continue

        indices = np.linspace(0, len(buffer) - 1, CLIP_NUM_FRAMES).round().astype(int)
        clip = [buffer[i] for i in indices]
        stacked = np.concatenate(clip, axis=0)  # (CLIP_NUM_FRAMES * h, w, 3) BGR
        ok_enc, encoded = cv2.imencode(".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok_enc:
            continue

        clip_path = out_dir / f"{frame_idx}.jpg"
        clip_path.write_bytes(encoded.tobytes())
        entries.append({"path": str(clip_path.relative_to(cache_dir)), "label": state, "frame_h": clip[0].shape[0]})

    cap.release()
    manifest_path.write_text(json.dumps(entries))
    print(f"cached {dataset_name}/{video_id}: {len(entries)} clips")


def build_cache(args):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(BASE_MODEL_DIR / "config.json") as f:
        label2id = json.load(f)["label2id"]

    jobs = []
    for root in args.dataset_roots:
        for video_id, video_path, labels_path in _find_dataset_videos(root):
            jobs.append((root.name, video_id, video_path, labels_path, label2id, args.cache_dir))

    print(f"caching {len(jobs)} video(s) with {args.workers} worker(s)...")
    if args.workers <= 1:
        for job in jobs:
            _cache_video(*job)
    else:
        with mp.Pool(processes=args.workers) as pool:
            pool.starmap(_cache_video, jobs)


def _load_manifest_entries(video_list, cache_dir: Path):
    entries = []
    for dataset_name, video_id, _video_path, _labels_path in video_list:
        manifest_path = cache_dir / dataset_name / video_id / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No cache for {dataset_name}/{video_id} - run `trainGameStatus.py cache` first.")
        entries.extend(json.loads(manifest_path.read_text()))
    return entries


class ClipDataset(Dataset):
    """augment=True (training only) takes a randomly-placed crop and a
    coin-flip horizontal mirror - applied identically across all
    CLIP_NUM_FRAMES of a clip so the motion stays coherent - instead of
    inference's fixed center crop, plus feeds the model dropout instead of
    the base checkpoint's dropout-disabled config. Both target the
    overfitting the first two training runs showed (smoothly falling train
    loss, wildly oscillating val_acc): the safety valve of always seeing
    the exact same 224x224 pixels for a given clip, epoch after epoch, on a
    dataset this size (~15k clips) gives the model an easy way to
    memorise rather than generalise. Validation always uses augment=False,
    matching inference exactly, so it stays an unbiased estimate."""

    def __init__(self, entries, cache_dir: Path, processor, label2id, augment):
        self.entries = entries
        self.cache_dir = cache_dir
        self.processor = processor
        self.label2id = label2id
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        stacked = cv2.imread(str(self.cache_dir / entry["path"]))
        h = entry["frame_h"]
        frames_bgr = [stacked[i * h:(i + 1) * h] for i in range(CLIP_NUM_FRAMES)]

        if self.augment:
            crop_h, crop_w = h, frames_bgr[0].shape[1]
            top = random.randint(0, crop_h - CROP_SIZE)
            left = random.randint(0, crop_w - CROP_SIZE)
            flip = random.random() < 0.5
            cropped = []
            for f in frames_bgr:
                f = f[top:top + CROP_SIZE, left:left + CROP_SIZE]
                cropped.append(f[:, ::-1] if flip else f)
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in cropped]
            inputs = self.processor(frames_rgb, do_resize=False, do_center_crop=False, return_tensors="pt")
        else:
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
            inputs = self.processor(frames_rgb, return_tensors="pt")

        pixel_values = inputs["pixel_values"][0]  # (num_frames, 3, 224, 224)
        return pixel_values, self.label2id[entry["label"]]


def _class_weights(entries, label2id, device):
    counts = Counter(e["label"] for e in entries)
    total = sum(counts.values())
    weights = torch.zeros(len(label2id))
    for label, idx in label2id.items():
        weights[idx] = total / (len(label2id) * max(1, counts.get(label, 0)))
    return weights.to(device)


@torch.no_grad()
def _evaluate(model, loader, device, label2id):
    id2label = {v: k for k, v in label2id.items()}
    model.eval()
    correct, total = 0, 0
    confusion = Counter()
    for pixel_values, labels in loader:
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for p, l in zip(preds.tolist(), labels.tolist()):
            confusion[f"true={id2label[l]} pred={id2label[p]}"] += 1
    return correct / max(1, total), dict(confusion)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = VideoMAEImageProcessor.from_pretrained(str(BASE_MODEL_DIR))
    # The base checkpoint was fine-tuned with dropout disabled entirely
    # (hidden_dropout_prob=attention_probs_dropout_prob=0.0 in its
    # config.json) - fine for a checkpoint that's done training, but a
    # weak spot for further fine-tuning on a small dataset. Re-enabling it
    # here only affects behaviour while model.train() is active; eval()
    # (validation, and inference in gameStatusDetection.py) zeroes dropout
    # regardless of this config value, so it's safe to bake into the saved
    # checkpoint too.
    model = VideoMAEForVideoClassification.from_pretrained(
        str(BASE_MODEL_DIR), hidden_dropout_prob=args.dropout, attention_probs_dropout_prob=args.dropout)
    _patch_legacy_attention_bias(model, BASE_MODEL_DIR)
    model = model.to(device)
    label2id = model.config.label2id

    train_videos, val_videos = _split_train_val(args.dataset_roots)
    print(f"train videos: {[f'{d}/{v}' for d, v, _, _ in train_videos]}")
    print(f"val videos:   {[f'{d}/{v}' for d, v, _, _ in val_videos]}")

    train_entries = _load_manifest_entries(train_videos, args.cache_dir)
    val_entries = _load_manifest_entries(val_videos, args.cache_dir)
    print(f"train clips: {len(train_entries)} {dict(Counter(e['label'] for e in train_entries))}")
    print(f"val clips:   {len(val_entries)} {dict(Counter(e['label'] for e in val_entries))}")

    train_loader = DataLoader(ClipDataset(train_entries, args.cache_dir, processor, label2id, augment=True),
                               batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(ClipDataset(val_entries, args.cache_dir, processor, label2id, augment=False),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_entries, label2id, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Linear warmup for the first `warmup_steps` steps (set by hand below,
    # not via a scheduler object - composing a per-step warmup with a
    # per-epoch plateau scheduler through SequentialLR isn't supported),
    # then held at args.lr and cut in half whenever val_acc stalls for
    # `plateau_patience` epochs. This - not a fixed-length schedule like
    # OneCycleLR - is what makes an open-ended, early-stopped run coherent:
    # we don't know in advance how many epochs it'll take.
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                                     patience=args.plateau_patience)

    # Seeded from whatever's already saved there (if anything), not -1.0 -
    # otherwise a fresh run's first epoch always looks like an
    # "improvement" over nothing and overwrites a better checkpoint from a
    # previous run before it's had a chance to catch back up.
    metrics_path = args.output_dir / "training_metrics.json"
    best_val_acc = json.loads(metrics_path.read_text())["val_acc"] if metrics_path.exists() else -1.0
    print(f"starting best_val_acc: {best_val_acc:.4f}" + (" (fresh)" if best_val_acc < 0 else " (from existing checkpoint)"))
    epochs_without_improvement = 0
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, (pixel_values, labels) in enumerate(train_loader):
            global_step += 1
            if global_step <= args.warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * global_step / args.warmup_steps

            pixel_values, labels = pixel_values.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = criterion(model(pixel_values=pixel_values).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running_loss / (step + 1):.4f}")

        # Saved before validation runs, so a crash during eval (a GPU fault,
        # not our code, took out a run mid-validation once already) still
        # leaves this epoch's trained weights on disk instead of losing them.
        last_dir = args.output_dir.parent / "last"
        last_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        val_acc, confusion = _evaluate(model, val_loader, device, label2id)
        if global_step > args.warmup_steps:
            plateau_scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch} done - val_acc {val_acc:.4f} lr {current_lr:.2e} {confusion}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            metrics_path.write_text(json.dumps({"val_acc": val_acc, "epoch": epoch}))
            print(f"saved new best checkpoint (val_acc={val_acc:.4f}) to {args.output_dir}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping - no val_acc improvement in {args.patience} epochs")
                break

    print(f"training complete - best val_acc={best_val_acc:.4f}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cache_p = sub.add_parser("cache", help="Decode datasets once into a compact clip cache.")
    cache_p.add_argument("--dataset-roots", nargs="+", type=Path, default=DEFAULT_DATASET_ROOTS)
    cache_p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    cache_p.add_argument("--workers", type=int, default=os.cpu_count() or 1)

    train_p = sub.add_parser("train", help="Fine-tune from the cache and save the best checkpoint.")
    train_p.add_argument("--dataset-roots", nargs="+", type=Path, default=DEFAULT_DATASET_ROOTS)
    train_p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    train_p.add_argument("--output-dir", type=Path, default=CUSTOM_MODEL_DIR)
    train_p.add_argument("--epochs", type=int, default=300, help="Upper bound - early stopping normally ends the run well before this.")
    train_p.add_argument("--patience", type=int, default=10, help="Stop after this many epochs with no val_acc improvement.")
    train_p.add_argument("--plateau-patience", type=int, default=3, help="Halve the LR after this many epochs with no val_acc improvement.")
    train_p.add_argument("--warmup-steps", type=int, default=200)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=2e-5)
    train_p.add_argument("--weight-decay", type=float, default=0.05)
    train_p.add_argument("--dropout", type=float, default=0.1)
    train_p.add_argument("--workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.command == "cache":
        build_cache(args)
    elif args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
