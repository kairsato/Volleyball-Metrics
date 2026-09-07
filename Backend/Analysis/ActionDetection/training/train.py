"""Trains the per-player action classifier (spike/set/dig/block) against the
merged crop dataset datasetGather.py's ActionDatasets class builds from
several public volleyball-action sources (see that class's docstring for
the full source list and licensing notes).

Usage (from Backend/Analysis/, so ActionDetection is importable as a
top-level package):
    python -m ActionDetection.training.train \\
        --data MachineLearning/dataset_actionDetection/merged --epochs 30 --out ../action_classifier.pt

--data must contain train/<class>/*.jpg and val/<class>/*.jpg subfolders,
one folder per class in APP_ACTION_CLASSES - exactly what
ActionDatasets.build_train_val_split produces. Ordinarily you'd reach this
indirectly via MachineLearning/mainTrainingModels.py, which gathers+merges
every source and then calls train_from_folder() below directly; this CLI
is for retraining against an already-merged folder without re-downloading
anything.
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .dataset import APP_ACTION_CLASSES
from .model import build_model

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Mild augmentation - these are small, tightly-cropped player images, not
# full scenes, so aggressive crops/rotations risk cutting off the exact
# limb position that distinguishes e.g. a set from a dig.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def train_from_folder(
    data_dir,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-4,
    architecture: str = "resnet50",
    out_path=Path("action_classifier.pt"),
) -> Path:
    """Trains from an ImageFolder-style directory (data_dir/train/<class>,
    data_dir/val/<class>) and saves the best-val-accuracy checkpoint as a
    dict of {state_dict, classes, architecture} - the class list travels
    with the weights so inference (actionDetection.py) never has to
    hardcode or guess index-to-label ordering.

    Batch size defaults large and training runs under mixed precision
    (torch.amp) with pinned-memory, multi-worker data loading and
    cudnn.benchmark on - tuned to keep a modern GPU's compute busy rather
    than data-loading-bound, since a small per-crop ResNet forward/backward
    pass is cheap enough that a small batch/timid data pipeline would
    otherwise leave most of the GPU idle between steps.
    """
    data_dir = Path(data_dir)
    train_set = ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORM)
    val_set = ImageFolder(data_dir / "val", transform=VAL_TRANSFORM)

    if train_set.classes != APP_ACTION_CLASSES:
        raise ValueError(
            f"Merged dataset classes {train_set.classes} don't match the app's "
            f"expected {APP_ACTION_CLASSES} - re-run "
            f"datasetGather.ActionDatasets.build_train_val_split, or a class "
            f"folder under {data_dir} is missing/misnamed."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    torch.backends.cudnn.benchmark = use_cuda
    print(f"Training on {device} - {len(train_set)} train / {len(val_set)} val crops, classes={train_set.classes}")

    # These are small per-crop images (a few hundred KB decoded), so a
    # single GPU training step is fast relative to JPEG-decode +
    # augmentation CPU work - without enough worker parallelism/prefetch,
    # the GPU sits idle between steps waiting on the data loader rather
    # than actually being kept busy. Uses (nearly) every CPU core and
    # queues several batches ahead per worker to keep the GPU fed.
    num_workers = max(1, (os.cpu_count() or 4) - 2)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=use_cuda, persistent_workers=num_workers > 0, drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=num_workers,
        pin_memory=use_cuda, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model = build_model(num_classes=len(APP_ACTION_CLASSES), architecture=architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_val_acc = -1.0
    out_path = Path(out_path)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)

        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=use_cuda)
                labels = labels.to(device, non_blocking=use_cuda)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / max(1, len(train_set))
        val_acc = correct / max(1, total)
        elapsed = time.time() - start
        print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}  ({elapsed:.1f}s)")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "classes": train_set.classes, "architecture": architecture},
                out_path,
            )

    print(f"Saved best model (val_acc={best_val_acc:.3f}) to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--architecture", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--out", type=Path, default=Path("action_classifier.pt"))
    args = parser.parse_args()

    train_from_folder(
        args.data, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, architecture=args.architecture, out_path=args.out,
    )


if __name__ == "__main__":
    main()
