"""Trains the per-player action classifier against the Ibrahim et al.
Volleyball dataset (https://github.com/mostafa-saad/deep-activity-rec).

Usage (from Backend/, with Analysis/ on sys.path):
    python -m ActionDetection.training.train \\
        --annotations /path/to/volleyball_tracking_annotation \\
        --images /path/to/videos_root \\
        --epochs 15 --out action_classifier.pt

--images only needs to contain whichever clips you actually have frame
images for - see dataset.py's PlayerActionCrops for how the two are paired;
samples missing an image are simply skipped, so this also runs (as a
smoke test, not a usable model) against a small partial image set.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import ACTION_LABELS, PlayerActionCrops
from .model import build_model

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=Path("action_classifier.pt"))
    args = parser.parse_args()

    dataset = PlayerActionCrops(args.annotations, args.images, transform=TRANSFORM)
    print(f"Loaded {len(dataset)} labeled player crops.")
    if len(dataset) == 0:
        raise SystemExit("No samples found - check --annotations/--images point at matching clip folders.")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")

    model = build_model(num_classes=len(ACTION_LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        train_loss = total_loss / max(1, len(train_set))
        val_acc = correct / max(1, total)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved model weights to {args.out}")


if __name__ == "__main__":
    main()
