"""CNN backbones for the per-player action crops in dataset.py. Uses an
ImageNet-pretrained backbone as a starting point, same idea
PlayerDetection/trackerRaw.py's AppearanceEncoder already relies on for
re-identification embeddings - staying consistent with how the rest of
this pipeline uses a CNN backbone rather than introducing a second
convention.

resnet50 is the default: with 5-8 merged public datasets (tens of
thousands of crops, see ../datasets.py) there's enough data to benefit
from the extra capacity over resnet18, and the larger backbone also
does a better job of actually using a high-end GPU's compute during
training rather than leaving it idle waiting on the data loader.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

_ARCHITECTURES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
}


def build_model(num_classes: int, architecture: str = "resnet50", pretrained: bool = True) -> nn.Module:
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}' - choose one of {list(_ARCHITECTURES)}.")

    constructor, default_weights = _ARCHITECTURES[architecture]
    model = constructor(weights=default_weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
