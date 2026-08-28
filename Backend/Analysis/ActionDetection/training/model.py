"""A ResNet18-based classifier for the per-player action crops in
dataset.py. Uses an ImageNet-pretrained backbone as a starting point, same
idea PlayerDetection/trackerRaw.py's AppearanceEncoder already relies on
for re-identification embeddings - staying consistent with how the rest of
this pipeline uses a CNN backbone rather than introducing a second
convention."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
