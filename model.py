"""
Model module: an ImageNet-pretrained backbone with a one-logit head
for pothole / no pothole.
"""

import torch
from torch import nn
from torchvision import models

BACKBONES = {
    'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
}


def build_model(backbone='mobilenet_v3_small', pretrained=True):
    """Return a classifier that outputs one logit (pothole if > 0)."""
    if backbone not in BACKBONES:
        raise ValueError(f'Unknown backbone {backbone}; choose from {list(BACKBONES)}')
    fn, weights = BACKBONES[backbone]
    model = fn(weights=weights if pretrained else None)
    if backbone == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model


def last_conv_layer(model):
    """The layer Grad-CAM reads from."""
    return model.layer4 if hasattr(model, 'layer4') else model.features[-1]


def save_model(model, path, **info):
    torch.save({'state_dict': model.state_dict(), **info}, path)


def load_model(path):
    """Load a checkpoint saved by save_model. Returns (model, info)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ckpt['backbone'], pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt
