"""The classifiers trained here. Each takes patches as B × RGB × height × width
in [0, 1] and returns one logit per patch, positive for occupied."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small from torchvision's ImageNet weights, with one output.
    The patch is upsampled ``upsample`` times before the network."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, upsample: int = 1) -> None:
        super().__init__()
        self.net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        last = self.net.classifier[-1]
        assert isinstance(last, nn.Linear)
        self.net.classifier[-1] = nn.Linear(last.in_features, 1)
        self.upsample = upsample
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std).squeeze(1)
