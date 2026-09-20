"""Coarse, nonnegative background estimate; subtraction remains signed and inspectable."""

import torch
from torch import nn
from torch.nn import functional as F
from compact_model import CompactUNet


class BackgroundNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        self.core = CompactUNet("bareband", base_channels)
        nn.init.constant_(self.core.head.bias, -3.0)

    def forward(self, x):
        # Fixed normalized grid is deliberate: background should vary slowly across a cut.
        low = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        field = self.core(low)
        coarse = F.adaptive_avg_pool2d(field, (32, 32))
        return F.interpolate(
            coarse, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


def load_background(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if (
        saved.get("architecture") != "background_coarse_v1"
        or saved.get("task") != "background"
    ):
        raise ValueError("Expected background_coarse_v1 checkpoint")
    model = BackgroundNet(saved["base_channels"]).to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved
