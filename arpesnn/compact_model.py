"""Lower-cost U-Net for v2 tasks; legacy checkpoints retain their old model."""

import torch
from torch import nn
from torch.nn import functional as F


class CompactUNet(nn.Module):
    def __init__(self, task="denoise", base_channels=16):
        super().__init__()
        if task not in {"denoise", "bareband"}:
            raise ValueError("Unknown task")
        self.task = task
        self.base_channels = base_channels

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.SiLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(4, cout),
                nn.SiLU(),
            )

        c = base_channels
        self.enc1 = block(1, c)
        self.enc2 = block(c, 2 * c)
        self.center = block(2 * c, 4 * c)
        self.dec2 = block(6 * c, 2 * c)
        self.dec1 = block(3 * c, c)
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        c = self.center(F.avg_pool2d(e2, 2))
        d2 = self.dec2(
            torch.cat(
                [
                    F.interpolate(
                        c, size=e2.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    e2,
                ],
                1,
            )
        )
        d1 = self.dec1(
            torch.cat(
                [
                    F.interpolate(
                        d2, size=e1.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    e1,
                ],
                1,
            )
        )
        out = self.head(d1)
        return (x + out).clamp_min(0) if self.task == "denoise" else F.softplus(out)
