"""Image pathways running in parallel to the DPT decoder.

Three variants live here, in increasing order of expressiveness (same lineage
as ``vggt-up-highrank/vggt/heads/light_conv.py``):

  * ``LightConvPathway``        -- fixed stride-2 conv pyramid (``DPT_Up``).
  * ``QueryDynamicConvPathway`` -- the query predicts a per-pixel 3x3 kernel
    with a *linear* 1x1 convolution (``DPT_DyUp``). Empirically collapses to
    rank-1: one shared filter shape times a per-pixel gain.
  * ``DynamicShallowConvPathway`` (in ``dpt/shallow_pathway.py``) -- Plan v2:
    a per-pixel *depthwise* dynamic kernel over multi-channel shallow features,
    predicted by a nonlinear MLP from a high-rank conditioning vector. This is
    the pathway ``DPT_Up2`` is built around.

`ConvBlock` is shared with the shallow encoder in ``dpt/shallow_pathway.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Lightweight conv block: Conv -> Norm -> ReLU."""

    def __init__(self, in_ch, out_ch, stride=1, use_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride,
            padding=1, bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightConvPathway(nn.Module):
    """Lightweight fixed convolutional pathway running parallel to DPT.

    Extracts multi-scale features from the original image through
    pyramid-shaped downsampling (stride-2 at each stage), producing features at
    H/4, H/8, H/16, H/32 to match DPT's reassemble/fusion stages. Ported
    verbatim from ``DPT_Up/dpt/light_conv.py`` and kept here so that
    ``bypass_type="light"`` (the original `DPT_Up` baseline) can be reproduced
    through the same unified switch.

    Architecture:
        input (3, H, W)
          ├─ stage1: ConvBlock(s=2) → ConvBlock(s=1) → ConvBlock(s=2) → ConvBlock(s=1)
          │   └─ f1: (64, H/4, W/4)  ── proj1 → (features, H/4, W/4)
          ├─ stage2: ConvBlock(s=2) → ConvBlock(s=1)
          │   └─ f2: (128, H/8, W/8) ── proj2 → (features, H/8, W/8)
          ├─ stage3: ConvBlock(s=2) → ConvBlock(s=1)
          │   └─ f3: (256, H/16, W/16) ── proj3 → (features, H/16, W/16)
          └─ stage4: ConvBlock(s=2) → ConvBlock(s=1)
              └─ f4: (256, H/32, W/32) ── proj4 → (features, H/32, W/32)
    """

    def __init__(self, features=256, use_bn=False):
        super().__init__()

        # Stage 1: input → H/4, 64 channels
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32, stride=2, use_bn=use_bn),     # H/2
            ConvBlock(32, 32, stride=1, use_bn=use_bn),
            ConvBlock(32, 64, stride=2, use_bn=use_bn),    # H/4
            ConvBlock(64, 64, stride=1, use_bn=use_bn),
        )

        # Stage 2: H/4 → H/8, 128 channels
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2, use_bn=use_bn),   # H/8
            ConvBlock(128, 128, stride=1, use_bn=use_bn),
        )

        # Stage 3: H/8 → H/16, 256 channels
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=2, use_bn=use_bn),  # H/16
            ConvBlock(256, 256, stride=1, use_bn=use_bn),
        )

        # Stage 4: H/16 → H/32, 256 channels
        self.stage4 = nn.Sequential(
            ConvBlock(256, 256, stride=2, use_bn=use_bn),  # H/32
            ConvBlock(256, 256, stride=1, use_bn=use_bn),
        )

        # 1×1 projections to match DPT's feature dimension at each scale
        self.proj1 = nn.Conv2d(64, features, kernel_size=1)
        self.proj2 = nn.Conv2d(128, features, kernel_size=1)
        self.proj3 = nn.Conv2d(256, features, kernel_size=1)
        self.proj4 = nn.Conv2d(256, features, kernel_size=1)

    def forward(self, x):
        """x: (B, 3, H, W) -> (p1..p4) at H/4..H/32, each `features` channels."""
        f1 = self.stage1(x)    # (B, 64,  H/4,  W/4)
        f2 = self.stage2(f1)   # (B, 128, H/8,  W/8)
        f3 = self.stage3(f2)   # (B, 256, H/16, W/16)
        f4 = self.stage4(f3)   # (B, 256, H/32, W/32)

        p1 = self.proj1(f1)    # (B, features, H/4,  W/4)
        p2 = self.proj2(f2)    # (B, features, H/8,  W/8)
        p3 = self.proj3(f3)    # (B, features, H/16, W/16)
        p4 = self.proj4(f4)    # (B, features, H/32, W/32)

        return p1, p2, p3, p4


class QueryDynamicConvPathway(nn.Module):
    """Generate per-location 3x3 filters from DPT context and apply to RGB."""

    def __init__(self, features=256, num_scales=4):
        super().__init__()
        self.capture_kernels = False
        self.captured_kernels = {}
        self.kernel_generators = nn.ModuleList([
            nn.Conv2d(features, 27, kernel_size=1) for _ in range(num_scales)
        ])
        self.projections = nn.ModuleList([
            nn.Conv2d(3, features, kernel_size=1) for _ in range(num_scales)
        ])

        for generator in self.kernel_generators:
            nn.init.zeros_(generator.weight)
            nn.init.zeros_(generator.bias)
            with torch.no_grad():
                generator.bias[4::9] = 1.0

    def forward(self, image, query_context, output_size, scale_index):
        """Filter the resized RGB input with a query-generated kernel per pixel.

        Each RGB channel gets its own 3x3 spatial kernel. A learned 1x1
        projection maps the filtered image to the fusion feature channels.
        """
        image = F.interpolate(
            image, size=output_size, mode="bilinear", align_corners=False
        )
        batch, _, height, width = image.shape
        kernels = self.kernel_generators[scale_index](query_context)
        kernels = kernels.reshape(batch, 3, 9, height * width)
        if self.capture_kernels:
            self.captured_kernels[scale_index] = kernels.detach().reshape(
                batch, 3, 9, height, width
            )

        patches = F.unfold(image, kernel_size=3, padding=1)
        patches = patches.reshape(batch, 3, 9, height * width)
        filtered = (patches * kernels).sum(dim=2).reshape(batch, 3, height, width)
        return self.projections[scale_index](filtered)
