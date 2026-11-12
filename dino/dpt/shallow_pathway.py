"""Plan v2 image pathway: dynamic kernels that actually filter *shallow features*.

Ported from ``vggt-up-highrank/vggt/heads/shallow_pathway.py`` (see
`PATHWAY_V2.md` there). All of the motivation below was measured on the trained
`mixture` / `dyup` / `light` checkpoints of that repo and applies unchanged here:

  * The old pathway filters the **3-channel RGB image** with a per-pixel 3x3
    kernel and then projects 3 -> `features` with a 1x1 conv. A 1x1 linear
    projection of a 3-channel filtered image carries no edge/texture
    information that the reassembled DPT feature `r_i` does not already have.
  * Consequence (measured): replacing the K/V with **random noise** changes the
    metrics by <2e-4, and zeroing the attention branch changes nothing either.
    With a fixed K/V the attention degenerates into
    ``x_attn = W_o * Attn(q, K_fixed, V_fixed)``, i.e. a *fixed learnable
    transform of the query* -- the image never actually enters.
  * The kernel generator was a single linear 1x1 conv on a spatially rank-1
    query context, so the per-pixel kernels provably collapse to
    ``K(p) ~= a(p) * s`` (measured kernel PC1 = 1.000, gain CV = 0.00).

This module fixes all three:

  1. **Multi-channel target.** A lightweight stride-2 conv *stem*
     (`ShallowEncoder`) produces `shallow_ch`-channel features at the two
     finest DPT scales. The dynamic kernel is a per-pixel **depthwise** 3x3
     filter over those channels (not 3), so it can express edges/texture.
  2. **Rank-raising conditioning.** The kernel is predicted from
     ``[ query context c_i | local shallow feature | local high-pass |
        unfolded RGB 3x3 ]`` through a **2-layer GELU MLP**. The local image
     statistics are spatially high-rank, so the generator is no longer capped by
     the rank-1 query context, and the MLP is nonlinear (a linear map cannot
     raise rank at all).
  3. **LayerNorm before the 1x1 projection**, so the projection cannot absorb
     all of the learning (measured in the source repo: the old `projections`
     never moved from its random init, while `o_proj` -- whose input was
     therefore uninformative -- did all the moving).

Where it runs
-------------
Only at ``patch_scale >= 2``, i.e. the two finest DPT scales H/4 and H/8
(DPT patch size 16 -> patch_scale 4 and 2). At ``patch_scale = 1`` (H/16) each
feature pixel *is* a patch centre (offset == 0) and at ``patch_scale = 0.5``
(H/32) the feature grid is coarser than the patch grid, so the "recover
sub-patch detail from the image" premise does not exist there. Those scales
keep a `UpBlock` with the FeatX branch only (``use_attn=False``).

Exact warm start
----------------
`kernel_heads`' **last** layer is zero-initialised and its centre-tap bias is
set to 1 for every channel, so at init the per-pixel depthwise kernel is the
**identity** filter and ``filtered == shallow feature``. `UpBlock.o_proj` is
zero-initialised (ReZero) so ``x_attn == 0`` regardless of the K/V, making a
freshly built model **bit-identical** to the original DPT head.

Kernel capture (`capture_kernels`) uses the same ``(B, C_sh, 9, h, w)`` layout as
`QueryDynamicConvPathway`, so the existing diagnostics tools keep working.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .light_conv import ConvBlock
from .up_block import LayerNorm2d, _trunc_normal_


class ShallowEncoder(nn.Module):
    """Small stride-2 conv stem producing multi-channel *shallow* features.

    Deliberately shallow (4 conv layers, 2 of them strided): its job is to give
    the dynamic kernel something richer than 3-channel RGB to filter, while
    staying cheap. Because it is a plain CNN on the raw image it retains the
    high-frequency detail the ViT backbone discards -- exactly the information
    `r_i` lacks.

    Outputs are produced at H/4 and H/8, which for DPT (patch size 16) are
    exactly the resolutions of `layer_1_rn` and `layer_2_rn`, so no resampling
    is needed for the two scales that carry the attention branch.
    """

    def __init__(self, shallow_ch=64, use_bn=False):
        super().__init__()
        self.shallow_ch = shallow_ch
        c = shallow_ch
        # H/2 -> H/4  (fp1, `c` channels)
        self.stem = nn.Sequential(
            ConvBlock(3, max(16, c // 2), stride=2, use_bn=use_bn),   # H/2
            ConvBlock(max(16, c // 2), max(16, c // 2), stride=1, use_bn=use_bn),
            ConvBlock(max(16, c // 2), c, stride=2, use_bn=use_bn),   # H/4
            ConvBlock(c, c, stride=1, use_bn=use_bn),
        )
        # H/4 -> H/8  (fp2, `c` channels)
        self.stage2 = nn.Sequential(
            ConvBlock(c, c, stride=2, use_bn=use_bn),                 # H/8
            ConvBlock(c, c, stride=1, use_bn=use_bn),
        )

    def forward(self, x):
        """x: (B, 3, H, W) -> (fp1 (B,C,H/4,W/4), fp2 (B,C,H/8,W/8))"""
        f1 = self.stem(x)
        f2 = self.stage2(f1)
        return f1, f2


class DynamicShallowConvPathway(nn.Module):
    """Per-pixel depthwise dynamic filtering of multi-channel shallow features.

    Layout::

        image ──ShallowEncoder──> f1 (C_sh, H/4)   f2 (C_sh, H/8)
                                   │                 │
        (resize to the DPT scale)  ▼                 ▼
                                  F_i  (C_sh, h_i, w_i)          <- the filter target
                                   ├── F_hi = F_i - blur3x3(F_i) <- local high-pass
                                   └── rgb_patch = unfold(image, 3x3)
                                                │
        DPT query context c_i (features, h_i, w_i) ──┐
                                                ▼   ▼
              kernel = MLP([c_i | F_i | F_hi | rgb_patch])   ->  (C_sh * 9, h_i, w_i)
                                                │
                          filtered = Σ_taps unfold(F_i) ⊙ kernel      (depthwise 3x3)
                                                │
                          kv = proj( LayerNorm2d(filtered) )          -> (features, h_i, w_i)
    """

    def __init__(self, features=256, shallow_ch=64, hidden=128,
                 num_scales=4, attn_scales=(0, 1), kernel_std=0.02,
                 use_bn=False, use_highpass_cond=True, use_rgb_cond=True,
                 norm_cond=True):
        super().__init__()
        self.features = features
        self.shallow_ch = int(shallow_ch)
        self.attn_scales = tuple(int(s) for s in attn_scales)
        self.use_highpass_cond = bool(use_highpass_cond)
        self.use_rgb_cond = bool(use_rgb_cond)
        self.norm_cond = bool(norm_cond)
        self.capture_kernels = False
        self.captured_kernels = {}
        self.captured_condition = {}

        self.encoder = ShallowEncoder(self.shallow_ch, use_bn=use_bn)

        # conditioning width = c_i (features) + F_i (C_sh) [+ F_hi] [+ rgb 27]
        cond_dim = features + self.shallow_ch
        if self.use_highpass_cond:
            cond_dim += self.shallow_ch
        if self.use_rgb_cond:
            cond_dim += 3 * 9
        self.cond_dim = cond_dim
        # widths of each conditioning block, in concat order
        self.cond_widths = [features, self.shallow_ch]
        if self.use_highpass_cond:
            self.cond_widths.append(self.shallow_ch)
        if self.use_rgb_cond:
            self.cond_widths.append(3 * 9)
        assert sum(self.cond_widths) == cond_dim, (self.cond_widths, cond_dim)

        # one kernel head per enabled scale
        self.kernel_heads = nn.ModuleDict()
        for s in self.attn_scales:
            head = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.shallow_ch * 9),
            )
            # nonlinear, and the LAST layer is zero-init + centre tap = 1
            _trunc_normal_(head[0].weight, std=kernel_std)
            nn.init.zeros_(head[0].bias)
            nn.init.zeros_(head[2].weight)
            nn.init.zeros_(head[2].bias)
            with torch.no_grad():
                head[2].bias[4::9] = 1.0        # centre offset (0,0) -> identity
            self.kernel_heads[str(s)] = head

        # 1x1 projection into the attention feature space, with a LayerNorm in
        # front so the projection cannot absorb all of the learning
        self.norm_filt = LayerNorm2d(self.shallow_ch)
        self.projections = nn.ModuleDict({
            str(s): nn.Conv2d(self.shallow_ch, features, kernel_size=1)
            for s in self.attn_scales
        })

        # cache of the shallow pyramid, filled once per forward by `prepare`
        self._shallow = None

    # ------------------------------------------------------------------
    def prepare(self, image):
        """Run the shallow encoder once per forward pass (all scales share it)."""
        f1, f2 = self.encoder(image)
        self._shallow = (f1, f2)

    def _clear(self):
        self._shallow = None

    # ------------------------------------------------------------------
    def forward(self, image, query_context, output_size, scale_index):
        """Filter the shallow feature at `scale_index` with a query-generated kernel.

        Args:
            image (Tensor): (B, 3, H, W) normalized input image (used for the
                shallow encoder and, optionally, for the local RGB patches).
            query_context (Tensor): (B, features, h, w) DPT context at this scale.
            output_size (tuple): (h, w) target size.
            scale_index (int): which scale; must be in `self.attn_scales`.

        Returns:
            Tensor (B, features, h, w) K/V for the UpBlock at this scale, or
            None when this scale has no attention branch.
        """
        key = str(int(scale_index))
        if key not in self.kernel_heads:
            return None

        if self._shallow is None:            # direct call (diagnostics scripts)
            self.prepare(image)
        f1, f2 = self._shallow

        # pick the shallow feature whose stride is closest to the target scale,
        # then resample: H/4 -> H/4, H/8 -> H/8 (exact for DPT patch size 16)
        h, w = int(output_size[0]), int(output_size[1])
        H = image.shape[-2]
        src = f1 if abs(h - H / 4.0) <= abs(h - H / 8.0) else f2
        if src.shape[-2:] != (h, w):
            src = F.interpolate(src, size=(h, w), mode="bilinear",
                                align_corners=False)

        batch = src.shape[0]
        cond = [query_context]
        cond.append(src)
        if self.use_highpass_cond:
            low = F.avg_pool2d(src, kernel_size=3, stride=1, padding=1)
            cond.append(src - low)
        if self.use_rgb_cond:
            img = F.interpolate(image, size=(h, w), mode="bilinear",
                                align_corners=False)
            patch = F.unfold(img, kernel_size=3, padding=1)          # (B, 27, hw)
            cond.append(patch.reshape(batch, 27, h, w))

        c = torch.cat(cond, dim=1)                                   # (B, cond_dim, h, w)
        # ---- per-block normalisation -------------------------------------
        # Essential: the DPT context c_i has a magnitude of O(10..10^3) while
        # the shallow/image blocks are O(1). Concatenating them raw makes the
        # *conditioning itself* rank-1 (measured shape_PC1 = 0.988 on real data)
        # because c_i dominates the variance, which in turn forces the predicted
        # kernels back to rank-1. Standardising each block to unit variance per
        # pixel lets the high-rank image statistics contribute.
        if self.norm_cond:
            chunks, off = [], 0
            for bw in self.cond_widths:      # NOTE: not `w` -- that is the spatial width
                blk = c[:, off:off + bw]
                off += bw
                mu = blk.mean(dim=1, keepdim=True)
                sd = blk.std(dim=1, keepdim=True).clamp_min(1e-5)
                chunks.append((blk - mu) / sd)
            c = torch.cat(chunks, dim=1)
        c = c.permute(0, 2, 3, 1).reshape(batch, h * w, self.cond_dim)
        if self.capture_kernels:
            self.captured_condition[int(scale_index)] = c.detach()

        kernels = self.kernel_heads[key](c)                          # (B, hw, C*9)
        kernels = kernels.reshape(batch, h * w, self.shallow_ch, 9)
        kernels = kernels.permute(0, 2, 3, 1)                        # (B, C, 9, hw)

        patches = F.unfold(src, kernel_size=3, padding=1)            # (B, C*9, hw)
        patches = patches.reshape(batch, self.shallow_ch, 9, h * w)
        filtered = (patches * kernels).sum(dim=2).reshape(batch, self.shallow_ch, h, w)

        if self.capture_kernels:
            self.captured_kernels[int(scale_index)] = kernels.detach().reshape(
                batch, self.shallow_ch, 9, h, w)

        return self.projections[key](self.norm_filt(filtered))
