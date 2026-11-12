import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
from .light_conv import LightConvPathway, QueryDynamicConvPathway
from .shallow_pathway import DynamicShallowConvPathway
from .up_block import UpBlock


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    """DPT with the ViT-Up style image pathway fused after the reassemble conv.

    `bypass_type` selects which image pathway is built (unified switch, same
    semantics as `vggt-up-highrank`):

      * ``"v2"``  (default) -- `DynamicShallowConvPathway`: per-pixel *depthwise*
        3x3 dynamic kernels over ``shallow_ch``-channel shallow features,
        predicted by a nonlinear MLP from a high-rank conditioning vector. Only
        the two finest scales carry the attention branch (``bypass_attn_scales``).
      * ``"dyup"``          -- `QueryDynamicConvPathway` (`DPT_DyUp`): a *linear*
        1x1 generator predicts a per-pixel 3x3 kernel applied to the 3-channel
        RGB image. Measured to collapse to rank-1.
      * ``"light"``         -- `LightConvPathway` (`DPT_Up`): a fixed stride-2
        conv pyramid.
    """

    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
        bypass_type="v2",
        bypass_shallow_ch=64,
        bypass_shallow_hidden=128,
        bypass_attn_scales=(0, 1),
        bypass_num_heads=4,
        bypass_window_size=16,
        bypass_use_featx=True,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

        # ---- 图像通路 (unified switch) ----
        # 保留 legacy 属性名 `light_conv`: 训练脚本的参数组按该前缀区分新通路。
        if bypass_type not in ("v2", "dyup", "light", "none"):
            raise ValueError(
                f"bypass_type must be 'v2', 'dyup', 'light' or 'none', "
                f"got {bypass_type!r}")
        self.bypass_type = bypass_type
        # 动态核通路 (需要每尺度的 query context) vs 前向一次算完的固定通路
        self.bypass_is_dynamic = bypass_type in ("v2", "dyup")

        if bypass_type == "v2":
            self.light_conv = DynamicShallowConvPathway(
                features=features,
                shallow_ch=bypass_shallow_ch,
                hidden=bypass_shallow_hidden,
                attn_scales=tuple(bypass_attn_scales),
            )
        elif bypass_type == "dyup":
            self.light_conv = QueryDynamicConvPathway(features=features)
        elif bypass_type == "light":
            self.light_conv = LightConvPathway(features=features, use_bn=use_bn)
        else:  # "none": 消融用 -- 不构造图像通路, 也不构造注意力支路,
            # 只保留 FeatX (identity 占位, 参数为 0, 便于统一参数组逻辑)
            self.light_conv = nn.Identity()

        # v2 只在 patch_scale >= 2 的两个细尺度启用注意力 (见 PATHWAY_V2.md);
        # dyup / light 保留全部四个尺度, 与各自的历史 checkpoint 一致;
        # none 没有 K/V, 因此四个尺度全部关掉注意力, 只留 FeatX。
        if bypass_type == "v2":
            attn_scales = tuple(int(s) for s in bypass_attn_scales)
        elif bypass_type == "none":
            attn_scales = ()
        else:
            attn_scales = (0, 1, 2, 3)
        self.attn_scales = attn_scales

        # ViT-Up 式三支路融合块 (窗口交叉注意力 + FeatX 子 token 支路,
        # 无门控): out = residual + x_attn + x_subtoken。注入点与 DPT
        # 加性融合版完全一致: 全部在 reassemble (layer_X_rn) 之后、
        # refinenet 金字塔之前。patch_scale = 该尺度每 patch 像素数
        # (相对 ViT patch 网格 H/16), 供 FeatX 计算亚 patch 相对偏移:
        #   up_blocks[0] at H/4  -> layer_1_rn  (patch_scale 4)
        #   up_blocks[1] at H/8  -> layer_2_rn  (patch_scale 2)
        #   up_blocks[2] at H/16 -> layer_3_rn  (patch_scale 1, Δx≡0)
        #   up_blocks[3] at H/32 -> layer_4_rn  (patch_scale 0.5)
        self.up_blocks = nn.ModuleList([
            UpBlock(features=features, num_heads=bypass_num_heads,
                    window_size=bypass_window_size, patch_scale=ps,
                    use_featx=bypass_use_featx,
                    use_attn=(i in attn_scales))
            for i, ps in enumerate((4.0, 2.0, 1.0, 0.5))
        ])

    def forward(self, x, return_aux=False):
        """Forward pass.

        Args:
            x: (B, 3, H, W) 输入图像。
            return_aux: 为 True 时额外返回 path_2 (H/8 尺度、features 通道的
                融合特征), 供分割任务里的辅助分类头 (auxlayer) 使用。
                默认 False, 深度估计等原有任务行为完全不变。
        """
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        if getattr(self.light_conv, "capture_kernels", False):
            self.light_conv.captured_kernels.clear()
            if hasattr(self.light_conv, "captured_condition"):
                self.light_conv.captured_condition.clear()

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # Image pathway -> per-scale K/V, then each UpBlock fuses at the DPT
        # injection point. 与 DPT 加性融合版 (DPT/dpt/models.py) 一一对应,
        # 只是把 `layer_X_rn + conv_X` 换成三支路
        # out = residual + x_attn + x_subtoken:
        #   - 注入全部发生在 reassemble 之后、refinenet 金字塔之前;
        #   - x_attn: 窗口交叉注意力输出 (query 来自 layer_X_rn 的 3x3
        #     局部聚合, key/value 来自旁路) -- 加性细化, 无门控;
        #   - x_subtoken: FeatX 支路 (相对 patch 偏移条件的 FiLM + MLP);
        #   - refinenet 金字塔保持原版 DPT 完全不变。
        if self.bypass_type == "none":
            # 消融 "关掉轻量卷积旁路": 没有 K/V, 注意力支路整体不存在,
            # 只保留 FeatX 子 token 支路 (up_blocks 的 use_attn 全为 False)。
            layer_1_rn = self.up_blocks[0](None, layer_1_rn)   # H/4
            layer_2_rn = self.up_blocks[1](None, layer_2_rn)   # H/8
            layer_3_rn = self.up_blocks[2](None, layer_3_rn)   # H/16
            layer_4_rn = self.up_blocks[3](None, layer_4_rn)   # H/32
        elif not self.bypass_is_dynamic:
            # "light": 固定 conv 金字塔, 前向一次算完 (不需要 query context)
            conv_1, conv_2, conv_3, conv_4 = self.light_conv(x)
            layer_1_rn = self.up_blocks[0](conv_1, layer_1_rn)   # H/4
            layer_2_rn = self.up_blocks[1](conv_2, layer_2_rn)   # H/8
            layer_3_rn = self.up_blocks[2](conv_3, layer_3_rn)   # H/16
            layer_4_rn = self.up_blocks[3](conv_4, layer_4_rn)   # H/32
        else:
            # "v2" / "dyup": 核由该尺度的 query context 生成, 因此只能在这里
            # 逐尺度算; query context 同时被 UpBlock 复用 (每尺度只聚合一次)。
            if hasattr(self.light_conv, "prepare"):
                # v2 的浅层编码器对四个尺度共享, 每个 forward 只跑一次
                self.light_conv.prepare(x)
            reassembled = [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]
            refined = []
            for scale_index, (up_block, residual) in enumerate(
                zip(self.up_blocks, reassembled)
            ):
                if not up_block.use_attn:
                    # 粗尺度: 只有 FeatX, 没有注意力支路, 也不需要 K/V
                    refined.append(up_block(None, residual))
                    continue
                query_context = up_block.query_context(residual)
                kv = self.light_conv(
                    x, query_context, residual.shape[-2:], scale_index
                )
                refined.append(up_block(kv, residual, query_context=query_context))
            layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn = refined

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        if return_aux:
            return out, path_2
        return out

    def load(self, path):
        """Load a complete task checkpoint, including encoder, with strict keys."""
        parameters = torch.load(path, map_location="cpu", weights_only=True)
        parameters = parameters.get("model", parameters)
        self.load_state_dict(parameters, strict=True)
        self._load_missing = []
        self._load_unexpected = []


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)

    def forward(self, x, return_aux=False):
        """Forward pass.

        Args:
            x: (B, 3, H, W) 输入图像。
            return_aux: 为 True 时额外返回辅助头 (auxlayer) 在 H/4 尺度上的
                类别 logits, 用于训练期的辅助监督 (权重默认 0.4, 与原始
                DPT/ADE20K 分割训练一致)。推理时用 False 即可, 只返回主头
                输出。
        """
        if return_aux:
            out, path_2 = super().forward(x, return_aux=True)
            return out, self.auxlayer(path_2)
        return super().forward(x)
