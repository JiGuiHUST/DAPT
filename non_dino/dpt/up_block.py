"""ViT-Up 式三支路融合: 卷积旁路特征经交叉注意力 + FeatX 注入 DPT (无门控).

与 DPT_query (门控版) 的区别:
    删除 sigmoid 门控, 改为 ViT-Up 原版三支路相加融合
    (x_fused = x + x_attn + x_subtoken):
      - x        : 被注入的 DPT 特征 (reassemble 后 layer_X_rn, 主干残差)
      - x_attn   : 窗口交叉注意力输出 -- query 由 layer_X_rn 像素经 3x3
                   卷积局部聚合生成 (判断边缘/纹理等局部结构, ViT-Up 的
                   patch-embedding 思想), key/value 来自卷积旁路;
                   加性特征细化, 不再做门控选择
      - x_subtoken: FeatX 支路 -- 最近 patch 相对偏移 Δx 条件的 FiLM 调制
                   + 子 token MLP, 恢复 patch 内亚位置细节 (ViT-Up FeatX)

注入点与 DPT 加性融合版 (DPT/dpt/models.py) 完全一致:
全部发生在 reassemble (layer_X_rn) 之后、refinenet 金字塔之前,
只是把 `layer_X_rn + conv_X` 换成三支路:

    out = residual + o_proj(W-MHA(q(residual), kv)) + FeatX(residual, Δx)

位置编码:
  - 注意力内部: 可学习 Swin 式相对位置偏置 (窗口内), 无全局坐标 RoPE
    (窗口下与相对偏置冗余, 且对推理分辨率敏感)
  - FeatX: 查询像素相对最近 ViT patch 中心的偏移 Δx (patch 单位,
    静态几何量, 随 (H, W, patch_scale) 现算, 正弦编码)

The DPT part of the network (ViT backbone, reassemble, refinenet pyramid,
output head) is kept exactly as in the original DPT, so the original DPT
weights load directly (strict=False) before training the bypass pathway.

三支路注入点 (per scale, DPT 同款):
    input:  kv       (conv bypass feature at scale i, C x H_i x W_i)
            residual (reassembled DPT feature layer_X_rn at scale i,
                      C x H_i x W_i -- DPT 同款注入点, refinenet 之前)
    steps:
        1. query     : q = Q_proj(LN(GELU(Conv3x3(residual))))
        2. cross-attn: z = W-MHA(q, k, v)            # 4 头 x 64 维窗口注意力
                        (windowed cross-attention, 4 heads, window 16)
        3. x_attn    : x_attn = o_proj(z)            # 加性注意力支路
        4. x_subtoken: FeatX(residual, Δx)           # 子 token 细节支路
        5. output    : out = residual + x_attn + x_subtoken
    Small-nonzero initialization: q/k/v 投影权重保持 trunc normal 0.02
    (非零); o_proj 权重与 bias 零初始化 (ReZero 式) -> 第 0 步 x_attn ≡ 0
    (训练与评测均为精确 warm start)。o_proj 自身第 0 步即有梯度 (z≠0),
    注意力与旁路上游随 ||W_o|| 增长快速解锁 (无门控时 o_proj 零初始化
    不会锁死: 不同于门控版 sigmoid 压缩, 这里 o_proj 直接接收梯度);
    light_conv 置零时 (kv=0) x_attn ≡ 0。

初始化设计 (近似 warm start):
  - 注意力支路: q/k/v/o 投影小随机 (trunc normal 0.02), o_proj bias 零
    -> 第 0 步 x_attn 为小随机噪声 (扰动与门控版 g≈0.018 同量级),
    注意力与旁路上游从第 0 步即有梯度 (避免 ReZero 慢解锁);
    light_conv 置零时 (kv = 0) x_attn ≡ 0
  - FeatX 支路: FiLM 与 sub MLP 的最后一层均零初始化 -> 第 0 步
    x_subtoken ≡ 0 (对预训练解码器零扰动, 精确 warm start); 支路自身
    最后一层立即获得梯度, 上游随其权重增长逐步解锁 -- FeatX 是细节
    细化支路, 缓慢打开是期望行为
  - 无 transition MLP / fusion MLP (ViT-Up 论文有, 本版省略, 需要时可加)
  - 新构建模型 (FeatX 零初始化) + light_conv 置零 -> 输出精确等于原版
    DPT (eval_scannet_compare.py 的原始 DPT 基线构造方式不变)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension for 4D feature maps (B, C, H, W)."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # (B, C, H, W) -> (B, H, W, C) -> LN(C) -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


def _trunc_normal_(tensor, std=0.02):
    with torch.no_grad():
        tensor.normal_(0, std).clamp_(-2 * std, 2 * std)
    return tensor


class WindowCrossAttention(nn.Module):
    """Windowed multi-head cross-attention between two aligned feature maps.

    The feature maps are split into non-overlapping windows of size
    `window_size` x `window_size`; within each window, every query token
    attends to all key/value tokens of the *other* pathway (the conv bypass).
    A learnable relative-position bias (Swin-style) is added to the attention
    logits, which gives the attention an explicit locality prior and provides
    position information for the conv-bypass features (which carry none
    otherwise).

    Design rationale: at the refinement stage the useful bypass information is
    local, so global exchange between distant positions is unnecessary and
    only adds compute plus a risk of feature leakage across visually similar
    regions. Windowing also bounds the cost: at H/4 (128x128 for a 512 input)
    a 16x16 window means 64 windows of 256 tokens instead of one 16384-token
    map. Inputs are (B, C, H, W); H/W are padded to a multiple of
    `window_size` internally.
    """

    def __init__(self, features, num_heads=2, window_size=16, dropout=0.0):
        super().__init__()
        assert features % num_heads == 0
        self.features = features
        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # (2w-1, 2w-1, num_heads) relative position bias table
        w = window_size
        self.rel_bias = nn.Parameter(torch.zeros(2 * w - 1, 2 * w - 1, num_heads))
        _trunc_normal_(self.rel_bias, std=0.02)

        self.attn_drop = nn.Dropout(dropout)

        # Precompute the (N, N) relative-offset index table (N = w * w),
        # where entry (i, j) indexes rel_bias[dx + w - 1, dy + w - 1, :].
        coords = torch.stack(torch.meshgrid(
            torch.arange(w), torch.arange(w), indexing="ij"), dim=-1)  # (w, w, 2)
        coords = coords.flatten(0, 1)  # (N, 2)
        rel = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
        rel += w - 1
        self.register_buffer("rel_idx", rel)  # (N, N, 2)

    def forward(self, q, k, v, H, W):
        """Args:
            q, k, v: (B, C, H, W) tensors, q from backbone, k/v from bypass.
        Returns:
            (B, C, H, W) attention output at the same resolution.
        """
        B, C = q.shape[:2]
        w = self.window_size
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h or pad_w:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // w, Wp // w

        # To windows: (B, C, nH, w, nW, w) -> (B, nH*nW, C, w, w)
        def to_windows(x):
            x = x.unfold(2, w, w).unfold(3, w, w)          # (B, C, nH, w, nW, w)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()    # (B, nH, nW, C, w, w)
            return x.view(B, nH * nW, C, w, w)

        qw = to_windows(q)  # (B, nWin, C, w, w)
        kw = to_windows(k)
        vw = to_windows(v)

        # Flatten + split heads: (B, nWin, num_heads, d, N)
        N = w * w
        qw = qw.view(B, -1, self.num_heads, self.head_dim, N)
        kw = kw.view(B, -1, self.num_heads, self.head_dim, N)
        vw = vw.view(B, -1, self.num_heads, self.head_dim, N)

        attn = torch.matmul(qw.transpose(-1, -2), kw) * self.scale  # (B, nWin, H, N, N)
        # Add relative position bias
        idx = self.rel_idx  # (N, N, 2)
        bias = self.rel_bias[idx[..., 0], idx[..., 1], :]  # (N, N, num_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, vw.transpose(-1, -2))  # (B, nWin, H, N, d)
        out = out.transpose(-1, -2).contiguous().view(B, nH * nW, C, w, w)

        # Fold windows back
        out = out.view(B, nH, nW, C, w, w)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Hp, Wp)
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


def _posenc_1d(offsets, freqs):
    """NeRF 式正弦位置编码: offsets (N,) in [-0.5, 0.5] -> (N, 2L) (sin/cos 交错)."""
    enc = []
    for f in freqs:
        angle = math.pi * f * offsets
        enc.append(torch.sin(angle))
        enc.append(torch.cos(angle))
    return torch.stack(enc, dim=-1)


def _patch_offsets(H, W, patch_scale, device, dtype):
    """查询像素相对最近 ViT patch 中心的偏移 (patch 单位), 各返回 (H, W).

    patch_scale = 该尺度每 patch 像素数 (= 该尺度分辨率 / ViT patch 网格
    分辨率): H/4 -> 4, H/8 -> 2, H/16 -> 1 (Δx≡0), H/32 -> 0.5 (Δx=±0.5)。
    纯几何量, 与图像内容无关, 每次 forward 现算 (H/4=128x128, 成本可忽略;
    输入按比例 resize, 不同图像 H/W 不同, 不缓存)。
    """
    ys = torch.arange(H, device=device, dtype=dtype).view(-1, 1) + 0.5
    xs = torch.arange(W, device=device, dtype=dtype).view(1, -1) + 0.5
    nn_y = (torch.floor(ys / patch_scale) + 0.5) * patch_scale
    nn_x = (torch.floor(xs / patch_scale) + 0.5) * patch_scale
    dy = ((ys - nn_y) / patch_scale).expand(H, W)
    dx = ((xs - nn_x) / patch_scale).expand(H, W)
    return dx, dy


class OffsetFeatX(nn.Module):
    """FeatX (ViT-Up): 最近 patch 相对偏移条件的子 token 细节支路.

    调制对象为 backbone 侧特征 (residual = layer_X_rn); 相对偏移
    Δx = (x_q - x_nn) / p ∈ [-0.5, 0.5] 是查询像素相对最近 ViT patch
    中心的偏移 (patch 单位), 正弦编码后由 FiLM MLP 预测 (γ, β):

        h̃     = (1 + γ) ⊙ LN(x) + β
        x_sub = MLP_sub(h̃)

    初始化: FiLM 与 sub 的最后一层均零初始化 ->
        γ = β = 0 (调制恒等) 且 x_sub ≡ 0 (第 0 步零扰动, 精确 warm start);
        sub 最后一层立即获得梯度, 上游 (FiLM) 随其权重增长逐步解锁 --
        FeatX 是细节细化支路, 缓慢打开是期望行为。配合 light_conv 置零
        (x_attn ≡ 0), 新构建模型输出精确等于原版 DPT。
    """

    def __init__(self, features, pe_freqs=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm2d(features)
        pe_dim = 4 * pe_freqs  # dx/dy 各 sin+cos, 每坐标 pe_freqs 个频率
        hidden = int(features * mlp_ratio)

        self.film_fc1 = nn.Linear(pe_dim, hidden)
        self.film_fc2 = nn.Linear(hidden, 2 * features)
        self.sub_fc1 = nn.Linear(features, hidden)
        self.sub_fc2 = nn.Linear(hidden, features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # 小非零初始化 (与注意力支路同一哲学), bias 置零
        for lin in (self.film_fc1, self.film_fc2, self.sub_fc1, self.sub_fc2):
            _trunc_normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        # 关键: 最后一层零初始化 -> γ=β=0 且 x_sub ≡ 0 (精确 warm start)
        nn.init.zeros_(self.film_fc2.weight)
        nn.init.zeros_(self.sub_fc2.weight)

        # 频率表 2^0 ... 2^(L-1), buffer 随模块搬到设备
        self.register_buffer("freqs", torch.tensor(
            [2.0 ** k for k in range(pe_freqs)], dtype=torch.float32))

    def forward(self, x, H, W, patch_scale):
        """Args:
            x: (B, C, H, W) backbone 侧特征 (residual)
            H, W: 空间尺寸; patch_scale: 每 patch 像素数 (见 _patch_offsets)
        Returns:
            x_sub: (B, C, H, W) 子 token 细节特征
        """
        B, C = x.shape[:2]
        dx, dy = _patch_offsets(H, W, patch_scale, x.device, x.dtype)
        pe = torch.cat([
            _posenc_1d(dx.reshape(-1), self.freqs),
            _posenc_1d(dy.reshape(-1), self.freqs),
        ], dim=-1)  # (H*W, 4L)

        gamma, beta = self.film_fc2(
            self.drop(self.act(self.film_fc1(pe)))).chunk(2, dim=-1)

        h = self.norm(x)  # (B, C, H, W) LN(C)
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h = (1.0 + gamma) * h + beta

        x_sub = self.sub_fc2(self.drop(self.act(self.sub_fc1(h))))
        return x_sub.reshape(B, H, W, C).permute(0, 3, 1, 2)


class UpBlock(nn.Module):
    """ViT-Up 式三支路融合 (无门控, 无 transition/fusion MLP).

        out = residual + x_attn + x_subtoken

      residual  : reassembled DPT 特征 layer_X_rn (注入对象 + query 来源)
      x_attn    : o_proj(WindowCrossAttn(q, k, v)) -- 窗口交叉注意力
                  (query 由 residual 像素经 3x3 局部聚合生成, key/value
                  来自卷积旁路), 加性特征细化, 不再做门控选择
      x_subtoken: OffsetFeatX(residual, Δx) -- 相对 patch 偏移条件的
                  子 token 细节 (ViT-Up FeatX)

    位置信息: 窗口内 Swin 式相对位置偏置 (注意力) + 相对 patch 偏移
    (FeatX), 无全局坐标 RoPE。

    `use_attn` (Plan v2, 来自 vggt-up-highrank):
      只有 `patch_scale >= 2` 的尺度 (H/4, H/8) 才保留 query/KV 机制。粗尺度
      (`patch_scale <= 1`) 上"从图像恢复亚 patch 细节"的前提不存在 (s=1 时
      每个特征像素就是 patch 中心, Δx≡0; s=0.5 时特征网格比 patch 网格还粗),
      因此整条注意力支路被去掉, 只保留 FeatX —— 实测中唯一有贡献的支路。
      被去掉的模块不构造, 参数为零 (见 PATHWAY_V2.md §5 结构检查)。

    初始化 (精确 warm start):
      - q/k/v 投影 trunc normal 0.02; o_proj 权重+bias 零初始化 (ReZero 式)
        -> 第 0 步 x_attn ≡ 0 (零扰动), o_proj 自身第 0 步即有梯度,
        注意力与旁路上游随其权重增长快速解锁 (无门控无 sigmoid 压缩)
      - FeatX 两条支路最后一层零初始化 -> x_subtoken ≡ 0 (零扰动);
        light_conv 置零时 kv=0 -> x_attn ≡ 0, 新模型+置零旁路 = 原版 DPT
    """

    def __init__(self, features=256, num_heads=4, window_size=16,
                 patch_scale=1.0, dropout=0.0, use_featx=True, use_attn=True):
        super().__init__()
        self.features = features
        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.window_size = window_size
        self.patch_scale = patch_scale
        self.use_featx = use_featx
        self.use_attn = use_attn

        if use_attn:
            # 1. query 生成: residual 像素 -> 3x3 局部聚合 (判断边缘/纹理等局部结构)
            self.query_agg = nn.Conv2d(features, features, kernel_size=3, stride=1,
                                       padding=1, bias=True)
            self.norm_q = LayerNorm2d(features)
            self.q_proj = nn.Linear(features, features, bias=False)

            # 2. K/V 来自卷积旁路
            self.norm_kv = LayerNorm2d(features)
            self.k_proj = nn.Linear(features, features, bias=False)
            self.v_proj = nn.Linear(features, features, bias=False)

            # 3. 窗口交叉注意力 -> 加性注意力支路 (无门控)
            self.attn = WindowCrossAttention(features, num_heads=num_heads,
                                             window_size=window_size, dropout=dropout)
            self.o_proj = nn.Linear(features, features, bias=True)

        # 4. FeatX 子 token 支路 (patch 相对偏移条件的 FiLM + MLP)
        self.featx = OffsetFeatX(features, dropout=dropout) if use_featx else None

        # 初始化: q/k/v 投影小非零 (trunc normal 0.02) -> 注意力内部第 0
        # 步即有信息交换; o_proj 权重+bias 零初始化 (ReZero 式) -> 第 0 步
        # x_attn ≡ 0 (精确 warm start), o_proj 自身立即获得梯度 (z≠0),
        # 上游随 ||W_o|| 增长快速解锁。注意: 不能对 FeatX 递归 apply
        # (会破坏其最后一层零初始化), 故显式初始化本模块自己的层。
        if use_attn:
            _trunc_normal_(self.query_agg.weight, std=0.02)
            nn.init.zeros_(self.query_agg.bias)
            for lin in (self.q_proj, self.k_proj, self.v_proj):
                _trunc_normal_(lin.weight, std=0.02)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            nn.init.zeros_(self.o_proj.weight)
            nn.init.zeros_(self.o_proj.bias)

    def _to_tokens(self, x):
        """(B, C, H, W) -> (B, H*W, C), then apply a linear and go back."""
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def _to_spatial(self, x, B, C, H, W):
        """(B, H*W, C) -> (B, C, H, W)."""
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def query_context(self, residual):
        """Aggregate each residual pixel with its 3x3 neighbourhood.

        与 DPT_DyUp 的 `UpBlock.query_context` 定义完全一致: 动态核生成器
        消费的正是这个张量, `forward` 在外部传入时复用它 (每尺度只聚合一次)。
        """
        if not self.use_attn:
            raise RuntimeError(
                "query_context() is unavailable: this UpBlock was built with "
                "use_attn=False (coarse scale, attention branch omitted)")
        return F.gelu(self.query_agg(residual))

    def forward(self, kv, residual, query_context=None):
        """Args:
            kv      (tensor): conv-bypass feature at scale i   (B, C, H, W)
                              `use_attn=False` 时可为 None (不参与计算)
            residual(tensor): reassembled DPT feature layer_X_rn at scale i
                              (B, C, H, W) -> query 来源 + 被注入的对象
                              (DPT 同款注入点: reassemble 之后, refinenet 之前)
            query_context (tensor, optional): 预先算好的
                              `self.query_context(residual)`, 由动态核通路传入,
                              避免重复做 3x3 聚合。
        Returns:
            refined feature: residual + x_attn + x_subtoken (ViT-Up 三支路).
        """
        B, C, H, W = residual.shape

        x_attn = 0.0
        if self.use_attn:
            # 1. query: 3x3 conv 局部聚合周边像素 (判断边缘/纹理等局部结构)
            x = self.query_context(residual) if query_context is None else query_context
            q = self._to_spatial(self.q_proj(self._to_tokens(self.norm_q(x))), B, C, H, W)

            # 2. K/V 来自卷积旁路
            k = self._to_spatial(self.k_proj(self._to_tokens(self.norm_kv(kv))), B, C, H, W)
            v = self._to_spatial(self.v_proj(self._to_tokens(self.norm_kv(kv))), B, C, H, W)

            # 3. 窗口交叉注意力 -> 加性注意力支路 (位置信息 = 窗口内相对偏置)
            z = self.attn(q, k, v, H, W)  # (B, C, H, W)
            x_attn = self._to_spatial(self.o_proj(self._to_tokens(z)), B, C, H, W)

        # 4. FeatX 子 token 支路 (最后一层零初始化 -> 第 0 步 ≡ 0)
        x_sub = self.featx(residual, H, W, self.patch_scale) \
            if self.featx is not None else 0.0

        # 5. ViT-Up 三支路融合 (无门控, 无 transition/fusion MLP)
        return residual + x_attn + x_sub
