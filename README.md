# DAPT

DAPT 模型与 RGB 演示代码。`non_dino` 使用 ViT-L/16 编码器进行单目深度估计；`dino` 使用 DINOv2-B/14（4 个 register token）进行语义分割。两套模型源码、权重分开保存。

```text
DAPT_public/
├── non_dino/                 # 非 DINO 模型代码
│   ├── dpt/
│   ├── model.py
│   ├── demo.py
│   └── weights/dapt_monodepth.pth
├── dino/                     # DINO 编码器模型代码
│   ├── dpt/
│   ├── third_party/dinov2/   # 随仓库提供的编码器源码及许可证
│   ├── model.py
│   ├── demo.py
│   ├── palettes.json
│   └── weights/
│       ├── dapt_dino_ade20k.pth
│       └── dapt_dino_nyuv2.pth
├── fig/                      # 初始为空，放置自己的 RGB 图片
│   ├── monodepth/
│   └── seg/
│       ├── ade20k/
│       └── nyuv2/
├── demo.py
└── weights_manifest.json
```

## 权重

| 文件 | 编码器 | 用途与选择 |
|---|---|---|
| `non_dino/weights/dapt_monodepth.pth` | ViT-L/16 | 最新六数据集联合续训保留的唯一综合最优单目权重，`joint_polish_1200` |
| `dino/weights/dapt_dino_ade20k.pth` | DINOv2-B/14 reg4 | ADE20K，150 类，最终第 10,000 步权重 |
| `dino/weights/dapt_dino_nyuv2.pth` | DINOv2-B/14 reg4 | NYUv2，41 个输出通道（40 个语义类），最终第 6,000 步权重 |

三份权重均包含完整编码器、DAPT 解码器和任务头。导出时只去除训练记录，参数数值和精度保持不变；无需另外下载 MiDaS/DINOv2 预训练权重。文件大小、SHA256、模型设置和已测试环境见 `weights_manifest.json`。`.pth` 已配置 Git LFS 属性。

## 安装与生成 demo

服务器实测 Python 3.10.20。`requirements.txt` 已按服务器锁定 PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、timm、NumPy、Pillow 及推理所需依赖版本，包含官方 CUDA 12.4 wheel 源；对应组合可查阅 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)。

```bash
pip install -r requirements.txt
```

把 RGB 图片分别放入三个 `fig` 子目录，然后在仓库根目录运行：

```bash
python demo.py
```

此命令按文件夹依次生成三种任务的结果，自动选择对应权重。有 RGB 图片的任务才加载模型；空目录会提示输入位置。支持 JPEG、PNG、BMP、TIFF、WebP 和子目录。空输入目录在缺失时会自动创建。

也可分别运行。默认输入、权重与输出路径按仓库位置定位，不依赖启动时的工作目录；手动指定的相对路径按当前工作目录解释：

```bash
python non_dino/demo.py
python dino/demo.py --dataset ade20k
python dino/demo.py --dataset nyuv2

# 自定义输入、输出或计算设备
python non_dino/demo.py --input path/to/rgb --output output/my_depth --device cuda
python dino/demo.py --dataset nyuv2 --input path/to/rgb --device cpu

# 指定权重；使用完整 FP32 推理
python dino/demo.py --dataset ade20k --checkpoint dino/weights/dapt_dino_ade20k.pth --precision fp32
```

默认输出分别位于 `output/monodepth/`、`output/seg/ade20k/`、`output/seg/nyuv2/`。原始输入不会被覆盖。各图片会保存：

- **单目深度**：原图尺寸的 `*_inv_depth.npy`、近亮远暗的 `*_depth.png`、RGB/预测并列的 `*_comparison.png`、预处理与输出说明。
- **分割**：类别 ID `*_labels.png`、彩色预测 `*_prediction.png`、RGB 叠加 `*_overlay.png`、三列对比图、类别名称和颜色说明。

单目原始输出是**相对逆深度**，具有尺度/偏移不确定性，不能直接当作米制深度；深度预览按单图第 1–99 百分位归一化，不需要 GT。分割 ADE20K 标签 ID 为 0–149（与原始标注比较时加 1），NYUv2 为 1–40，额外输出 0 表示未标注。放大类别 ID 图请使用最近邻插值。

预处理与原权重对应：非 DINO 使用 448×576 等比例留边和 `(RGB/255−0.5)/0.5`；DINO 使用 ImageNet 均值/方差，ADE20K 短边 448、尺寸取 28 的倍数，NYUv2 为 448×588。输出恢复到输入 RGB 的尺寸。

## 在代码中加载模型

```python
from non_dino.model import load_model as load_depth
from dino.model import load_model as load_segmentation

depth_model = load_depth(device="cuda")
ade_model = load_segmentation("ade20k", device="cuda")
nyu_model = load_segmentation("nyuv2", device="cuda")
```

输入处理及保存结果可直接复用 `demo_utils.py`。加载采用 `strict=True`；使用错误的模型或任务权重会报错。

许可证和第三方来源见 `LICENSE`、`THIRD_PARTY_NOTICES.md` 及 `dino/third_party/dinov2/LICENSE`。
