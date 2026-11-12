from pathlib import Path
import torch
from .dpt.models import DPTDepthModel

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "weights" / "dapt_monodepth.pth"


def build_model():
    return DPTDepthModel(
        backbone="vitl16_384", features=256, non_negative=True, use_bn=False,
        bypass_type="v2", bypass_shallow_ch=64, bypass_shallow_hidden=128,
        bypass_attn_scales=(0, 1), bypass_num_heads=4,
        bypass_window_size=16, bypass_use_featx=True,
    )


def load_model(checkpoint=None, device="cpu"):
    checkpoint = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Full DAPT depth checkpoint not found: {checkpoint}")
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if obj.get("metadata", {}).get("task", "monodepth") != "monodepth":
        raise ValueError("A non-DINO monocular-depth checkpoint is required.")
    model = build_model()
    model.load_state_dict(obj.get("model", obj), strict=True)
    return model.to(device).eval()
