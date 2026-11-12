from pathlib import Path
import torch
from .dpt.models import DPTSegmentationModel

CHECKPOINTS = {k: Path(__file__).resolve().parent / "weights" / v for k, v in {
    "ade20k": "dapt_dino_ade20k.pth", "nyuv2": "dapt_dino_nyuv2.pth"}.items()}
NUM_CLASSES = {"ade20k": 150, "nyuv2": 41}


def build_model(dataset="ade20k"):
    if dataset not in NUM_CLASSES:
        raise ValueError("Choose ade20k or nyuv2.")
    model = DPTSegmentationModel(
        NUM_CLASSES[dataset], backbone="dinov2_vitb14", features=256,
        bypass_type="v2", bypass_shallow_ch=64, bypass_shallow_hidden=128,
        bypass_attn_scales=(0, 1), bypass_num_heads=4,
        bypass_window_size=16, bypass_use_featx=True,
    )
    model.freeze_encoder(freeze_readout=False)
    return model


def load_model(dataset="ade20k", checkpoint=None, device="cpu"):
    if dataset not in CHECKPOINTS:
        raise ValueError("Choose ade20k or nyuv2.")
    checkpoint = Path(checkpoint) if checkpoint else CHECKPOINTS[dataset]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Full DAPT-DINO checkpoint not found: {checkpoint}")
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if obj.get("metadata", {}).get("task", dataset) != dataset:
        raise ValueError(f"The checkpoint is not the {dataset} segmentation model.")
    model = build_model(dataset)
    model.load_state_dict(obj.get("model", obj), strict=True)
    return model.to(device).eval()
