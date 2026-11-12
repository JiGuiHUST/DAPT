"""Build the bundled DINOv2 encoder; full task checkpoints supply all weights."""
import sys
from pathlib import Path
DINOV2_B_DIM = 768
DINOV2_B_PATCH = 14
def load_dinov2(name="dinov2_vitb14_reg", verbose=False):
    if name != "dinov2_vitb14_reg":
        raise ValueError("Released models use DINOv2-B/14 with four register tokens.")
    source = Path(__file__).resolve().parents[1] / "third_party" / "dinov2"
    if not (source / "dinov2/hub/backbones.py").is_file():
        raise FileNotFoundError(f"Bundled DINOv2 source missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vitb14_reg
    return dinov2_vitb14_reg(pretrained=False)
