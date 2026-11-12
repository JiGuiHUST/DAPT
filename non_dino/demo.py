"""Default inputs: ../fig/monodepth; default weights: weights/dapt_monodepth.pth."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_utils import variant_main
if __name__ == "__main__":
    variant_main("non_dino")
