"""Run all three demos against their matching RGB input folders by default."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "monodepth", "ade20k", "nyuv2"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32"], default="auto")
    args = parser.parse_args()
    for task in (["monodepth", "ade20k", "nyuv2"] if args.task == "all" else [args.task]):
        relative = "non_dino/demo.py" if task == "monodepth" else "dino/demo.py"
        command = [sys.executable, str(ROOT / relative), "--device", args.device, "--precision", args.precision]
        if task != "monodepth":
            command += ["--dataset", task]
        subprocess.run(command, check=True)
if __name__ == "__main__":
    main()
