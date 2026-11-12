"""Default RGB inputs, task preprocessing and demo output generation."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
FIG_DIRS = {"monodepth": ROOT / "fig/monodepth", "ade20k": ROOT / "fig/seg/ade20k", "nyuv2": ROOT / "fig/seg/nyuv2"}
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in SUFFIXES:
            raise ValueError(f"Unsupported RGB image: {path}")
        return [path], path.parent
    if not path.exists():
        path.mkdir(parents=True)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES), path


def prepare_depth(image):
    width, height = image.size
    scale = min(576 / width, 448 / height)
    nw, nh = max(1, round(width * scale)), max(1, round(height * scale))
    ox, oy = (576 - nw) // 2, (448 - nh) // 2
    x = np.full((448, 576, 3), .5, dtype=np.float32)
    x[oy:oy+nh, ox:ox+nw] = np.asarray(image.resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return (x - .5) / .5, (ox, oy, nw, nh)


def prepare_segmentation(image, dataset):
    width, height = image.size
    if dataset == "nyuv2":
        nw, nh = 588, 448
    else:
        scale = 448 / min(width, height)
        nw, nh = [max(28, round(v * scale / 28) * 28) for v in (width, height)]
        if max(nw, nh) > 2048:
            scale = 2048 / max(nw, nh)
            nw, nh = [max(28, round(v * scale / 28) * 28) for v in (nw, nh)]
    x = np.asarray(image.resize((nw, nh), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return (x - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)


def native_labels(logits, height, width):
    import torch
    import torch.nn.functional as F
    logits = logits.float()
    if height * width <= 1_000_000:
        return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
    result = np.empty((height, width), dtype=np.uint8)
    xs = (torch.arange(width, device=logits.device, dtype=torch.float32) + .5) * (2 / width) - 1
    step = max(1, 200_000 // width)
    for start in range(0, height, step):
        end = min(height, start + step)
        ys = (torch.arange(start, end, device=logits.device, dtype=torch.float32) + .5) * (2 / height) - 1
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        block = F.grid_sample(logits, torch.stack((xx, yy), -1)[None], mode="bilinear", padding_mode="border", align_corners=False)
        result[start:end] = block.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return result


def comparison(images, labels):
    width, height = images[0].size
    out = Image.new("RGB", (len(images) * width, height + 30), "white")
    draw = ImageDraw.Draw(out)
    for i, (image, name) in enumerate(zip(images, labels)):
        out.paste(image.convert("RGB"), (i * width, 30))
        draw.text((i * width + 8, 8), name, fill="black")
    return out


def run_task(task, args):
    import contextlib
    paths, input_root = find_images(args.input or FIG_DIRS[task])
    if not paths:
        print(f"No RGB images found. Put your RGB images in: {input_root.resolve()}")
        return
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or a compatible PyTorch build.")
    output = args.output or (ROOT / "output/monodepth" if task == "monodepth" else ROOT / "output/seg" / task)
    if task == "monodepth":
        from non_dino.model import load_model, DEFAULT_CHECKPOINT
        checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
        model = load_model(checkpoint, device)
    else:
        from dino.model import load_model, CHECKPOINTS
        checkpoint = args.checkpoint or CHECKPOINTS[task]
        model = load_model(task, checkpoint, device)
        palettes = json.loads((ROOT / "dino/palettes.json").read_text(encoding="utf-8"))
        palette = np.array(palettes[task]["rgb"], dtype=np.uint8)
    for path in paths:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if task == "monodepth":
            x, (ox, oy, nw, nh) = prepare_depth(image)
        else:
            x = prepare_segmentation(image, task)
        data = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(device)
        amp = contextlib.nullcontext()
        if device.type == "cuda" and args.precision != "fp32":
            amp = torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        with torch.inference_mode(), amp:
            logits = model(data)
        target = Path(output) / path.relative_to(input_root).parent / (path.stem + "_" + path.suffix[1:].lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        meta = {"task": task, "input": path.name, "checkpoint": Path(checkpoint).name, "network_input_hw": list(x.shape[:2]), "output_hw": [image.height, image.width]}
        with torch.inference_mode():
            if task == "monodepth":
                pred = F.interpolate(logits.float().reshape(1, 1, *logits.shape[-2:]), size=(448, 576), mode="bilinear", align_corners=False)
                pred = F.interpolate(pred[..., oy:oy+nh, ox:ox+nw], size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                if not np.isfinite(pred).all():
                    raise RuntimeError(f"Non-finite prediction for {path}")
                np.save(str(target) + "_inv_depth.npy", pred.astype(np.float32))
                lo, hi = np.percentile(pred, [1, 99])
                preview = Image.fromarray(np.rint(np.clip((pred-lo) / max(float(hi-lo), 1e-8), 0, 1) * 255).astype(np.uint8))
                preview.save(str(target) + "_depth.png")
                comparison([image, preview], ["RGB", "DAPT relative depth (near=bright)"]).save(str(target) + "_comparison.png")
                meta.update(raw_output="relative inverse depth; arbitrary scale/shift, not meters", visualization_percentiles=[float(lo), float(hi)])
            else:
                pred = native_labels(logits, image.height, image.width)
                colored = Image.fromarray(palette[pred])
                overlay = Image.fromarray(np.rint(np.asarray(image)*.55 + palette[pred]*.45).astype(np.uint8))
                Image.fromarray(pred).save(str(target) + "_labels.png")
                colored.save(str(target) + "_prediction.png")
                overlay.save(str(target) + "_overlay.png")
                comparison([image, colored, overlay], ["RGB", "DAPT + DINOv2", "Overlay"]).save(str(target) + "_comparison.png")
                meta.update(label_ids="0..149; add 1 for ADE annotation IDs" if task == "ade20k" else "1..40 semantic classes; output 0 is unlabeled", classes_present=[{"id":int(k),"name":palettes[task]["names"].get(str(k),str(k)),"rgb":palette[k].tolist()} for k in np.unique(pred)])
        Path(str(target) + "_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {target}", flush=True)


def variant_main(variant):
    parser = argparse.ArgumentParser(description="Generate DAPT demo outputs from RGB images.")
    if variant == "dino":
        parser.add_argument("--dataset", choices=["ade20k", "nyuv2"], default="ade20k")
    for name in ["input", "output", "checkpoint"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32"], default="auto")
    args = parser.parse_args()
    run_task("monodepth" if variant == "non_dino" else args.dataset, args)
