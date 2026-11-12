# Third-party components

- The DPT model implementation is derived from DPT (Intel ISL). Its original MIT license and copyright notice are retained in `LICENSE` and the two model folders. The DAPT pathway, fusion blocks, DINOv2 integration and demo interfaces are included in this distribution.
- `dino/third_party/dinov2` contains the DINOv2 encoder implementation by Meta Platforms, Inc. and affiliates, with the original Apache-2.0 license and source headers preserved. Only the model/layer/hub code needed by the encoder is bundled. The public loader constructs this architecture with `pretrained=False`; all trained encoder tensors come from the included complete task checkpoints.
- PyTorch, torchvision, timm, NumPy and Pillow are installed dependencies and retain their respective upstream licenses.
- ADE20K class colors/names follow the DPT/ADE20K convention; NYUv2 uses the repository's 40-class IDs and VOC-style palette. `dino/palettes.json` associates model output IDs with those colors and names.

No datasets, example photographs, training logs, optimizer states, server paths, account configuration or caches are included in this folder. Add your own RGB inputs to `fig` to generate demos.
