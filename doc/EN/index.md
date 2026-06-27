# English Documentation Index

This directory is the English documentation entry point for `yolo26-rs`. It covers Rust/Candle inference and training for YOLO26 / YOLOE-26, official `.pt` weight loading/export, and end-to-end examples organized by usage scenario.

Scope boundary: this repository only covers YOLO26 / YOLOE-26. APIs, models, and features in Ultralytics documentation or source code that belong to YOLOv8, YOLO11, YOLO-World, SAM, or other non-YOLO26 families are not implementation targets for this crate. When they are mentioned in the documentation, it is only to explain official background or differences.

## Documentation Navigation

### API Reference

- [Task Inference API](tasks.md): `detect`, `segment`, `semantic`, `classify`, `pose`, `obb`.
- [Training API](train.md): `train` feature, `Model`, `Session`, dataset adapters.
- [YOLOE API](yoloe.md): text prompt, visual prompt, prompt-free, RepRTA, SAVPE, LRPC.

### Usage Guides

- [Usage Guide Index](usage/index.md): grouped by task, YOLOE, and training scenarios.
- [Object detection detect](usage/detect.md): official `.pt` loading, CPU/GPU, FP16, class filtering.
- [Image classification classify](usage/classify.md): ImageNet-style whole-image classification, top-k.
- [Instance segmentation segment](usage/segment.md): detection boxes + masks, high-resolution masks.
- [Semantic segmentation semantic](usage/semantic.md): dense per-pixel class maps.
- [Pose estimation pose](usage/pose.md): detection boxes + keypoint skeletons.
- [Oriented box detection obb](usage/obb.md): DOTA-style boxes with angles.
- [SAHI sliced detection](usage/sahi.md): high-resolution small-object sliced inference + merging.
- [YOLOE text prompt](usage/yoloe-text.md): class-name lists generate text embedding classifiers.
- [YOLOE visual prompt](usage/yoloe-visual.md): box/mask examples retrieve same-class objects.
- [YOLOE prompt-free / LRPC](usage/yoloe-promptfree.md): static open-vocabulary inference without prompts.
- [Instance segmentation training train-seg](usage/train-seg.md): standard YOLO26 segmentation training loop.
- [YOLOE training train-yoloe](usage/train-yoloe.md): YOLOE text/visual/prompt-free training.

### Supplemental Documentation

- [Candle 0.10.2 Patch Guide](candle-0.10.2-patch-guide.md): Candle patches relevant to training.

## Quick Start

Minimal inference example (full API in [tasks.md](tasks.md)):

```rust
use yolo26_rs::{FilterOption, Image, detect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let detections = model.predict(&image, &FilterOption::default())?;
    Ok(())
}
```

Build/verify:

```bash
cargo build --release
cargo build --release --features train
```

Do not use `--all-features` as the default verification command when the local machine does not have the CUDA toolkit; the `cuda` feature builds `cudarc` and requires `nvcc`.

## Relationship to Official Ultralytics

- Official Ultralytics is the Python/PyTorch ecosystem. It loads `.pt` directly and provides full trainers, augmentation, validation, export, and deployment tools.
- This crate is a Rust/Candle implementation. Official `.pt` is the main weight format (default `pt` feature), and both inference and training run on the Rust side. After training, `save_pt` can write back official PyTorch-readable `.pt` files.
- YOLOE export semantics match the official behavior: dynamic prompts must be baked before export; after export, the model no longer accepts new text/visual prompts.
