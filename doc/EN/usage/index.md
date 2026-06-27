# Usage Guide Index

This directory contains `yolo26-rs` English documentation organized **by usage scenario**. Each page corresponds to a typical workflow and includes runnable Rust examples, matching official Python code, required features, and details about the related APIs/configuration.

It complements the module-oriented API reference in the parent `doc/EN/` directory (`tasks.md`/`yoloe.md`/`train.md`): this directory is the "end-to-end usage guide", while the API reference is the "item-by-item manual". The two sets of documents cross-link each other.

## Scenario Navigation

### Inference (`detect` is default; other tasks require matching features)

- [Object detection detect](detect.md): find objects in an image and return their positions and classes.
- [Image classification classify](classify.md): classify the whole image into the most likely categories.
- [Instance segmentation segment](segment.md): identify each object and separate its occupied pixels.
- [Semantic segmentation semantic](semantic.md): assign a class to every pixel in the image.
- [Pose estimation pose](pose.md): locate object keypoints to describe poses or skeletons.
- [Oriented box detection obb](obb.md): detect targets with orientation angles, useful for aerial, remote-sensing, and rotated-object scenes.
- [SAHI sliced detection sahi](sahi.md): split large images into small tiles to improve dense small-object recall.

> Default features only include `detect` + `image` + `pt`. classify/segment/semantic/pose/obb each require `--features <task>` (see Feature Quick Reference below).

### YOLOE (requires `--features yoloe`)

- [YOLOE text prompt](yoloe-text.md): specify target classes with text for open-vocabulary detection or segmentation.
- [YOLOE visual prompt](yoloe-visual.md): provide boxes or masks as examples so the model finds same-class objects.
- [YOLOE prompt-free / LRPC](yoloe-promptfree.md): predict directly from the built-in open vocabulary without manual prompts.

> `yoloe` is an aggregate feature (= `yoloe-text` + `yoloe-visual` + `yoloe-pf`). For text prompt only, use `--features yoloe-text`; for visual prompt, use `--features yoloe-visual`; for prompt-free (including the built-in 4585-name vocabulary), use `--features yoloe-pf`. Text prompt requires callers to explicitly provide CLIP weight and tokenizer paths (see [yoloe-text.md](yoloe-text.md)).

### Training (requires `train` feature)

- [Instance segmentation training train-seg](train-seg.md): standard YOLO26 segmentation training loop.
- [YOLOE training train-yoloe](train-yoloe.md): YOLOE text/visual/prompt-free training, fine-tune/from-scratch, and related flows.

## Feature Quick Reference

| Capability | Feature | When needed |
| --- | --- | --- |
| Detection inference / SAHI / image loading / `.pt` loading | default (`detect`+`image`+`pt`) | Works out of the box |
| Classification inference | `--features classify` | |
| Instance segmentation inference | `--features segment` | |
| Semantic segmentation inference | `--features semantic` | |
| Pose inference | `--features pose` | |
| OBB inference | `--features obb` | |
| YOLOE inference | `--features yoloe` | |
| Training (all tasks + YOLOE) | `--features train` | Automatically includes all tasks + `image` |
| Apple GPU | `--features metal` | macOS Metal backend |
| NVIDIA GPU | `--features cuda` | Requires CUDA toolkit + nvcc |
| Browser | `--features wasm` | wasm32 target |

> Do not use `--all-features` without the CUDA toolkit; the `cuda` feature requires `nvcc`. The default verification command is `cargo build --features train`.

## Minimal Quick Start (detect)

```rust
use yolo26_rs::{FilterOption, Image, detect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let detections = model.predict(&image, &FilterOption::default())?;
    Ok(())
}
```

Full API reference entry: [index.md](../index.md).
