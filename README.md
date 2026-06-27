# yolo26-rs

[Chinese](README_CN.md) | English

`yolo26-rs` is a pure Rust YOLO26 / YOLOE-26 runtime built on [Candle](https://github.com/huggingface/candle). The default path focuses on inference; enabling the `train` feature provides native Candle training APIs and CLI support.

Supported tasks:

| Task | Module | Weight file example | Output type |
| --- | --- | --- | --- |
| Object detection | `detect` | `yolo26s.pt` | `Vec<detect::Prediction>` |
| Image classification | `classify` | `yolo26s-cls.pt` | `Vec<classify::Prediction>` |
| Instance segmentation | `segment` | `yolo26s-seg.pt` | `Vec<segment::Prediction>` |
| Pose estimation | `pose` | `yolo26s-pose.pt` | `Vec<pose::Prediction>` |
| Semantic segmentation | `semantic` | `yolo26s-sem.pt` | `semantic::Prediction` |
| Oriented box detection | `obb` | `yolo26s-obb.pt` | `Vec<obb::Prediction>` |
| YOLOE text prompt | `yoloe::segment` | `yoloe-26s-seg.pt` | `Vec<segment::Prediction>` |
| YOLOE visual prompt | `yoloe::segment` | `yoloe-26s-seg.pt` | `Vec<segment::Prediction>` |
| YOLOE prompt-free | `yoloe::segment` | `yoloe-26s-seg-pf.pt` | `Vec<segment::Prediction>` |

Model sizes use `Scale::{N, S, M, L, X}`, matching the `n/s/m/l/x` suffix in weight file names.

## Setup

Prefer official `.pt` weights. Download the matching YOLO26 / YOLOE-26 checkpoints from [Ultralytics GitHub Releases](https://github.com/ultralytics/assets/releases), then load them directly with `Model::from_file(...)` or `Model::from_pt_file(...)`.

Weight naming convention:

```text
yolo26{size}.pt                    # detection
yolo26{size}-{task}.pt             # cls / seg / pose / sem / obb
yoloe-26{size}-seg.pt              # YOLOE text / visual prompt
yoloe-26{size}-seg-pf.pt           # YOLOE prompt-free
```

CPU inference is the default. When used as a dependency, hardware backends can be enabled through Cargo features:

```bash
# CPU (default)
cargo add yolo26-rs

# macOS Metal
cargo add yolo26-rs --features metal

# CUDA (requires nvcc)
cargo add yolo26-rs --features cuda
```

If you need `.safetensors`, converted weights are also available at [wzh19960613/yolo26-safetensors](https://huggingface.co/wzh19960613/yolo26-safetensors).

When writing examples or integration tests directly in this repository, use the crate name `yolo26_rs`.

## Quick Start: Detection

```rust
use std::error::Error;

use yolo26_rs::{FilterOption, Image, detect};

fn main() -> Result<(), Box<dyn Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;

    let detections = model.predict(&image, &FilterOption::default())?;
    for det in detections {
        println!(
            "class={} conf={:.3} bbox={:?}",
            det.class_id, det.confidence, det.bbox,
        );
    }

    Ok(())
}
```

## Other Tasks

All task loading flows are the same: read weights, construct a config, create a `Model`, then call `predict`.

```rust
// Classification: returns class scores sorted by confidence.
let model = classify::Model::from_file("yolo26s-cls.pt")?;
let scores = model.predict(&image)?;
```

```rust
// Instance segmentation: returns detection + mask.
let masks = segment_model.predict(
    &image,
    &FilterOption::default(),
    &MaskOption {
        high_resolution: true,
    },
)?;
```

```rust
// Pose estimation: returns bbox, class_id, confidence, and keypoints.
let poses = pose_model.predict(&image, &FilterOption::default())?;
```

```rust
// Semantic segmentation: returns per-class logits; class_ids() gives the argmax class map.
let semantic = semantic_model.predict(
    &image,
    &MaskOption {
        high_resolution: true,
    },
)?;
let class_map = semantic.class_ids();
```

```rust
// OBB: returns rotated boxes with center, size, and angle in bbox.
let obbs = obb_model.predict(&image, &FilterOption::default())?;
```

## SAHI Sliced Detection

The detection module includes SAHI-style sliced inference for large images or small-object scenarios:

```rust
use yolo26_rs::detect::sahi;

let options = sahi::Options {
    slice_width: 320,
    slice_height: 320,
    overlap_width_ratio: 0.25,
    overlap_height_ratio: 0.25,
    include_full_image: true,
    ..sahi::Options::default()
};

let detections = model.predict_sahi(
    &image,
    &FilterOption::default(),
    &options,
)?;
```

You can also call `detect::sahi::sliced_predict(&model, &image, &filter, &options)` directly.

## API Design Overview

The public API is organized by task, and each task module follows the same structure:

```text
detect / classify / segment / pose / semantic / obb
|-- Config           # model config type for the task
|-- config_builder() # config builder with task defaults
|-- Model            # inference model with loaded weights
`-- Prediction       # strongly typed prediction result for the task
```

Core design points:

- `Image` is the unified input type. It stores width, height, and RGB pixel bytes (3 channels).
- `config_builder()` provides task defaults, such as COCO class count for detection/segmentation, ImageNet class count for classification, DOTA class count for OBB, and Cityscapes class count for semantic segmentation.
- `Model::from_file(path)` auto-detects `.pt` / `.safetensors` and infers scale and class count from checkpoint shapes; `Model::from_file_with(path, config)` can override device, dtype, input size, and related settings.
- `predict(...)` is the high-level API. It performs preprocessing, forward inference, and postprocessing, and returns results in source-image coordinates.
- `forward_tensor(...)` is the low-level API for callers with custom preprocessing or custom postprocessing.
- `FilterOption` controls confidence and class filtering for detection-style tasks; `MaskOption` controls whether segmentation outputs are returned at source-image resolution.
- `default_labels` (requires the `default_labels` feature, disabled by default) provides `COCO`, `IMAGENET`, `CITYSCAPES`, and `DOTA` class name tables. Prediction results store numeric `class_id` values only. With `.pt` weights, checkpoint class names can also be read directly through `pt_loader::load_pt_metadata().names`.

## Common Config Options

```rust
let config = detect::config_builder()
    .with_scale(Scale::S)              // model size; default loading infers this from the checkpoint
    .with_device(device::auto())       // CPU / Metal / CUDA
    .with_input_size(640)              // square input, snapped to a multiple of 32
    .with_max_predictions(300)         // maximum retained predictions
    .with_labels_count(80)             // for custom models; default loading infers this from the checkpoint
    .build();
```

The default dtype is `Auto`: on GPU it follows the weight dtype, while on CPU / wasm32 it uses F32. Call `with_dtype(...)` only when you need to force precision.

`with_image_size(width, height)` and `with_input_shape(ImageSize)` support non-square inputs. Sizes are snapped upward to multiples of 32 through `ImageSize::snapped()`.

## Features

Default features are `detect`, `image`, and `pt` (detection + image loading + `.pt` loading). Enable the rest as needed:

| Feature | Purpose |
| --- | --- |
| `detect` (default) | Detection inference + SAHI |
| `image` (default) | `Image::from_file` image loading |
| `pt` (default) | Load official `.pt`; `save_pt` / `Model::save_pt` / `Seg::save_pt` write official-readable `.pt` files |
| `classify` | Classification inference |
| `segment` | Instance segmentation inference |
| `semantic` | Semantic segmentation inference |
| `pose` | Pose inference |
| `obb` | OBB inference |
| `yoloe` | YOLOE open-vocabulary inference (aggregate = `yoloe-text` + `yoloe-visual` + `yoloe-pf`) |
| `yoloe-text` | Text-prompt inference path, depends on external `mobileclip2-b-rs` |
| `yoloe-visual` | Visual-prompt inference path (SAVPE encoding + visuals) |
| `yoloe-pf` | Prompt-free LRPC path and built-in 4585-name `LRPC_VOCAB` |
| `default_labels` | Built-in standard dataset class name tables (`COCO`/`DOTA`/`CITYSCAPES`/`IMAGENET`); disabled by default and only compiled in when enabled |
| `train` | Training API (automatically includes all tasks + YOLOE + `image`) |
| `cuda` | Enable Candle CUDA backend |
| `metal` | Enable Candle Metal backend |
| `wasm` | Enable wasm-bindgen bindings for browser-side detection inference |

`device::auto()` tries CUDA or Metal according to enabled features, then falls back to CPU if unavailable.

## YOLOE

Official reference: [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/).

The crate's `yoloe` module supports the three YOLOE-26 inference paths: text prompt, visual prompt, and prompt-free. Full API documentation is in [doc/EN/yoloe.md](doc/EN/yoloe.md).

Text prompt:

Required resources:

- YOLOE segment checkpoint: [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt).
- MobileCLIP2-B text encoder weights: [mobileclip2_b.pt](https://huggingface.co/apple/MobileCLIP2-B/resolve/main/mobileclip2_b.pt).
- CLIP BPE tokenizer: [tokenizer.json](https://huggingface.co/openai/clip-vit-base-patch32/raw/main/tokenizer.json).

The text encoder uses [wzh19960613/mobileclip2-b-rs](https://github.com/wzh19960613/mobileclip2-b-rs) through the `yoloe-text` feature.

```rust
use yolo26_rs::{
    FilterOption, Image, MaskOption,
    yoloe::{ClipTextEncoder, Session, segment::Model},
};

let image = Image::from_file("examples/bus.jpg")?;
let model = Model::from_file("yoloe-26s-seg.pt")?;
let encoder = ClipTextEncoder::from_files(
    "mobileclip2_b.pt",
    "tokenizer.json",
)?;
let session = Session::text(&encoder, model.reprta(), ["person", "bus"])?;
let predictions = model.predict(
    &image,
    &session,
    &FilterOption::default(),
    &MaskOption::default(),
)?;
```

Visual prompt:

```rust
use yolo26_rs::{
    FilterOption, Image, MaskOption,
    yoloe::{Session, Visual, VisualSource, segment::Model},
};

let image = Image::from_file("examples/bus.jpg")?;
let model = Model::from_file("yoloe-26s-seg.pt")?;
let prompts = vec![Visual::from_box(0, [10.0, 20.0, 90.0, 160.0])?];
let session = Session::visual(prompts.clone())?;
let predictions = model.predict_visual_prompts(
    &image,
    &prompts,
    VisualSource::Boxes,
    &session,
    &FilterOption::default(),
    &MaskOption::default(),
)?;
```

Prompt-free:

```rust
use yolo26_rs::{
    FilterOption, Image, MaskOption,
    yoloe::{Session, segment::Model},
};

let image = Image::from_file("examples/bus.jpg")?;
let model = Model::from_file("yoloe-26s-seg-pf.pt")?;
let session = Session::prompt_free_default()?;
let predictions = model.predict_prompt_free(
    &image,
    &session,
    &FilterOption::default(),
    &MaskOption::default(),
)?;
```

## Training

> **Training has not been rigorously tested. Use it carefully; training quality, convergence, and numeric equivalence with the official PyTorch Trainer are not guaranteed. Validate the full flow on a small dataset and a short run before using it for real training.**

Training depends on Candle autograd and backend convolution implementations. Candle 0.10.2 has upsample-nearest backward and Metal non-contiguous convolution kernel paths that can affect gradient correctness. For training, especially on Metal, first apply the local patched `candle-core` described in the [Candle 0.10.2 Patch Guide](doc/EN/candle-0.10.2-patch-guide.md), or wait for an upstream version that contains the fixes.

```rust
use yolo26_rs::{Scale, detect, train};

let yaml = train::dataset::ultralytics::Yaml::from_file("data.yaml")?;
let config = train::ModelConfig::Detect(
    detect::config_builder()
        .with_scale(Scale::S)
        .with_labels_count(yaml.names.len())
        .build(),
);
let mut model = train::Model::from_pt_file("yolo26s.pt", config)?;
model.set_class_names(yaml.names.clone())?;
let mut session = train::Session::new(
    model,
    train::OptimizerConfig::AdamW {
        params: Default::default(),
    },
)?;

// session.train_batch(&input, &target)?;
// session.model().save_pt("trained.pt")?;
```
