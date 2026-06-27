# Task Inference API

This page covers the six task root modules: `detect`, `segment`, `semantic`, `classify`, `pose`, and `obb`. Every task follows the same shape: `Model::from_file()` loads weights, `predict()` returns strongly typed results, and `forward_tensor()` exposes low-level tensor forwarding. Use `config_builder()` when you need to override the default device, dtype, or input size.

## Quick Start

```rust
use yolo26_rs::{FilterOption, Image, detect};

let image = Image::from_file("examples/bus.jpg")?;
let model = detect::Model::from_file("yolo26s.pt")?;
let detections = model.predict(&image, &FilterOption::default())?;
```

## Full API

### Common Inputs and Configuration

| API | Description |
| --- | --- |
| `Image::new(width, height, bytes)` | Constructs the unified image input. Supports `Rgb8` and `Rgba8`. |
| `Scale::{N,S,M,L,X}` | Model size, matching `n/s/m/l/x` in weight file names. |
| `device::auto()` | Tries CUDA/Metal according to enabled features, then falls back to CPU. |
| `DType` | Candle dtype, commonly `F32` and `F16`. `with_dtype` can be omitted: default `DtypeRequest::Auto` resolves from **device + target arch + weight dtype**. GPU follows the weight dtype (F16 weights -> F16); CPU / wasm32 forces F32. |
| `FilterOption` | `confidence_threshold` and `class_filter` for detection-style tasks. |
| `MaskOption` | Whether segmentation masks are returned at source-image resolution. |
| `ImageSize` | Input-size descriptor for non-square inputs and snapping to multiples of 32. |

Common builder methods:

```rust
let config = detect::config_builder()
    .with_scale(Scale::S)
    .with_device(device::auto())
    .with_input_size(640)
    .with_image_size(640, 384)
    .with_max_predictions(300)
    .with_rectangular_padding(true)
    .with_labels_count(80)
    // with_dtype can be omitted: Auto follows the weight dtype on GPU and forces F32 on CPU.
    .build();
```

### detect

| API | Description |
| --- | --- |
| `detect::Config` | Type alias for `model::config::Base`. |
| `detect::config_builder()` | Defaults to COCO class count and input size 640. |
| `detect::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a detection model. |
| `model.forward_tensor(&input)` | Runs raw forward on a preprocessed tensor. |
| `model.predict(&image, &filter)` | Returns `Vec<detect::Prediction>`. |
| `model.predict_sahi(&image, &filter, &options)` | SAHI-style sliced detection. |
| `detect::sahi::Options` | Slice width/height, overlap ratios, and whether to include the full image. |

`detect::Prediction` fields: `bbox: BBox`, `confidence: f32`, `class_id: u32`.

### segment

| API | Description |
| --- | --- |
| `segment::Config` | Type alias for `model::config::Base`. |
| `segment::config_builder()` | Defaults to COCO class count and input size 640. |
| `segment::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads an instance segmentation model. |
| `model.forward_tensor(&input)` | Returns raw prediction tensors. |
| `model.predict(&image, &filter, &mask)` | Returns `Vec<segment::Prediction>`. |

`segment::Prediction` fields: `detection: detect::Prediction`, `mask: segment::Mask`.

### semantic

| API | Description |
| --- | --- |
| `semantic::Config` | Type alias for `model::config::Base`. |
| `semantic::config_builder()` | Defaults to Cityscapes class count. |
| `semantic::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a semantic segmentation model. |
| `model.predict(&image, &mask)` | Returns `semantic::Prediction`. |

`semantic::Prediction` fields: `width`, `height`, `classes`, `logits`. Common methods: `class_id(x, y)`, `class_ids()`, `resize(width, height)`.

### classify

| API | Description |
| --- | --- |
| `classify::Config` | Type alias for `model::config::Base`. |
| `classify::config_builder()` | Defaults to ImageNet class count and input size 224. |
| `classify::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a classification model. |
| `model.predict(&image)` | Returns `Vec<classify::Prediction>` sorted by confidence. |

`classify::Prediction` fields: `class_id: u32`, `confidence: f32`.

### pose

| API | Description |
| --- | --- |
| `pose::Config` | Type alias for `model::config::ForPose`. |
| `pose::config_builder()` | Defaults to 17 keypoints, 3 values per keypoint. |
| `with_keypoints_count(n)` | Sets the number of keypoints. |
| `with_keypoint_dims(n)` | Sets values per keypoint; must include at least x/y. |
| `pose::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a pose model. |
| `model.predict(&image, &filter)` | Returns `Vec<pose::Prediction>`. |

`pose::Prediction` fields: `bbox`, `confidence`, `class_id`, `keypoints: Vec<Keypoint>`.

### obb

| API | Description |
| --- | --- |
| `obb::Config` | Type alias for `model::config::Base`. |
| `obb::config_builder()` | Defaults to DOTA class count. |
| `obb::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads an oriented box model. |
| `model.predict(&image, &filter)` | Returns `Vec<obb::Prediction>`. |

`obb::Prediction` fields: `bbox: obb::BBox`, `confidence`, `class_id`.

### `forward_tensor` Output Layout

`forward_tensor(&input)` is low-level raw forward. Return shapes vary by task (`predict()` already includes postprocessing such as top-k, NMS, and mask decoding; see each task source for details):

| Task | Raw output shape |
| --- | --- |
| detect | `[B, A, nc + 4 + 1]` (class scores + box dist + obj, decoded by the head) |
| segment | detect output + proto `[B, 1, mh, mw]` |
| semantic | `[B, nc, H, W]` dense logits |
| classify | `[B, nc]` |
| pose | `[B, A, nc + 5 + nk*kd]` (includes keypoints) |
| obb | `[B, A, nc + 5 + 1]` (includes angle) |

## Differences from Official Ultralytics

- Official Ultralytics Python uses `YOLO("model.pt").predict(...)`; this crate uses `Model::from_file("model.pt")?.predict(...)`.
- The official API can consume paths, URLs, cameras, and numpy/PIL inputs directly; this crate's high-level API uses the unified `Image` type, and the caller is responsible for image loading.
- Official result objects include visualization, saving, DataFrame helpers, and more; this crate returns lightweight Rust structures suitable for embedded services, WASM, or custom postprocessing.
- Official export formats and hardware backends are handled by the Python toolchain; this crate defaults to local Candle inference, while export and quantization are planned through `export`/`quant` modules or the official CLI.

### WASM Bindings

The `wasm` feature exposes browser-oriented entry points:

| API | Description |
| --- | --- |
| `WasmConfig::new` / `default_config` | Builds wasm runtime config and holds a `DeviceSpec`. |
| `WasmConfig::with_cpu_device` / `with_cuda_device` / `with_metal_device` | Selects a device (CUDA/Metal fall back to CPU on wasm32). |
| `FilterOption::with_classes` | Restricts returned class ids. |
| `start()` | wasm export entry point. |
