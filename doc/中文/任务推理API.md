# 任务推理 API

本页覆盖六个任务根模块：`detect`、`segment`、`semantic`、`classify`、`pose`、`obb`。每个任务都遵循同一形状：`Model::from_file()` 加载权重，`predict()` 返回强类型结果，`forward_tensor()` 暴露低级 tensor 前向；需要覆盖默认 device、dtype 或输入尺寸时再用 `config_builder()`。

## 快速开始

```rust
use yolo26_rs::{FilterOption, Image, detect};

let image = Image::from_file("examples/bus.jpg")?;
let model = detect::Model::from_file("yolo26s.pt")?;
let detections = model.predict(&image, &FilterOption::default())?;
```

## 完整 API

### 通用输入和配置

| API | 说明 |
| --- | --- |
| `Image::new(width, height, bytes)` | 构造统一图像输入，支持 `Rgb8` 和 `Rgba8`。 |
| `Scale::{N,S,M,L,X}` | 模型尺寸，对应权重文件名中的 `n/s/m/l/x`。 |
| `device::auto()` | 按 feature 尝试 CUDA/Metal，失败回退 CPU。 |
| `DType` | Candle dtype，常用 `F32`、`F16`。`with_dtype` 可省略——默认 `DtypeRequest::Auto` **综合 device + target arch + 权重 dtype** 解析：GPU 上与权重一致（F16 权重 → F16），CPU / wasm32 上强制 F32。 |
| `FilterOption` | 检测类任务的 `confidence_threshold` 和 `class_filter`。 |
| `MaskOption` | 分割任务 mask 是否返回源图分辨率。 |
| `ImageSize` | 非方形输入、对齐到 32 倍数的输入尺寸描述。 |

通用 builder 方法：

```rust
let config = detect::config_builder()
    .with_scale(Scale::S)
    .with_device(device::auto())
    .with_input_size(640)
    .with_image_size(640, 384)
    .with_max_predictions(300)
    .with_rectangular_padding(true)
    .with_labels_count(80)
    // with_dtype 可省略：Auto 在 GPU 上跟随权重 dtype，CPU 上强制 F32。
    .build();
```

### detect

| API | 说明 |
| --- | --- |
| `detect::Config` | `model::config::Base` 类型别名。 |
| `detect::config_builder()` | 默认 COCO 类别数，输入尺寸 640。 |
| `detect::Model::from_file(path)` | 自动识别 `.pt` / `.safetensors` 并加载检测模型。 |
| `model.forward_tensor(&input)` | 对预处理 tensor 做 raw forward。 |
| `model.predict(&image, &filter)` | 返回 `Vec<detect::Prediction>`。 |
| `model.predict_sahi(&image, &filter, &options)` | SAHI-style 切片检测。 |
| `detect::sahi::Options` | 切片宽高、重叠比例、是否包含全图。 |

`detect::Prediction` 字段：`bbox: BBox`、`confidence: f32`、`class_id: u32`。

### segment

| API | 说明 |
| --- | --- |
| `segment::Config` | `model::config::Base` 类型别名。 |
| `segment::config_builder()` | 默认 COCO 类别数，输入尺寸 640。 |
| `segment::Model::from_file(path)` | 自动识别 `.pt` / `.safetensors` 并加载实例分割模型。 |
| `model.forward_tensor(&input)` | 返回 raw prediction tensor。 |
| `model.predict(&image, &filter, &mask)` | 返回 `Vec<segment::Prediction>`。 |

`segment::Prediction` 字段：`detection: detect::Prediction`、`mask: segment::Mask`。

### semantic

| API | 说明 |
| --- | --- |
| `semantic::Config` | `model::config::Base` 类型别名。 |
| `semantic::config_builder()` | 默认 Cityscapes 类别数。 |
| `semantic::Model::from_file(path)` | 自动识别 `.pt` / `.safetensors` 并加载语义分割模型。 |
| `model.predict(&image, &mask)` | 返回 `semantic::Prediction`。 |

`semantic::Prediction` 字段：`width`、`height`、`classes`、`logits`。常用方法：`class_id(x, y)`、`class_ids()`、`resize(width, height)`。

### classify

| API | 说明 |
| --- | --- |
| `classify::Config` | `model::config::Base` 类型别名。 |
| `classify::config_builder()` | 默认 ImageNet 类别数，输入尺寸 224。 |
| `classify::Model::from_file(path)` | 自动识别 `.pt` / `.safetensors` 并加载分类模型。 |
| `model.predict(&image)` | 返回按置信度排序的 `Vec<classify::Prediction>`。 |

`classify::Prediction` 字段：`class_id: u32`、`confidence: f32`。

### pose

| API | 说明 |
| --- | --- |
| `pose::Config` | `model::config::ForPose` 类型别名。 |
| `pose::config_builder()` | 默认 17 个 keypoints，每点 3 个值。 |
| `with_keypoints_count(n)` | 设置关键点数量。 |
| `with_keypoint_dims(n)` | 设置每个关键点维度，至少包含 x/y。 |
| `pose::Model::from_file(path)` | 自动识别 `.pt` / `.safetensors` 并加载姿态模型。 |
| `model.predict(&image, &filter)` | 返回 `Vec<pose::Prediction>`。 |

`pose::Prediction` 字段：`bbox`、`confidence`、`class_id`、`keypoints: Vec<Keypoint>`。

### obb

| API | 说明 |
| --- | --- |
| `obb::Config` | `model::config::Base` 类型别名。 |
| `obb::config_builder()` | 默认 DOTA 类别数。 |
| `obb::Model::from_file(path)` | 自动识别 `.pt` / `.safetensors` 并加载旋转框模型。 |
| `model.predict(&image, &filter)` | 返回 `Vec<obb::Prediction>`。 |

`obb::Prediction` 字段：`bbox: obb::BBox`、`confidence`、`class_id`。

### `forward_tensor` 输出布局

`forward_tensor(&input)` 是低级 raw forward，返回形状随任务不同（`predict()` 已含 top-k/NMS/mask decode 等后处理，详见各任务源码）：

| 任务 | raw output 形状 |
| --- | --- |
| detect | `[B, A, nc + 4 + 1]`（类分数 + box dist + obj，经 head 解码） |
| segment | detect 输出 + proto `[B, 1, mh, mw]` |
| semantic | `[B, nc, H, W]` dense logits |
| classify | `[B, nc]` |
| pose | `[B, A, nc + 5 + nk*kd]`（含 keypoint） |
| obb | `[B, A, nc + 5 + 1]`（含角度） |

## 与官方的差异

- 官方 Ultralytics Python API 使用 `YOLO("model.pt").predict(...)`；本 crate 使用 `Model::from_file("model.pt")?.predict(...)`。
- 官方可以直接处理路径、URL、摄像头和 numpy/PIL 输入；本 crate 的高级 API 使用统一 `Image`，调用方负责读取图像。
- 官方结果对象包含可视化、保存、DataFrame 等工具；本 crate 返回轻量 Rust 结构，适合嵌入服务、WASM 或自定义后处理。
- 官方导出格式和硬件后端由 Python 工具链处理；本 crate 默认只做 Candle 本地推理，导出和量化通过 `export`/`quant` 模块建计划或调用官方 CLI。

### WASM 绑定

`wasm` feature 暴露面向浏览器的入口：

| API | 说明 |
| --- | --- |
| `WasmConfig::new` / `default_config` | 构造 wasm 运行配置，持有 `DeviceSpec`。 |
| `WasmConfig::with_cpu_device` / `with_cuda_device` / `with_metal_device` | 指定设备（wasm32 下 CUDA/Metal 回退 CPU）。 |
| `FilterOption::with_classes` | 限制返回的类别 id。 |
| `start()` | wasm 导出入口。 |
