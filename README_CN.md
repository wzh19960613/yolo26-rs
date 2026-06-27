# yolo26-rs

[English](README.md) | 中文

`yolo26-rs` 是一个基于 [Candle](https://github.com/huggingface/candle) 的纯 Rust YOLO26 / YOLOE-26 运行时。默认路径专注推理；启用 `train` feature 后提供原生 Candle 训练 API/CLI。

支持的任务：

| 任务 | 模块 | 权重文件示例 | 输出类型 |
| --- | --- | --- | --- |
| 目标检测 | `detect` | `yolo26s.pt` | `Vec<detect::Prediction>` |
| 图像分类 | `classify` | `yolo26s-cls.pt` | `Vec<classify::Prediction>` |
| 实例分割 | `segment` | `yolo26s-seg.pt` | `Vec<segment::Prediction>` |
| 姿态估计 | `pose` | `yolo26s-pose.pt` | `Vec<pose::Prediction>` |
| 语义分割 | `semantic` | `yolo26s-sem.pt` | `semantic::Prediction` |
| 旋转框检测 | `obb` | `yolo26s-obb.pt` | `Vec<obb::Prediction>` |
| YOLOE 文本提示 | `yoloe::segment` | `yoloe-26s-seg.pt` | `Vec<segment::Prediction>` |
| YOLOE 视觉提示 | `yoloe::segment` | `yoloe-26s-seg.pt` | `Vec<segment::Prediction>` |
| YOLOE 无提示 | `yoloe::segment` | `yoloe-26s-seg-pf.pt` | `Vec<segment::Prediction>` |

模型尺寸使用 `Scale::{N, S, M, L, X}`，对应权重文件名中的 `n/s/m/l/x`。

## 环境准备

优先使用官方 `.pt` 权重。到 [Ultralytics GitHub Releases](https://github.com/ultralytics/assets/releases) 下载对应的 YOLO26 / YOLOE-26 checkpoint，然后直接用 `Model::from_file(...)` 或 `Model::from_pt_file(...)` 加载。

权重文件命名约定：

```text
yolo26{size}.pt                    # 检测
yolo26{size}-{task}.pt             # cls / seg / pose / sem / obb
yoloe-26{size}-seg.pt              # YOLOE text / visual prompt
yoloe-26{size}-seg-pf.pt           # YOLOE prompt-free
```

默认使用 CPU 推理。作为依赖使用时，可通过 Cargo feature 启用硬件后端：

```bash
# CPU（默认）
cargo add yolo26-rs

# macOS Metal
cargo add yolo26-rs --features metal

# CUDA（需 nvcc）
cargo add yolo26-rs --features cuda
```

如果需要 `.safetensors`，也可以使用 [wzh19960613/yolo26-safetensors](https://huggingface.co/wzh19960613/yolo26-safetensors) 中的转换权重。

如果直接在本仓库里写示例或集成测试，可以使用库名 `yolo26_rs`。

## 快速开始：目标检测

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

## 其他任务用法

各任务的加载流程一致：读取权重、构造 config、创建 `Model`、调用 `predict`。

```rust
// 分类：返回按置信度降序排列的类别分数。
let model = classify::Model::from_file("yolo26s-cls.pt")?;
let scores = model.predict(&image)?;
```

```rust
// 实例分割：返回 detection + mask。
let masks = segment_model.predict(
    &image,
    &FilterOption::default(),
    &MaskOption {
        high_resolution: true,
    },
)?;
```

```rust
// 姿态估计：返回 bbox、class_id、confidence 和 keypoints。
let poses = pose_model.predict(&image, &FilterOption::default())?;
```

```rust
// 语义分割：返回每类 logits，可用 class_ids() 取 argmax 类别图。
let semantic = semantic_model.predict(
    &image,
    &MaskOption {
        high_resolution: true,
    },
)?;
let class_map = semantic.class_ids();
```

```rust
// OBB：返回旋转框，bbox 中包含中心点、宽高和角度。
let obbs = obb_model.predict(&image, &FilterOption::default())?;
```

## SAHI 切片检测

检测模块内置了 SAHI-style 切片推理，适合大图或小目标场景：

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

也可以直接调用 `detect::sahi::sliced_predict(&model, &image, &filter, &options)`。

## API 设计概览

公开 API 按任务分层，每个任务模块都遵循相同结构：

```text
detect / classify / segment / pose / semantic / obb
├── Config           # 该任务的模型配置类型
├── config_builder() # 带任务默认值的配置构造器
├── Model            # 已加载权重的推理模型
└── Prediction       # 该任务的强类型预测结果
```

核心设计点：

- `Image` 是统一输入类型，保存宽高和 RGB 像素字节（3 通道）。
- `config_builder()` 提供任务默认值，例如检测/分割默认 COCO 类别数，分类默认 ImageNet 类别数，OBB 默认 DOTA 类别数，语义分割默认 Cityscapes 类别数。
- `Model::from_file(path)` 自动识别 `.pt` / `.safetensors`，并从 checkpoint shape 推断 scale 和类别数；`Model::from_file_with(path, config)` 可覆盖 device、dtype、输入尺寸等配置。
- `predict(...)` 是高级接口，内部完成预处理、前向推理和后处理，返回源图坐标系下的结果。
- `forward_tensor(...)` 是低级接口，适合已有自定义预处理或需要接入自定义后处理的场景。
- `FilterOption` 用于检测类任务的置信度过滤和类别过滤；`MaskOption` 控制分割结果是否返回源图分辨率。
- `default_labels`（需 `default_labels` feature，默认关）提供 `COCO`、`IMAGENET`、`CITYSCAPES`、`DOTA` 类名表；预测结果本身只保存数值 `class_id`。用 `.pt` 权重时也可直接 `pt_loader::load_pt_metadata().names` 读 checkpoint 自带类名。

## 常用配置项

```rust
let config = detect::config_builder()
    .with_scale(Scale::S)              // 模型尺寸；默认加载会从 checkpoint 推断
    .with_device(device::auto())       // CPU / Metal / CUDA
    .with_input_size(640)              // 方形输入，自动对齐到 32 的倍数
    .with_max_predictions(300)         // 保留的最大预测数
    .with_labels_count(80)             // 自定义模型时使用；默认加载会从 checkpoint 推断
    .build();
```

默认 dtype 是 `Auto`：GPU 上跟随权重 dtype，CPU / wasm32 上使用 F32。需要强制精度时再调用 `with_dtype(...)`。

`with_image_size(width, height)` 和 `with_input_shape(ImageSize)` 可用于非方形输入。尺寸会通过 `ImageSize::snapped()` 自动向上对齐到 32 的倍数。

## 特性开关

默认启用 `detect`、`image`、`pt`（检测 + 读图 + 加载 `.pt`），其余按需：

| Feature | 作用 |
| --- | --- |
| `detect`（默认） | 检测推理 + SAHI |
| `image`（默认） | `Image::from_file` 读图 |
| `pt`（默认） | 加载官方 `.pt`；`save_pt` / `Model::save_pt` / `Seg::save_pt` 写回官方可读 `.pt` |
| `classify` | 分类推理 |
| `segment` | 实例分割推理 |
| `semantic` | 语义分割推理 |
| `pose` | 姿态推理 |
| `obb` | OBB 推理 |
| `yoloe` | YOLOE 开放词表推理（聚合 = `yoloe-text` + `yoloe-visual` + `yoloe-pf`） |
| `yoloe-text` | text-prompt 推理路径，依赖外部 `mobileclip2-b-rs` |
| `yoloe-visual` | visual-prompt 推理路径（SAVPE 编码 + visuals） |
| `yoloe-pf` | prompt-free LRPC 路径与内置 4585 词表 `LRPC_VOCAB` |
| `default_labels` | 内置标准数据集类名表（`COCO`/`DOTA`/`CITYSCAPES`/`IMAGENET`）；默认关，开启才编入二进制 |
| `train` | 训练 API（自动拉入全部任务 + YOLOE + `image`） |
| `cuda` | 启用 Candle CUDA 后端 |
| `metal` | 启用 Candle Metal 后端 |
| `wasm` | 启用 wasm-bindgen 绑定，面向浏览器检测推理 |

`device::auto()` 会根据编译 feature 优先尝试 CUDA 或 Metal，失败时回退到 CPU。

## YOLOE

官方参考：[Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)。

本 crate 的 `yoloe` 模块支持 YOLOE-26 的文本提示、视觉提示和无提示（prompt-free）三种推理路径。完整 API 见 [doc/中文/YOLOE接口.md](doc/中文/YOLOE接口.md)。

文本提示：

需要准备的资源：

- YOLOE segment checkpoint: [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt)。
- MobileCLIP2-B text encoder 权重: [mobileclip2_b.pt](https://huggingface.co/apple/MobileCLIP2-B/resolve/main/mobileclip2_b.pt)。
- CLIP BPE tokenizer: [tokenizer.json](https://huggingface.co/openai/clip-vit-base-patch32/raw/main/tokenizer.json)。

text encoder 使用 [wzh19960613/mobileclip2-b-rs](https://github.com/wzh19960613/mobileclip2-b-rs)，通过 `yoloe-text` feature 接入。

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

视觉提示：

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

无提示：

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

## 训练

> **训练尚未经过严格测试，请谨慎使用；当前不保证训练效果、收敛质量或与官方 PyTorch Trainer 的数值一致性。建议先用小数据集和短训练验证完整流程，再投入正式训练。**

训练依赖 Candle autograd 和后端卷积实现；Candle 0.10.2 的 upsample-nearest backward 与 Metal 非连续 convolution kernel 路径可能影响梯度正确性。需要训练，尤其是 Metal 训练时，请先按 [Candle 0.10.2 Patch Guide](doc/中文/Candle-0.10.2补丁指南.md) 使用本地 patched `candle-core`，或等待包含修复的上游版本。

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
