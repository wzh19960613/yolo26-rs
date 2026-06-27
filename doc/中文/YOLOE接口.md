# YOLOE

官方参考：[Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)。

YOLOE 模块覆盖 YOLOE-26 的 text prompt、visual prompt、prompt-free vocabulary、RepRTA、SAVPE、LRPC 和 open-vocabulary detect/segment head。实现参考 Ultralytics YOLOE/YOLO26 文档和 `ultralytics/models/yolo/yoloe/*`、`ultralytics/data/augment.py`、`ultralytics/nn/modules/head.py`；非 YOLO26 家族的 YOLOE 权重和示例只作为背景，不作为本 crate 的功能范围。

本 crate 的 YOLOE 权重以官方 `.pt` 为主要格式（默认 `pt` feature）：`Segment::from_pt_file(...)` / `Detect::from_pt_file(...)` 直接加载官方 `yoloe-26*-seg.pt` / `-seg-pf.pt`（也可用零参 `from_file(path)` 自动推断），训练后 `Seg::save_pt` 写回官方 PyTorch 可读 `.pt`（已 `torch.load` 验证）。`.safetensors` 转换权重同样可用：`from_file(path)` / `from_bytes(bytes)` / `from_safetensors(weights, config)` 都会从 checkpoint shape 推断 scale 和 head layout。`Checkpoint::parse()` 和 `Session::from_checkpoint()` 解析文件名里的 scale、segmentation、prompt-free 语义，`.pt` 与 `.safetensors` 后缀均接受。

> 命名：`Detect` / `Segment` 对齐六个任务根的 `Model` 形状。`Session` 是 YOLOE 的不可变 prompt 状态。

## 快速开始

### Text prompt

`Session::text(&encoder, model.reprta(), classes)` 用 `mobileclip2-rs` 依赖的 MobileCLIP2-b CLIP 文本编码器把类名编码成 `[classes, 512]` L2-normalized embedding（需 `yoloe-text` feature）。`encoder` 是一次构造、多次借用的 `ClipTextEncoder`；`model.reprta()` 返回 `Model` 在加载时从 `model.23.reprta` 持有的 `Option<&RepRta>`；`classes` 接受任何 `AsRef<str>`（`&str`、`String`、`&&str` 等，无需逐个 `.into()`）：

text encoder 使用 [wzh19960613/mobileclip2-b-rs](https://github.com/wzh19960613/mobileclip2-b-rs)。推荐准备这些官方资源：

- YOLOE segment checkpoint: [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt)。
- MobileCLIP2-B text encoder 权重: [mobileclip2_b.pt](https://huggingface.co/apple/MobileCLIP2-B/resolve/main/mobileclip2_b.pt)。
- CLIP BPE tokenizer: [tokenizer.json](https://huggingface.co/openai/clip-vit-base-patch32/raw/main/tokenizer.json)。

```rust
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::Session;
use yolo26_rs::yoloe::prompt::text_encoder::ClipTextEncoder;

let model = Model::from_file("yoloe-26s-seg.pt")?; // 加载时一并构造 RepRTA

// CLIP encoder 构造一次，多次复用
let encoder = ClipTextEncoder::from_files(
    "mobileclip2_b.pt",
    "tokenizer.json",
)?;

// 构造不可变 text prompt session（跨图可复用）
let session = Session::text(&encoder, model.reprta(), ["person", "bus"])?;
```

已有外部 embedding 时，用 `Session::text_with_embeddings` 直接注入：

```rust
use candle_core::{Device, Tensor};
use yolo26_rs::yoloe::{EmbeddingTable, Session};

let embeddings = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu)?;
let table = EmbeddingTable::new(embeddings, vec!["person".into(), "bus".into()])?;
let session = Session::text_with_embeddings(table)?;
```

### Visual prompt

```rust
use candle_core::Device;
use yolo26_rs::{
    ImageSize, Scale,
    yoloe::{Config, Session, VisualPrompt},
};

// visual prompt 是 per-image 的：每图新建 session；SAVPE 在前向时算
let session = Session::visual(vec![
    VisualPrompt::from_box(0, [10.0, 20.0, 90.0, 160.0])?,
])?;

// 底层 helper（对齐官方 visuals tensor 构造）
let visuals = yolo26_rs::yoloe::Visuals::from_boxes(
    &[VisualPrompt::from_box(0, [10.0, 20.0, 90.0, 160.0])?],
    ImageSize::new(image_width, image_height),
    0.25,
    &Device::Cpu,
)?;
```

官方文档建议 image-specific prompts 逐图运行；官方源码中存在 bbox-only batch 分支，会把每张图的 bbox visual prompts pad 成 `visuals`。本 crate 保留一个低级 helper 以对齐该 YOLOE-26 源码路径，但它不是优先推荐的高层用法：

```rust
use candle_core::Device;
use yolo26_rs::{
    ImageSize,
    yoloe::{BatchVisuals, VisualPrompt, VisualBatchItem},
};

let batch = BatchVisuals::from_boxes(
    &[
        VisualBatchItem::new(
            ImageSize::new(1280, 720),
            vec![VisualPrompt::from_box(0, [100.0, 80.0, 320.0, 360.0])?],
        )?,
        VisualBatchItem::new(
            ImageSize::new(640, 640),
            vec![
                VisualPrompt::from_box(1, [20.0, 20.0, 200.0, 260.0])?,
                VisualPrompt::from_box(2, [260.0, 80.0, 520.0, 520.0])?,
            ],
        )?,
    ],
    ImageSize::square(640),
    0.125,
    &Device::Cpu,
)?;
// batch.tensor 为 [batch, max_prompts, h, w]；batch.class_ids[b][p] 是原始 class id
```

### Prompt-free

```rust
use yolo26_rs::yoloe::Session;

let session = Session::prompt_free(vec!["person".into(), "bus".into()])?;
```

## 完整 API

### 配置和状态

| API | 说明 |
| --- | --- |
| `yoloe::config_builder()` | 返回默认 YOLOE `Config` 的 builder（对齐任务根 `config_builder()`）。 |
| `yoloe::Config::default()` | text/visual prompt YOLOE 配置（含 device/dtype/image_size/max_predictions，对齐任务根 `Base`）。 |
| `Config::segmentation(scale)` | segmentation-first YOLOE 配置。 |
| `Config::prompt_free(scale)` | prompt-free + LRPC 配置。 |
| `Checkpoint::parse(name)` | 解析 YOLOE checkpoint 名称，如 `yoloe-26s-seg-pf.pt`；`.pt` 与 `.safetensors` 后缀均接受。 |
| `Usage` | `TextPrompt`/`VisualPrompt`/`PromptFree`（推理 prompt 来源）；`FineTune`/`LinearProbe`/`Validate` 分别对应微调、线性探针、验证场景。 |
| `State` | `Empty`、`Text`、`Visual`、`PromptFree`。 |
| `Controller` | 管理 prompt 状态。 |
| `Session` | 不可变 prompt 状态，构造时一次确定（text/prompt_free 可跨图复用，visual 每图新建）。含 prompt、预测配置、scorer 和 prompt table。 |

### Session 构造（推荐入口）

| 构造函数 | 说明 |
| --- | --- |
| `Session::text(&encoder, model.reprta(), classes)` | 用可重用的 `ClipTextEncoder` 从类名生成 `EmbeddingTable`，并借用 `Model` 持有的 RepRTA 对齐（需 `yoloe-text` feature；`classes` 接受 `AsRef<str>` 迭代器）。 |
| `Session::text_with_embeddings(table)` | 激活外部提供的（如官方 CLIP）embedding。 |
| `Session::text_with_reprta(reprta, table, config)` | 手动指定 RepRTA + 外部 embedding 激活（高级用法；`Session::text` 已自动加载 RepRTA）。 |
| `Session::prompt_free(classes)` | 激活 prompt-free 词汇表。 |
| `Session::prompt_free_default()` | 用内置 `default_labels::LRPC_VOCAB`（4585 名）激活 prompt-free 词汇表（需 `yoloe-pf` feature，包含在 `yoloe` 聚合里；预测 class id 直接索引该表得可读名）。 |
| `Session::prompt_free_with_embeddings(table)` | 用静态 LRPC embedding 激活。 |
| `Session::visual(prompts)` | per-image visual prompt session（box/mask 由 `predict_visual_prompts` 的 `VisualSource` 判别，SAVPE 在前向时算）。 |
| `Session::new(config)` / `from_checkpoint(name)` | 低层构造，配合下面的 `set_*` 方法分步设置。 |

> 低层 `set_*` 方法（`set_classes_with_clip_embeddings`、`set_text_prompt_embeddings`、`set_visual_prompts`、`set_visual_prompt_embeddings`、`set_prompt_free_vocabulary` 等）仍保留，供高级用户分步构造或注入预计算 embedding。

### Text prompt（低层 set_* 方法）

| API | 说明 |
| --- | --- |
| `set_classes(classes)` | 仅记录类别和 embedding space，不生成 embedding；此后必须通过 `set_classes_with_clip_embeddings` / `set_text_prompt_embeddings` / `set_text_prompt_embeddings_with_reprta` 之一激活，否则 text-prompt scoring 返回错误。 |
| `set_classes_with_clip_embeddings(&encoder, reprta, classes)` | 用可重用的 `ClipTextEncoder` 从类名生成 `EmbeddingTable`，并借用 `Option<&RepRta>` 对齐（需 `yoloe-text` feature）。 |
| `EmbeddingTable::new(embeddings, classes)` | 创建 `[classes, dim]` prompt embedding 表。 |
| `set_text_prompt_embeddings(table)` | 激活 text prompt embeddings。 |
| `RepRta::load(vb)` | 从官方 `reprta` 权重加载 RepRTA。 |
| `RepRta::load_optional(vb)` | `vb` 含 `m.w12.weight` 时加载 RepRTA，否则返回 `None`（`Model` 加载时用）。 |
| `Model::reprta()` | 返回 `Model` 在加载时持有的 `Option<&RepRta>`，借给 `Session::text`。 |
| `set_text_prompt_embeddings_with_reprta(reprta, table)` | 先过 RepRTA 再激活 embedding（同时把 reprta 存入 session）。 |
| `set_reprta(reprta)` | 装载 RepRTA；此后 `set_text_prompt_embeddings` 在 `config.rep_rta.enabled` 时自动应用 RepRTA，匹配官方默认推理路径。 |
| `score_region_features(features)` | 对 `[regions, dim]` 或 batched region features 打分。 |
| `score_feature_map(feature_map)` | 对 dense feature map 打分。 |

### Visual prompt

| API | 说明 |
| --- | --- |
| `VisualPrompt::from_box(class_id, xyxy)` | box prompt 元数据。 |
| `VisualPrompt::from_mask(class_id, xyxy)` | mask prompt 元数据。 |
| `VisualKind::{Box, Mask}` | prompt 来源类型。 |
| `Visuals::from_boxes(prompts, source_size, scale_factor, device)` | 从 box prompts 构造单图官方-style `visuals` tensor（`[1, classes, h, w]`，同类已合并）。 |
| `Visuals::from_masks(prompts, masks, scale_factor)` | 从 segmentation masks 构造单图官方-style `visuals` tensor。 |
| `VisualBatchItem::new(source_size, prompts)` | 声明一张源图及其 bbox visual prompts；用于对齐官方源码中的 bbox-only batch 分支。 |
| `BatchVisuals::from_boxes(items, target_size, scale_factor, device)` | 对齐官方源码中的 bbox-only batch 分支，letterbox bbox 后输出 padded `[batch, max_prompts, h, w]` 张量 + 每张图 prompt 维对应的 `class_ids`。 |
| `BatchVisuals::from_tensor(tensor, class_ids)` | 直接包装调用者自造的 batch 张量（训练路径常用）。 |
| `VisualSource::Boxes` / `VisualSource::Masks(&tensor)` | `predict_visual_prompts` 的来源判别：box 光栅化或吃 source mask。 |
| `set_visual_prompts(prompts)` | 激活 visual prompt 状态。 |
| `set_visual_prompt_embeddings(prompts, table)` | 同时设置 visual prompts 和预计算 SAVPE embeddings。 |
| `encode_visuals(embedding_map, &visuals)` | fallback pooler，从 embedding map 聚合 visual embeddings。 |
| `Encoder::load(...)` | 加载官方 SAVPE。 |
| `encode_visuals_with_savpe(encoder, features, &visuals)` | 使用官方 SAVPE 从三层特征图编码 prompt。 |

visual prompt mask 规则：

- `Visuals.tensor` 形状为 `[1, classes, height, width]`，同类已按官方 `LoadVisualPrompt.get_visuals()` 语义合并并按 class id 排序。
- `Visuals::from_boxes()` / `from_masks()` 会把每条 prompt mask 合并、排序为 `[1, classes, mask_h, mask_w]`，可直接传给 `encode_visuals()` 或 `encode_visuals_with_savpe()`。
- session 的 `encode_visuals*` 会对传入 `Visuals` 再按当前 prompt 状态合并一次；`Visuals` 已经是按 class 合并过的官方 `visuals` 时会原样透传。
- `BatchVisuals::from_boxes()` 只处理 bbox prompts；这对应官方源码中的 `len(img) > 1` bbox-only 分支。官方文档对 image-specific prompts 的建议仍是逐图运行。
- batch helper 会对每张图执行官方-style letterbox bbox 映射、同类合并和 prompt 维 padding，适合直接传给 `Pooler::encode()` 或 `Encoder::forward()` 这类 batch tensor 入口。
- `BatchVisuals::from_boxes()` 返回的 `class_ids[b][p]` 表示 batch 第 `b` 张图未 padding 前第 `p` 个 prompt/class 对应的原始 class id，便于把预测结果映射回用户传入的类别。

### Prompt-free / LRPC

| API | 说明 |
| --- | --- |
| `set_prompt_free_vocabulary(classes)` | 激活 prompt-free class vocabulary。 |
| `set_prompt_free_embeddings(table)` | 设置 prompt-free vocabulary embeddings。 |
| `LrpcHead` | 轻量 proposal filtering + prompt scoring。 |
| `Official` | 加载官方单尺度 LRPC head。 |
| `Pyramid` | 三尺度官方 prompt-free pyramid head。 |
| `forward_official_lrpc_head(...)` | prompt-free detect head 前向。 |
| `forward_official_lrpc_segment_head(...)` | prompt-free segment head 前向。 |

### Open-vocabulary model/head

| API | 说明 |
| --- | --- |
| `detect::head::Head` | YOLOE detect head，接收 prompt embeddings。 |
| `Detect::from_file(path)` / `from_file_with(path, config)` / `from_pt_file(path, config)` / `from_bytes(bytes)` / `from_safetensors(weights, config)` | YOLOE detect-only model 加载入口；`.pt` 为优先路径，scale/head layout 从 checkpoint 推断，device/dtype/image_size/max_predictions 由 `yoloe::Config` 控制。 |
| `segment::head::Head` | YOLOE segment head，输出 predictions + proto masks。 |
| `Segment::from_file(path)` / `from_file_with(path, config)` / `from_pt_file(path, config)` / `from_bytes(bytes)` / `from_safetensors(weights, config)` | YOLOE segment model 加载入口；`.pt` 为优先路径。 |
| `forward_tensor(input, session)` | 根据 session 状态走 text/visual prompt 或 prompt-free raw forward；detect 返回 `[batch, det, 6]`，segment 返回 `([batch, det, 6 + mask_dim], proto)`。 |
| `forward_prompt_free_tensor(input, session)` | 显式 prompt-free LRPC raw forward。 |
| `predict(image, session, filter)` | detect-only typed 预测入口，返回 `Vec<detect::Prediction>`。 |
| `predict(image, session, filter, mask)` | segment typed 预测入口，返回 `Vec<segment::Prediction>`。 |
| `predict_visual_prompts(image, prompts, source, session, filter[, mask])` | visual-prompt typed 预测入口（detect 和 segment 都支持）；`source` 用 `VisualSource::Boxes` 走 box 光栅化，`VisualSource::Masks(&tensor)` 吃原图坐标 `[prompts, H, W]`/`[1, prompts, H, W]` 的 mask。内部完成 letterbox、SAVPE 编码和后处理。 |
| `encode_visual_prompts(reference_image, prompts, source)` | 跨图两步 API：在参考图上 SAVPE 编码出可复用 `EmbeddingTable`（官方 `vpe`），喂给 `Session::text_with_embeddings` 后用普通 `predict` 识别任意图。detect/segment 都支持。 |
| `predict_prompt_free(image, session, filter[, mask])` | prompt-free typed 预测入口；detect 和 segment 都支持。 |

open-vocabulary head 的 top-k 后处理支持 `[batch, anchors, channels]`，同一 prompt table 的 text prompt 或 prompt-free batch 推理会按 batch 样本分别排序和保留预测。多图 visual prompt 的 per-image class id 映射仍由 `BatchVisuals::from_boxes()` 返回的 `class_ids` 负责。

### Validate / Train

| API | 说明 |
| --- | --- |
| `validation_request()` | 生成当前 prompt 状态的 validation request。 |
| `train::yoloe::Session` | YOLOE 训练 session。三种训练 step 见下表。 |

**训练 step**(详见 [使用场景/YOLOE训练.md](使用场景/YOLOE训练.md)):

| prompt 模式 | 方法 | prompt 来源 |
| --- | --- | --- |
| text-prompt | `text_batch(&input, &target, &prompts)` | CLIP 编码的 `EmbeddingTable` |
| visual-prompt | `visual_batch(&input, &target, &visuals, classes)` | SAVPE 从 box/mask 生成 |
| prompt-free | `prompt_free_batch(&input, &target)` | checkpoint 内部 LRPC,无参数 |

## 与官方的差异

- 官方 `YOLOE("*.pt").set_classes([...])` 会在 Python 侧用 CLIP 文本编码器处理 text embedding；本 crate 在 `yoloe-text` feature 下通过 `mobileclip2-rs` 依赖调用 MobileCLIP2-b CLIP 文本编码器（`ClipTextEncoder`，可重用值类型），`Session::text(&encoder, model.reprta(), classes)`（底层 `set_classes_with_clip_embeddings`）把类名编码成 `[classes, 512]` L2-normalized embedding 并借用 `Model` 加载时持有的 RepRTA 对齐，与官方 CLIP → RepRTA → score 路径一致。CLIP encoder 由调用方构造，不在 YOLOE checkpoint 内。
- 官方 visual prompt 可以在 `predict(..., visual_prompts=...)` 直接传入图像、boxes、masks；本 crate 的 `Segment` / `Detect` 已提供单图 box 和 mask visual-prompt typed 预测入口（`predict_visual_prompts`，由 `VisualSource` 判别 box/mask），底层 `Visuals` 类型可直接构造官方 `visuals` tensor。
- 官方文档说明 batch inference 支持普通输入，但 image-specific prompts 应逐图运行；官方源码另有一个 `len(img) > 1`、bbox-only、`pad_sequence` 补齐的内部路径。本 crate 的 `BatchVisuals::from_boxes()` 只对齐这个 YOLOE-26 源码路径，不额外扩展多图 mask prompts。
- 官方 `LoadVisualPrompt` 会合并同类 prompt masks；本 crate 已在 `Visuals` 构造与 `Session` 的 mask 编码入口对齐这一点。
- 官方 YOLOE training 由 PyTorch trainer 类实现;本 crate 的 `train::yoloe::Session` 提供原生 batch step(text/visual/prompt-free 三种),用 `Mode` + `PromptMode` 表达对应 trainer 和 freeze recipe。当前不复刻官方 DDP/AMP/callback;YOLOE validation 已记录 native loss、box mAP、mask mAP。详见 [使用场景/YOLOE训练.md](使用场景/YOLOE训练.md)。
- 官方对 RepRTA-enabled checkpoint 默认自动启用 RepRTA；本 crate 的 `Model` 在 `from_file`/`from_pt_file`/`from_safetensors` 时用 `RepRta::load_optional` 从 `model.23.reprta` 自动构造并持有 RepRTA（RepRTA 仅文本路径需要，visual/prompt-free 不需加载），`Session::text` 通过 `model.reprta()` 借用，无需手动加载。
- 官方 visual prompt 返回的类是原始 category id；本 crate 的 visual prompt 类名为 `visual_class_{原始id}` 占位符（`EmbeddingTable.classes` 要求非空字符串），原始 id 由 `BatchVisuals::from_boxes()` 返回的 `class_ids` 提供。
- YOLOE one-to-one head 设计上不需要 NMS（每个 GT 一个 anchor），`postprocess_topk` 仅 argmax + top-k 排序；`PredictConfig.agnostic_nms` 字段当前是 no-op，保留供未来非 end2end 回退路径。
- fallback `Contrastive` 的 `logit_scale=1.0`/`bias=-10.0` 是经验值（无 checkpoint 可读），与官方 `BnContrastive` 的可学习 `logit_scale`/`bias` 数值不可比；使用官方权重走 `BnContrastiveFeatureHead` 路径会正确读取 checkpoint 的 `logit_scale`/`bias`。
- 官方 YOLOE detect（如 `yoloe-26s.pt`）与 segment 都是一等推理路径；本 crate 已提供 `Detect` 和 `Segment` 两个端到端封装。detect-only 返回 typed `detect::Prediction`，segment 返回 typed `segment::Prediction`。官方 `.pt` 是优先权重格式，`.safetensors` 转换权重仍支持。
- **纯检测权重（`yoloe-26s.pt`,或对应 `.safetensors`）不含 SAVPE**：在其上调用 `predict_visual_prompts` 会运行时报错（提示用 `-seg` 权重）。要做 visual prompt 的检测结果，请用 `-seg` 权重加载成 `Detect`（取框丢弃 mask）。
- `Checkpoint::parse()` 接受任意 `yoloe-` 前缀的名称，但仅 YOLOE-26 经过验证；`yoloe-11*` 虽能解析，但不在本 crate 支持范围。
- 官方 SAVPE 是三尺度卷积 + 多头空间注意力模块；本 crate 的 `encode_visuals()` fallback pooler 只做 masked-average-pooling + L2 归一化（单 embedding map 场景），与官方 SAVPE 数值不等价。官方 SAVPE 主路径 `Encoder` 已完整实现，单尺度 fallback 仅作兼容入口。
