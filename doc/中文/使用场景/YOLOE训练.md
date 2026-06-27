# YOLOE 训练（train-yoloe）

用 `train` feature 在 Rust 侧训练 YOLOE-26 分割模型。基于 Candle autograd,无 Python/PyTorch 依赖。

完整 API 参考见 [训练.md](../训练.md)。标准 YOLO26 分割训练见 [实例分割训练.md](实例分割训练.md)。

## 两种"模式"概念

YOLOE 训练涉及两个独立的"模式"枚举,不要混淆:

| 枚举 | 用途 | 取值 |
| --- | --- | --- |
| `PromptMode` | 构造网络时选定的 **prompt 对齐方式**(决定网络结构里哪个分支激活) | `TextPrompt` / `Visual` / `PromptFree` |
| `train::yoloe::Mode` | 对齐官方 **trainer recipe**(决定冻结哪些参数、需要什么数据) | `FineTune` / `LinearProbe` / `FromScratch` / `Visual` / `PromptFree` |

简单说:`PromptMode` 是网络层面的(构造 `ModelConfig` 时传),`Mode` 是训练策略层面的(决定 freeze 范围)。它们的对应关系:

| `Mode`(trainer recipe) | 对应 `PromptMode` | 说明 |
| --- | --- | --- |
| `FineTune` | `TextPrompt` | 在预训练 checkpoint 上微调,只训练最终分类投影 |
| `LinearProbe` | `TextPrompt` | 线性探测,冻结 backbone,只训练分类头 |
| `FromScratch` | `TextPrompt` | 从头训练全部参数 |
| `Visual` | `Visual` | 在已训练的 text-prompt 模型上训练 SAVPE visual-prompt 模块 |
| `PromptFree` | `PromptFree` | 训练 LRPC 分类分支(prompt-free) |

## 准备

- **Feature**:必须启用 `train`。text-prompt 模式额外需要 `yoloe-text`(CLIP 编码类名)。
- **数据**:Ultralytics YAML(detect/seg)或分类目录树。
- **权重**:FineTune/LinearProbe 优先用预训练 `.pt`;`.safetensors` 仍可选。FromScratch 可不带。

## 最小示例:text-prompt 微调一步

```rust
use candle_core::{DType, Device};
use yolo26_rs::{ImageSize, Scale};
use yolo26_rs::train::{OptimizerConfig, collate_segmentation_samples};
use yolo26_rs::train::dataset::{Dataset, Split, ultralytics};
use yolo26_rs::train::yoloe::{Model, ModelConfig};
use yolo26_rs::yoloe::prompt::text_encoder::ClipTextEncoder;
use yolo26_rs::yoloe::PromptMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 构造可训练模型(text-prompt 模式),从预训练权重初始化
    let config = ModelConfig::new(Scale::N, Device::Cpu, DType::F32, PromptMode::TextPrompt);
    let model = Model::from_pt_file(config, "yoloe-26n-seg.pt")?;

    // 2. 用预训练权重初始化后创建训练 session
    let mut session = yolo26_rs::train::yoloe::Session::new(
        model,
        OptimizerConfig::Sgd { learning_rate: 1e-4 },
    )?;

    // 3. 加载一个 batch 的数据
    let dataset = ultralytics::seg::from_file(
        "datasets/coco8-seg/dataset.yaml", Split::Train,
        ImageSize::square(640), ImageSize::square(160),
        DType::F32, Device::Cpu, 100,
    )?;
    let samples = (0..dataset.len().min(4)).map(|i| dataset.sample(i)).collect::<Result<Vec<_>, _>>()?;
    let batch = collate_segmentation_samples(&samples)?;
    // collate 返回的 target 是 Target::Segmentation(SegmentationTargets)
    let yolo26_rs::train::Target::Segmentation(ref seg_target) = batch.target else { unreachable!() };

    // 4. CLIP 编码类名成 prompt table(复用同一个 encoder)
    let encoder = ClipTextEncoder::from_files(
        "mobileclip2_b.pt",
        "tokenizer.json",
    )?;
    let class_names = ["person", "bicycle", "car"];
    let embeddings = encoder.embed_texts(class_names)?;
    let prompts = yolo26_rs::yoloe::usage::EmbeddingTable::new(
        embeddings,
        class_names.iter().map(|s| s.to_string()).collect(),
    )?;

    // 5. 一步训练
    let report = session.text_batch(&batch.input, seg_target, &prompts)?;
    println!("loss = {:.4}", report.loss);

    // 6. 保存
    session.save_pt("finetuned.pt")?;
    Ok(())
}
```

三种 prompt 模式的训练 step 区别只在最后一步:

| 模式 | 训练方法 | 额外参数 |
| --- | --- | --- |
| text-prompt | `session.text_batch(&input, &target, &prompts)` | CLIP 编码的 `EmbeddingTable` |
| visual-prompt | `session.visual_batch(&input, &target, &visuals, classes)` | `BatchVisuals`(box/mask) + 类名 |
| prompt-free | `session.prompt_free_batch(&input, &target)` | 无(用 checkpoint 内部 LRPC) |

## 对应官方 Python

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")
model.train(data="coco8-seg.yaml", epochs=10, batch=4, imgsz=640, mode="fine-tune")
```

## 冻结参数(Freeze recipe)

官方 trainer 会冻结部分参数。用 `train::yoloe::Session::new_with_variable_filter` 自定义,或参考 `Mode` 的默认 freeze 范围:

| `Mode` | 冻结范围 |
| --- | --- |
| `FineTune` / `LinearProbe` | backbone + neck,只训练最终分类投影 |
| `FromScratch` | 不冻结,训练全部参数 |
| `Visual` | 只训练 SAVPE visual-prompt 模块 |
| `PromptFree` | 只训练 LRPC 分类分支 |

## 与官方的差异

- 官方用 PyTorch Trainer 类(`YOLOEPESegTrainer` 等);本 crate 通过 `Mode` + `PromptMode` + 库 API 表达对应 trainer、freeze recipe 和数据需求。
- 训练 checkpoint 默认以 `.pt` 为主要输出格式(`save_pt`,`pt` feature 默认);`.safetensors` 仍可选。
- text-prompt 的类名 embedding 用 `yoloe-text` feature 下的 CLIP 编码器(`mobileclip2-rs` 依赖)生成,prompt-free 不需要。
- 当前不复刻官方 DDP/AMP/callback 和 PyTorch scheduler 对象恢复;native validation 已记录 loss、box mAP、mask mAP。
