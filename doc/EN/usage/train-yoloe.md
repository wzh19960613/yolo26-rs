# YOLOE Training (train-yoloe)

Use the `train` feature to train YOLOE-26 segmentation models on the Rust side. It is based on Candle autograd and has no Python/PyTorch dependency.

Full API reference: [train.md](../train.md). For standard YOLO26 segmentation training, see [train-seg.md](train-seg.md).

## Two "Mode" Concepts

YOLOE training involves two independent "mode" enums; do not confuse them:

| Enum | Purpose | Values |
| --- | --- | --- |
| `PromptMode` | **Prompt alignment method** selected when constructing the network (decides which branch is active in the network structure) | `TextPrompt` / `Visual` / `PromptFree` |
| `train::yoloe::Mode` | Official **trainer recipe** alignment (decides which parameters to freeze and what data is required) | `FineTune` / `LinearProbe` / `FromScratch` / `Visual` / `PromptFree` |

In short: `PromptMode` is network-level (passed when constructing `ModelConfig`), while `Mode` is training-strategy-level (decides freeze range). Their relationship:

| `Mode` (trainer recipe) | Matching `PromptMode` | Description |
| --- | --- | --- |
| `FineTune` | `TextPrompt` | Fine-tune on a pretrained checkpoint; only trains the final classification projection |
| `LinearProbe` | `TextPrompt` | Linear probing; freezes backbone and only trains classification head |
| `FromScratch` | `TextPrompt` | Train all parameters from scratch |
| `Visual` | `Visual` | Train SAVPE visual-prompt module on top of a trained text-prompt model |
| `PromptFree` | `PromptFree` | Train LRPC classification branch (prompt-free) |

## Preparation

- **Feature**: must enable `train`. Text-prompt mode additionally needs `yoloe-text` (CLIP class-name encoding).
- **Data**: Ultralytics YAML (detect/seg) or classification directory tree.
- **Weights**: FineTune/LinearProbe should preferably use pretrained `.pt`; `.safetensors` remains optional. FromScratch can run without weights.

## Minimal Example: One Text-Prompt Fine-Tuning Step

```rust
use candle_core::{DType, Device};
use yolo26_rs::{ImageSize, Scale};
use yolo26_rs::train::{OptimizerConfig, collate_segmentation_samples};
use yolo26_rs::train::dataset::{Dataset, Split, ultralytics};
use yolo26_rs::train::yoloe::{Model, ModelConfig};
use yolo26_rs::yoloe::prompt::text_encoder::ClipTextEncoder;
use yolo26_rs::yoloe::PromptMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Construct a trainable model (text-prompt mode), initialized from pretrained weights.
    let config = ModelConfig::new(Scale::N, Device::Cpu, DType::F32, PromptMode::TextPrompt);
    let model = Model::from_pt_file(config, "yoloe-26n-seg.pt")?;

    // 2. Create a training session after pretrained initialization.
    let mut session = yolo26_rs::train::yoloe::Session::new(
        model,
        OptimizerConfig::Sgd { learning_rate: 1e-4 },
    )?;

    // 3. Load one data batch.
    let dataset = ultralytics::seg::from_file(
        "datasets/coco8-seg/dataset.yaml", Split::Train,
        ImageSize::square(640), ImageSize::square(160),
        DType::F32, Device::Cpu, 100,
    )?;
    let samples = (0..dataset.len().min(4)).map(|i| dataset.sample(i)).collect::<Result<Vec<_>, _>>()?;
    let batch = collate_segmentation_samples(&samples)?;
    // The target returned by collate is Target::Segmentation(SegmentationTargets).
    let yolo26_rs::train::Target::Segmentation(ref seg_target) = batch.target else { unreachable!() };

    // 4. Encode class names with CLIP into a prompt table (reuse the same encoder).
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

    // 5. One training step.
    let report = session.text_batch(&batch.input, seg_target, &prompts)?;
    println!("loss = {:.4}", report.loss);

    // 6. Save.
    session.save_pt("finetuned.pt")?;
    Ok(())
}
```

The three prompt modes differ only in the final training step:

| Mode | Training method | Extra parameter |
| --- | --- | --- |
| text-prompt | `session.text_batch(&input, &target, &prompts)` | CLIP-encoded `EmbeddingTable` |
| visual-prompt | `session.visual_batch(&input, &target, &visuals, classes)` | `BatchVisuals` (box/mask) + class names |
| prompt-free | `session.prompt_free_batch(&input, &target)` | None (uses checkpoint-internal LRPC) |

## Matching Official Python

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")
model.train(data="coco8-seg.yaml", epochs=10, batch=4, imgsz=640, mode="fine-tune")
```

## Freeze Parameters (Freeze recipe)

Official trainers freeze some parameters. Use `train::yoloe::Session::new_with_variable_filter` to customize, or refer to the default freeze ranges for `Mode`:

| `Mode` | Freeze range |
| --- | --- |
| `FineTune` / `LinearProbe` | backbone + neck; only trains final classification projection |
| `FromScratch` | no freeze; trains all parameters |
| `Visual` | only trains SAVPE visual-prompt module |
| `PromptFree` | only trains LRPC classification branch |

## Differences from Official

- Official uses PyTorch Trainer classes (`YOLOEPESegTrainer`, and so on); this crate expresses the corresponding trainer, freeze recipe, and data requirements through `Mode` + `PromptMode` + library APIs.
- Training checkpoints default to `.pt` as the primary output format (`save_pt`, default `pt` feature); `.safetensors` remains optional.
- Text-prompt class-name embeddings are generated by the CLIP encoder under the `yoloe-text` feature (`mobileclip2-rs` dependency); prompt-free does not need them.
- This crate currently does not reproduce official DDP/AMP/callback support or PyTorch scheduler-object restoration; native validation records loss, box mAP, and mask mAP.
