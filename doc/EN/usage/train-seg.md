# Instance Segmentation Training (train-seg)

Use the `train` feature to train a standard YOLO26 instance segmentation model (non-YOLOE) on the Rust side. It is based on Candle autograd and has no Python/PyTorch dependency.

Full API reference: [train.md](../train.md). For YOLOE training, see [train-yoloe.md](train-yoloe.md).

## When to Use

- You want to train a YOLO26 segmentation model on a custom dataset (fixed classes, non-open-vocabulary).
- You need a Rust-native training loop without Python/PyTorch.
- You want fine control over optimizer, freeze, and learning-rate schedule through library APIs.

## Preparation

- **Feature**: `--features train` (automatically includes `segment` + `image`).
- **Data**: Ultralytics YAML-format segmentation dataset (for example, `coco8-seg/dataset.yaml`).
- **Weights** (optional): prefer initialization from pretrained `.pt`; `.safetensors` remains optional. Training from scratch (`Model::new`) is also supported.

## Rust Example (One Training Step)

```rust
use candle_core::{DType, Device};
use yolo26_rs::{ImageSize, Scale, segment};
use yolo26_rs::train::{
    ModelConfig, OptimizerConfig, Target,
    dataset::{Dataset, Split, collate::collate_segmentation_samples, ultralytics},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Construct a trainable segmentation model (initialized from pretrained weights).
    let yaml = ultralytics::Yaml::from_file("datasets/coco8-seg/dataset.yaml")?;
    let config = ModelConfig::Segment(
        segment::config_builder()
            .with_scale(Scale::N)
            .with_labels_count(yaml.names.len())
            .build(),
    );
    let mut model = yolo26_rs::train::Model::from_pt_file("yolo26n-seg.pt", config)?;
    model.set_class_names(yaml.names.clone())?;

    // 2. Create a training session (binds model and optimizer).
    let mut session = yolo26_rs::train::Session::new(
        model,
        OptimizerConfig::Sgd { learning_rate: 1e-3 },
    )?;

    // 3. Load one data batch.
    let dataset = ultralytics::seg::from_file(
        "datasets/coco8-seg/dataset.yaml", Split::Train,
        ImageSize::square(640), ImageSize::square(160),
        DType::F32, Device::Cpu, 100,
    )?;
    let samples = (0..dataset.len().min(4))
        .map(|i| dataset.sample(i))
        .collect::<Result<Vec<_>, _>>()?;
    let batch = collate_segmentation_samples(&samples)?;

    // 4. One training step (input + target -> loss -> backward -> update).
    let report = session.train_batch(&batch.input, &batch.target)?;
    println!("loss = {:.4}", report.loss);

    // 5. Save.
    session.model().save_pt("trained.pt")?;
    Ok(())
}
```

The `batch.target` returned by `collate_segmentation_samples` is already `Target::Segmentation(SegmentationTargets)` and can be passed directly to `train_batch`.

## Matching Official Python

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
model.train(data="coco8-seg.yaml", epochs=100, batch=4, imgsz=640)
```

## API Quick Reference

### Model Construction

| API | Description |
| --- | --- |
| `train::Model::new(config)` | Constructs from an empty VarMap (training from scratch). |
| `train::Model::new_with_class_names(config, names)` | Constructs from an empty VarMap and writes class names into later `.pt` exports. |
| `train::Model::from_pt_file(path, config)` | Initializes from an official `.pt` checkpoint (requires `pt` feature). |
| `train::Model::from_safetensors_file(path, config)` | Initializes from `.safetensors` pretrained weights. |
| `ModelConfig::Segment(segment::config_builder()...build())` | Segmentation task config. |

### Training Session

| API | Description |
| --- | --- |
| `train::Session::new(model, optimizer)` | Creates a training session bound to model and optimizer. |
| `Session::new_with_variable_filter(model, optimizer, filter)` | Trains only matching variables (implements freeze recipes). |
| `Session::train_batch(&input, &target)` | One training step: `input` (image tensor) + `target` (`Target::Segmentation`) -> loss -> backward -> update. |
| `Session::train_batch_with_loss_config(&input, &target, loss_config)` | Custom detection loss gains. |
| `Session::model()` | Borrows the inner `train::Model`, used for saving weights. |

### Optimizers

| API | Description |
| --- | --- |
| `OptimizerConfig::Sgd { learning_rate }` | SGD (with momentum). |
| `OptimizerConfig::AdamW { params }` | AdamW. |
| `OptimizerConfig::MuSgd { params, learning_rate }` | MuSGD (momentum + weight decay). |

### Data and Target

| API | Description |
| --- | --- |
| `ultralytics::seg::from_file(yaml, split, img_size, mask_size, dtype, device, cache)` | Loads an Ultralytics segmentation dataset. |
| `Dataset::sample(i)` | Returns sample `i` (`Sample`). |
| `collate_segmentation_samples(&samples)` | Merges multiple `Sample`s into one batch `Sample` (`input` + `Target::Segmentation`). |
| `Target::Segmentation(SegmentationTargets)` | Segmentation target (boxes + class ids + masks). |

### Saving

| API | Description |
| --- | --- |
| `Model::save_pt(path)` | Saves an official `.pt` checkpoint (requires `pt` feature), synchronizing current `labels_count`, head tensor shapes, and class names if set. |
| `Model::save_safetensors(path)` | Saves `.safetensors` (loadable directly by inference-side `segment::Model::from_file`). |

## Differences from Official / Notes

- Official uses the PyTorch Trainer + Ultralytics engine; this crate represents trainable networks and training loops through `train::Model` + `train::Session`, with no Python dependency.
- The VarMap dtype constructed for training is explicit (`DType::F32` is natural on CPU) and is unrelated to inference-side `Auto` dtype rules, which only apply when loading existing checkpoints.
- This crate currently does not reproduce official DDP/AMP/callback support or PyTorch scheduler-object restoration; native validation records loss, box mAP, and mask mAP.
- Freezing parameters is implemented by variable-name filtering through `Session::new_with_variable_filter` (matching the official `freeze` parameter).
