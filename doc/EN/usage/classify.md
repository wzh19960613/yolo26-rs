# Image Classification Inference (classify)

This guide shows how to use the `classify` task root for YOLO26 classification inference (ImageNet-style whole-image classification). The difference from detection-style tasks is that the default input size is 224, there is no box, and `predict()` does not accept `FilterOption`; it returns `Vec<Prediction>` sorted by confidence directly.

Full API reference: [tasks.md](../tasks.md).

## When to Use

- Whole-image classification (one label per image), such as product inspection or scene recognition.
- You need lightweight inference results without detection boxes or NMS.
- You want top-k probability distributions on the Rust side for custom postprocessing.

## Preparation

- **Feature**: requires `--features classify` (not included in default features). Add `--features metal`/`--features cuda` for GPU backends.
- **Weights**: official classification `.pt` weights are named like `yolo26s-cls.pt` (note the `-cls` suffix).

```bash
cargo build --release
```

## Rust Example

```rust
use yolo26_rs::{Image, classify};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = classify::Model::from_file("yolo26s-cls.pt")?;
    let predictions = model.predict(&image)?;
    for c in predictions.iter().take(5) {
        println!("class={} conf={:.2}", c.class_id, c.confidence);
    }
    Ok(())
}
```

## Matching Official Python Code

```python
from ultralytics import YOLO

model = YOLO("yolo26s-cls.pt")
results = model.predict("examples/bus.jpg", imgsz=224)
probs = results[0].probs
for i in probs.top5:
    print(i, probs.data[i].item())
```

## API and Configuration Details

### `classify::config_builder()` (returns `model::config::base::Builder`)

The classify builder defaults input size to **224** (not 640) and class count to **ImageNet (1000)**. Other builder methods are identical to detect (`with_scale`/`with_device`/`with_dtype`/`with_input_size`/`with_image_size`/`with_labels_count`, and so on; see [detect.md](detect.md)).

| Builder method | Default | Description |
| --- | --- | --- |
| `with_input_size(n)` | `224` | Classification default is 224; custom sizes are snapped to 32. |
| `with_labels_count(n)` | `IMAGENET.len()` | Set this for custom classification datasets. |

### Model and Prediction

| API | Description |
| --- | --- |
| `classify::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a classification model. |
| `classify::Model::from_pt_file(path, config)` | Loads from official `.pt`. |
| `classify::Model::from_safetensors(weights, config)` | Loads from `.safetensors` bytes. |
| `model.forward_tensor(&input)` | Raw forward, returning `[B, nc]`. |
| `model.predict(&image)` | End-to-end inference returning `Vec<classify::Prediction>` sorted by confidence descending. |

### `classify::Prediction` Fields

- `class_id: u32`.
- `confidence: f32` (`[0,1]`).

> classify has no `PredictOptions` type alias because it does not filter; you receive all results and take top-k yourself.

## Differences from Official / Notes

- Official `probs` objects provide convenience methods such as `top1`/`top5`/`argmax`; this crate returns a sorted `Vec`, so `predictions[0]` is top-1.
- Official classify also supports `.classify()`/training; this crate's classify path is inference-only, and training goes through the unified `train` module (see [train.md](../train.md)).
- Class count defaults to ImageNet 1000 (and `labels_count` is inferred from checkpoint head shapes). The class-name table `default_labels::IMAGENET` requires the `default_labels` feature (disabled by default), or you can read names directly from `.pt` weights through `pt_loader::load_pt_metadata().names`.
