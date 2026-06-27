# Semantic Segmentation Inference (semantic)

This guide shows how to use the `semantic` task root for YOLO26 semantic segmentation: predict one class for each pixel and output a dense class map for the whole image (not instance-level). This differs from instance segmentation (`segment`): semantic does not distinguish individuals, only the class of each region of pixels.

Full API reference: [tasks.md](../tasks.md).

## When to Use

- Per-pixel classification tasks, such as autonomous-driving scene understanding (Cityscapes) or remote-sensing land-cover classification.
- You do not care about instance boundaries, only region classes.
- You need a dense logits map for custom argmax/visualization.

## Preparation

- **Feature**: requires `--features semantic` (not included in default features).
- **Weights**: official semantic segmentation `.pt` weights are named like `yolo26s-sem.pt` (`-sem` suffix).
- **Default classes**: Cityscapes (19 classes). `labels_count` is inferred from checkpoint head shapes; `default_labels::CITYSCAPES` requires the `default_labels` feature (disabled by default), or read `.pt` names through `pt_loader::load_pt_metadata().names`.

```bash
cargo build --release
```

## Rust Example

```rust
use yolo26_rs::{Image, MaskOption, semantic};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = semantic::Model::from_file("yolo26s-sem.pt")?;
    let pred = model.predict(&image, &MaskOption { high_resolution: true })?;
    let class_ids = pred.class_ids();
    println!("map={}x{} classes={}", pred.width, pred.height, pred.class_ids().len());
    for y in 0..pred.height as usize {
        for x in 0..pred.width as usize {
            let cid = class_ids[y * pred.width as usize + x];
            let _ = cid;
        }
    }
    Ok(())
}
```

## Matching Official Python Code

The official YOLO26 semantic segmentation path runs through a dedicated semantic segmentation model/task interface. A typical usage:

```python
from ultralytics import YOLO

model = YOLO("yolo26s-sem.pt")
results = model.predict("examples/bus.jpg", imgsz=640)
mask = results[0]
```

> The exact official semantic segmentation result object shape should follow the current Ultralytics documentation. This crate aligns on dense per-pixel logits.

## API and Configuration Details

`semantic::config_builder()` defaults input size to 640 and class count to **Cityscapes**. Builder methods match detect (see [detect.md](detect.md)); use `with_labels_count` for custom datasets.

### Model and Prediction

| API | Description |
| --- | --- |
| `semantic::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a semantic segmentation model. |
| `semantic::Model::from_pt_file(path, config)` | Loads from official `.pt`. |
| `semantic::Model::from_safetensors(weights, config)` | Loads from `.safetensors` bytes. |
| `model.forward_tensor(&input)` | Raw forward, returning `[B, nc, H, W]` dense logits. |
| `model.predict(&image, &mask)` | End-to-end inference returning `semantic::Prediction`. |

### `semantic::Prediction`

| API | Description |
| --- | --- |
| `width` / `height` | Output-map size (affected by `MaskOption`). |
| `classes` | Class count `nc`. |
| `logits` | Per-pixel logits. |
| `class_id(x, y)` | Returns the argmax class id for that pixel. |
| `class_ids()` | Returns a `Vec<u32>` class map with the same size as the output map. |
| `resize(width, height)` | Resamples the logits map. |

### `MaskOption`

Semantic segmentation also uses `MaskOption { high_resolution }` to control whether the output map is upsampled to source-image resolution (same as segment).

## Differences from Official / Notes

- Official semantic segmentation result access can vary by version; this crate provides stable `Prediction { logits, class_ids() }`.
- semantic performs no NMS and produces no boxes; it is pure dense per-pixel classification.
- During training, unselected semantic segmentation pixels are marked ignore and excluded from loss; see [train.md](../train.md).
