# Oriented Object Detection Inference (obb)

This guide shows how to use the `obb` task root for YOLO26 Oriented Bounding Box detection: each object is represented by a rotated rectangle with an angle, commonly used for aerial/remote-sensing (DOTA), documents, and objects arranged in arbitrary orientations. `BBox` fields are center point + width/height + angle, unlike detect's axis-aligned boxes.

Full API reference: [tasks.md](../tasks.md).

## When to Use

- Objects have clear orientation (ships, vehicles, text lines, cells), where axis-aligned boxes overlap or waste area.
- DOTA-style oriented box detection tasks.
- You need to convert rotated boxes back to axis-aligned boxes (`axis_aligned_bbox`) or visualize corners.

## Preparation

- **Feature**: requires `--features obb` (not included in default features).
- **Weights**: official OBB `.pt` weights are named like `yolo26s-obb.pt` (`-obb` suffix).
- **Default classes**: DOTA (15 classes). `labels_count` is inferred from checkpoint head shapes; `default_labels::DOTA` requires the `default_labels` feature (disabled by default), or read names through `.pt` `pt_loader::load_pt_metadata().names`.

```bash
cargo build --release
```

## Rust Example

```rust
use yolo26_rs::{FilterOption, Image, obb};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/boats.jpg")?;
    let model = obb::Model::from_file("yolo26s-obb.pt")?;
    let dets = model.predict(&image, &FilterOption::default())?;
    for d in &dets {
        println!(
            "class={} conf={:.2} bbox={:?}",
            d.class_id, d.confidence, d.bbox,
        );
        let _ = d.bbox.axis_aligned_bbox();
    }
    Ok(())
}
```

## Matching Official Python Code

```python
from ultralytics import YOLO

model = YOLO("yolo26s-obb.pt")
results = model.predict("examples/boats.jpg", imgsz=640, conf=0.25)
for r in results:
    for i, b in enumerate(r.obb):
        print(b.cls.item(), b.conf.item(), b.xywhr[i].tolist())
```

## API and Configuration Details

`obb::config_builder()` defaults input size to 640 and class count to **DOTA**. Builder methods are identical to detect (see [detect.md](detect.md)).

### Model and Prediction

| API | Description |
| --- | --- |
| `obb::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads an oriented box model. |
| `obb::Model::from_pt_file(path, config)` | Loads from official `.pt`. |
| `obb::Model::from_safetensors(weights, config)` | Loads from `.safetensors` bytes. |
| `model.forward_tensor(&input)` | Raw forward, returning `[B, A, nc+5+1]` (includes angle). |
| `model.predict(&image, &filter)` | End-to-end inference returning `Vec<obb::Prediction>`. |

### `obb::BBox` (rotated box, distinct from `crate::BBox`)

| Field/method | Description |
| --- | --- |
| `center_x: f32` / `center_y: f32` | Rotated rectangle center (source-image coordinates). |
| `width: f32` / `height: f32` | Width and height before rotation. |
| `angle: f32` | Rotation angle in radians. |
| `axis_aligned_bbox()` | Returns the enclosing axis-aligned `crate::BBox`. |
| `translate(dx, dy)` | Translates the center; angle remains unchanged. |

### `obb::Prediction` Fields

- `bbox: obb::BBox`, `confidence: f32`, `class_id: u32`.

## Differences from Official / Notes

- Official `obb.xywhr` is a tensor; this crate uses the strongly typed `obb::BBox { center_x, center_y, width, height, angle }`.
- Visualizing rotated boxes requires computing the four corners yourself with `angle.sin_cos()`.
- OBB evaluation uses rotated/probIoU mAP (see the eval section in [train.md](../train.md)).
