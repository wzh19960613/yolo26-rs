# Instance Segmentation Inference (segment)

This guide shows how to use the `segment` task root for YOLO26 instance segmentation: each image returns multiple instances, and each instance has a detection box plus a binary mask. Compared with detect, `predict()` accepts an extra `MaskOption` that controls whether masks are upsampled to source-image resolution.

Full API reference: [tasks.md](../tasks.md).

## When to Use

- You need pixel-level instance contours (people, cars, cells, and so on), not just boxes.
- You need to overlay masks back onto the source image for visualization or feed them downstream (OCR, measurement).
- You want detect-like results plus proto mask decoding.

## Preparation

- **Feature**: requires `--features segment` (not included in default features).
- **Weights**: official segmentation `.pt` weights are named like `yolo26s-seg.pt` (`-seg` suffix).

```bash
cargo build --release
```

## Rust Example

```rust
use yolo26_rs::{FilterOption, Image, MaskOption, segment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = segment::Model::from_file("yolo26s-seg.pt")?;
    let mask_opt = MaskOption { high_resolution: true };
    let segs = model.predict(&image, &FilterOption::default(), &mask_opt)?;
    for s in &segs {
        let d = &s.detection;
        let mask = &s.mask;
        println!(
            "class={} conf={:.2} bbox={:?} mask={}x{}",
            d.class_id, d.confidence, d.bbox,
            mask.width, mask.height,
        );
        let _ = mask.data();
    }
    Ok(())
}
```

## Matching Official Python Code

```python
from ultralytics import YOLO

model = YOLO("yolo26s-seg.pt")
results = model.predict("examples/bus.jpg", imgsz=640, conf=0.25)
for r in results:
    for i, b in enumerate(r.boxes):
        m = r.masks[i]
        print(b.cls.item(), b.conf.item(), b.xyxy[0].tolist(), m.data.shape)
```

## API and Configuration Details

`segment::config_builder()` is identical to detect (default 640, COCO 80 classes). See [detect.md](detect.md) for builder methods.

### Model and Prediction

| API | Description |
| --- | --- |
| `segment::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a segmentation model. |
| `segment::Model::from_pt_file(path, config)` | Loads from official `.pt`. |
| `segment::Model::from_safetensors(weights, config)` | Loads from `.safetensors` bytes. |
| `model.forward_tensor(&input)` | Raw forward, returning `(predictions, proto)`. |
| `model.predict(&image, &filter, &mask)` | End-to-end inference returning `Vec<segment::Prediction>`. |

### `MaskOption`

| Field | Default | Description |
| --- | --- | --- |
| `high_resolution` | `false` | When `true`, masks are upsampled to source-image resolution; when `false`, native low-resolution model masks are kept. |

### `Mask` (`segment::Prediction.mask`)

| API | Description |
| --- | --- |
| `mask.width` / `mask.height` (`u16`) | Mask dimensions. |
| `mask.logits: Vec<f32>` | Per-pixel logits; `>0` means the pixel belongs to the instance. |
| `mask.data()` | Returns `Vec<u8>` binary mask (`logit>0` is 1). |
| `mask.get(x, y)` | Whether one pixel belongs to the instance. |
| `mask.resize_checked(w, h)` | Bilinear interpolation plus thresholding, returning `Result<Mask>` (recommended). |
| `mask.resize(w, h)` (`#[deprecated]`) | Panic version; do not use in new code. |

### `segment::Prediction` Fields

- `detection: detect::Prediction` (`bbox`/`confidence`/`class_id`).
- `mask: Mask`.

## Differences from Official / Notes

- Official masks are usually returned at source-image resolution; this crate defaults to the native low-resolution mask and requires `MaskOption { high_resolution: true }` for upsampling.
- Official `masks.data` is a tensor; this crate's `Mask` stores `Vec<f32>` logits and provides `data()` for binary masks.
- Boxes and masks are both in source-image coordinates, and `Prediction::translated(dx,dy)` translates them together (used by sliced detection).
- Segment target mask encoding during training (overlap/per-instance) is controlled by the `train` module; see [train.md](../train.md).
