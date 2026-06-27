# SAHI Sliced Detection (detect::sahi)

This guide shows how to use `detect::sahi` for SAHI (Slicing Aided Hyper Inference)-style small-object detection: split a large image into overlapping slices, run inference on each slice, translate coordinates back to the source image, then merge/deduplicate results. This can significantly improve recall for high-resolution, dense small-object scenes.

Full API reference: see the sahi section in [tasks.md](../tasks.md).

## When to Use

- Small objects in large aerial/remote-sensing/surveillance images (directly resizing to 640 loses small targets).
- You want to improve small-object recall without changing the model.
- You need control over slice size, overlap ratio, and merge strategy.

## Preparation

- **Feature**: enabled by default, **no extra feature required** (reuses the `detect` task).
- **Weights**: normal detect `.pt` weights, such as `yolo26s.pt`.
- **Image**: a high-resolution image (examples use `examples/boats.jpg`).

```bash
cargo build --release
```

## Rust Example

```rust
use yolo26_rs::{
    FilterOption, Image, detect,
    detect::sahi,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/boats.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let sahi_opts = sahi::Options {
        slice_width: 320,
        slice_height: 320,
        overlap_width_ratio: 0.25,
        overlap_height_ratio: 0.25,
        include_full_image: true,
        ..sahi::Options::default()
    };
    let dets = sahi::sliced_predict(&model, &image, &FilterOption::default(), &sahi_opts)?;
    println!("sahi detections: {}", dets.len());
    for d in &dets {
        println!("class={} conf={:.2}", d.class_id, d.confidence);
    }
    Ok(())
}
```

Generate slice windows only and schedule inference yourself (for example, multithreading or cross-device scheduling):

```rust
use yolo26_rs::{FilterOption, Image, detect, detect::sahi};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/boats.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let sahi_opts = sahi::Options {
        slice_width: 320, slice_height: 320,
        overlap_width_ratio: 0.25, overlap_height_ratio: 0.25,
        ..sahi::Options::default()
    };
    let windows = sahi::generate_slices(image.width, image.height, &sahi_opts);
    for w in &windows {
        let crop = image.crop(w.x, w.y, w.width, w.height)?;
        let mut part = model.predict(&crop, &FilterOption::default())?;
        for d in part.iter_mut() {
            *d = d.translated(w.x as f32, w.y as f32);
        }
    }
    Ok(())
}
```

## Matching Official Python Code

Official Ultralytics itself does not include SAHI; the common community approach uses the standalone `sahi` library with an Ultralytics model:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26s.pt",
    confidence_threshold=0.25,
)
result = get_sliced_prediction(
    "examples/boats.jpg",
    detection_model,
    slice_height=320, slice_width=320,
    overlap_height_ratio=0.25, overlap_width_ratio=0.25,
    perform_standard_pred=True,
)
for ann in result.object_prediction_list:
    print(ann.category.id, ann.score.value, ann.bbox.to_xyxy())
```

## API and Configuration Details

### `sahi::sliced_predict(detector, image, inference, options)`

End-to-end: internally runs `generate_slices` -> per-slice `predict` -> coordinate translation -> `merge_detections`.

| API | Description |
| --- | --- |
| `sahi::sliced_predict(...)` | Sliced inference + merging, returning `Vec<detect::Prediction>`. |
| `sahi::generate_slices(w, h, &options)` | Generates only slice windows `Vec<SliceWindow>`. |
| `sahi::merge_detections(dets, &options)` | Merges/deduplicates cross-slice detections. |

### `sahi::Options`

| Field | Default | Description |
| --- | --- | --- |
| `slice_width` / `slice_height` | `640` / `640` | Pixel size of one slice. |
| `overlap_width_ratio` / `overlap_height_ratio` | `0.2` / `0.2` | Overlap ratio between neighboring slices. |
| `include_full_image` | `false` | Whether to additionally run full-image inference and merge it. |
| `merge_strategy` | `GreedyNmm` | Merge strategy: `Nms` (non-maximum suppression) or `GreedyNmm` (greedy non-maximum merging). |
| `match_metric` | `Ios` | Match metric: `Iou` (intersection over union) or `Ios` (intersection over smaller box area). |
| `match_threshold` | `0.5` | Minimum match metric for merging. |
| `class_agnostic` | `false` | Whether boxes of different classes may merge. |

### `SliceWindow` (single slice)

| Field | Description |
| --- | --- |
| `x` / `y` | Top-left source-image coordinate of the slice. |
| `width` / `height` | Slice size. |

## Differences from Official / Notes

- This crate's SAHI is a native Rust implementation that reuses the same `detect::Model` and does not depend on the third-party Python `sahi` package.
- Sliced inference repeatedly calls the same model; if you need parallelism, use `generate_slices` and schedule it yourself (respecting Candle model thread-safety constraints).
- `include_full_image: true` merges full-image results with slice results, often useful for retaining large objects.
- Official `sahi` library `Nms`/`GreedyNmm` behavior is aligned in this crate; the default is `GreedyNmm + Ios`.
