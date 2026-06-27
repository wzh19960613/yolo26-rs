# Pose / Keypoint Inference (pose)

This guide shows how to use the `pose` task root for YOLO26 pose inference: each detected person/object has a detection box plus a set of keypoints (for example, the COCO 17-point skeleton). pose uses the specialized `ForPose` config, which can customize keypoint count and per-keypoint dimensions.

Full API reference: [tasks.md](../tasks.md).

## When to Use

- Human/animal pose estimation (action recognition, motion analysis, gestures).
- You need skeleton keypoints + visibility for each instance.
- You need downstream computation on keypoint coordinates over boxes.

## Preparation

- **Feature**: requires `--features pose` (not included in default features).
- **Weights**: official pose `.pt` weights are named like `yolo26s-pose.pt` (`-pose` suffix).
- **Default**: 17 keypoints, 3 values per point (x, y, visibility).

```bash
cargo build --release
```

## Rust Example

```rust
use yolo26_rs::{FilterOption, Image, pose};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = pose::Model::from_file("yolo26s-pose.pt")?;
    let poses = model.predict(&image, &FilterOption::default())?;
    for p in &poses {
        println!(
            "class={} conf={:.2} bbox={:?} kpts={}",
            p.class_id, p.confidence, p.bbox,
            p.keypoints.len(),
        );
        for (i, k) in p.keypoints.iter().enumerate() {
            println!("  kp{i}: ({:.1},{:.1}) vis={:?}", k.x, k.y, k.visibility);
        }
    }
    Ok(())
}
```

## Matching Official Python Code

```python
from ultralytics import YOLO

model = YOLO("yolo26s-pose.pt")
results = model.predict("examples/bus.jpg", imgsz=640, conf=0.25)
for r in results:
    for i, b in enumerate(r.boxes):
        kpts = r.keypoints[i]
        print(b.cls.item(), b.conf.item(), b.xyxy[0].tolist(), kpts.xy.shape)
```

## API and Configuration Details

pose uses the specialized `Config = model::config::ForPose`, which contains `base: Base` plus keypoint parameters.

### `pose::config_builder()` (returns `model::config::for_pose::Builder`)

It inherits all base builder methods (`with_scale`/`with_device`/`with_dtype`/`with_input_size`/`with_labels_count`, and so on; see [detect.md](detect.md)) and adds:

| Builder method | Default | Description |
| --- | --- | --- |
| `with_keypoints_count(n)` | `17` | Number of keypoints per instance. |
| `with_keypoint_dims(n)` | `3` | Number of values per keypoint (at least 2: x/y; COCO uses 3: x/y/visibility). |

### Model and Prediction

| API | Description |
| --- | --- |
| `pose::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads a pose model. |
| `pose::Model::from_pt_file(path, config)` | Loads from official `.pt`. |
| `pose::Model::from_safetensors(weights, config)` | Loads from `.safetensors` bytes. |
| `model.forward_tensor(&input)` | Raw forward, returning `[B, A, nc+5+nk*kd]`. |
| `model.predict(&image, &filter)` | End-to-end inference returning `Vec<pose::Prediction>`. |

### `pose::Prediction` Fields

- `bbox: BBox`, `confidence: f32`, `class_id: u32` (same as detect).
- `keypoints: Vec<Keypoint>`.

### `Keypoint` Fields

- `x: f32`, `y: f32` (source-image coordinates).
- `visibility: Option<f32>` (keypoint visibility/confidence).

## Differences from Official / Notes

- Official `keypoints` objects provide tensor views such as `xy`/`xyn`/`conf`; this crate returns strongly typed `Vec<Keypoint>` with optional `visibility`.
- You need to maintain skeleton connections yourself (such as the COCO 17-point bones).
- For custom keypoint datasets (animal 17 points, hand 21 points, and so on), ensure `with_keypoints_count`/`with_keypoint_dims` match the weights.
