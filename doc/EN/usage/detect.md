# Object Detection Inference (detect)

This guide shows how to run YOLO26 object detection inference through the `detect` task root in `yolo26-rs`: load official `.pt` weights, run one image, and filter results by confidence and class. The six inference task roots share the same shape, and detect is the most basic one, so it can be used as a template for other tasks.

Full API reference: [tasks.md](../tasks.md).

## When to Use

- You need to run YOLO26 detection in Rust services, WASM, embedded systems, or environments without Python.
- You need strongly typed, lightweight detection results (`BBox`/`confidence`/`class_id`) and want to control visualization, saving, or postprocessing yourself.
- You need local inference on CPU, Metal, or CUDA (`device::auto()` selects automatically).

## Preparation

- **Feature**: detect is enabled by default, so **no extra feature is required**. For GPU, compile with `--features metal` (Apple) or `--features cuda` (NVIDIA, requires nvcc).
- **Weights**: official `.pt` (default `pt` feature), such as `yolo26s.pt`, loaded directly with `from_pt_file`; converted `.safetensors` can also be used with `from_safetensors`.
- **Dependency**: examples use `Image::from_file` to load images in one line, which requires the `image` feature (included in `default`). Without it, construct `Image::new(...)` with any decoder.

```bash
cargo build --release
cargo build --release --features metal
```

## Rust Example

```rust
use yolo26_rs::{FilterOption, Image, detect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let detections = model.predict(&image, &FilterOption::default())?;
    for d in &detections {
        println!(
            "class={} conf={:.2} bbox={:?}",
            d.class_id, d.confidence, d.bbox,
        );
    }
    Ok(())
}
```

Keep only person and bus (COCO class ids 0 and 5) with `FilterOption`:

```rust
use yolo26_rs::{FilterOption, Image, detect};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let filter = FilterOption {
        confidence_threshold: 0.25,
        class_filter: vec![0, 5],
    };
    let detections = model.predict(&image, &filter)?;
    Ok(())
}
```

## Matching Official Python Code

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")
results = model.predict(
    "examples/bus.jpg",
    imgsz=640,
    conf=0.25,
    classes=[0, 5],
)
for r in results:
    for b in r.boxes:
        print(b.cls.item(), b.conf.item(), b.xyxy[0].tolist())
```

## API and Configuration Details

### `detect::config_builder()` (returns `model::config::base::Builder`)

| Builder method | Type/default | Description |
| --- | --- | --- |
| `with_scale(s)` | `Scale::{N,S,M,L,X}`, default `N` | Matches `n/s/m/l/x` in weight file names. |
| `with_device(d)` | `Device`, default `Device::Cpu` | Use `device::auto()` to select GPU automatically, or `DeviceSpec::Metal(0).to_device()`. |
| `with_dtype(t)` | `DType`, default **Auto** | Forces runtime precision (`F16`/`F32`) and overrides Auto. **When omitted, default `DtypeRequest::Auto`** resolves from device + target arch + weight dtype: GPU (CUDA/Metal) follows the weight dtype (F16 weights -> F16), while CPU / wasm32 forces F32 even for F16 weights. CPU deployment does not need manual `with_dtype(F32)`. |
| `with_input_size(n)` | `usize`, default `MODEL_INPUT_SIZE` (640) | Square input, snapped to a multiple of 32. |
| `with_image_size(w,h)` | `usize, usize` | Non-square `(width,height)` input, also snapped to 32. |
| `with_max_predictions(n)` | `usize`, default `300` | Maximum predictions retained by the head. |
| `with_rectangular_padding(b)` | `bool`, default `true` | Whether preprocessing preserves rectangles instead of square letterbox. |
| `with_labels_count(n)` | `usize`, default COCO (80) | Set for custom datasets. |

### Model and Prediction

| API | Description |
| --- | --- |
| `detect::Model::from_file(path)` | Auto-detects `.pt` / `.safetensors` and loads. |
| `detect::Model::from_pt_file(path, config)` | Loads directly from official `.pt` (through `pt_loader`). |
| `detect::Model::from_safetensors(weights, config)` | Loads from `.safetensors` bytes. |
| `model.forward_tensor(&input)` | Raw forward on a preprocessed tensor, returning `[B, A, nc+4+1]`. |
| `model.predict(&image, &filter)` | End-to-end inference returning `Vec<detect::Prediction>`. |

### `FilterOption` (alias of `detect::PredictOptions`)

| Field | Default | Description |
| --- | --- | --- |
| `confidence_threshold` | `0.25` | Detections below this score are discarded. |
| `class_filter` | `vec![]` | Empty keeps all classes; otherwise only listed class ids are kept. |

### `detect::Prediction` Fields

- `bbox: BBox` (`x_min/y_min/x_max/y_max`, source-image coordinates).
- `confidence: f32` (`[0,1]`).
- `class_id: u32`.

## Differences from Official / Notes

- Official `YOLO("model.pt")` consumes `.pt` directly; this crate also uses `.pt` as the primary path (`from_pt_file`, default `pt` feature), with `from_safetensors` as an optional byte-loading path.
- Official APIs accept paths/URLs/numpy/PIL; this crate's high-level API uses the unified `Image` type, with `Image::from_file` (under the `image` feature) or caller-constructed `Image::new`.
- Official result objects include visualization/saving/DataFrame helpers; this crate returns lightweight Rust structures.
- GPU builds: do not use `--all-features` without the CUDA toolkit, because the `cuda` feature requires `nvcc`. See the quick start in [index.md](../index.md).
