# YOLOE Text Prompt Inference (Open-Vocabulary Segmentation/Detection)

Pass an arbitrary list of class names, and the model uses CLIP text embeddings as the classifier for open-vocabulary detection/segmentation. Class names are encoded by the MobileCLIP2-b CLIP encoder from `mobileclip2-rs`, then aligned by the RepRTA held by `Model`, matching the official `set_classes` flow.

Full API reference: [yoloe.md](../yoloe.md).

## When to Use

- Open vocabularies where classes are not fixed and change with business logic (for example, "person", "bus", "my_custom_thing").
- You do not want to retrain the classification head for every dataset and want to define classes on the fly with text.

## Preparation

- **Feature**: `--features yoloe-text` (included by aggregate `yoloe`).
- **Weights**: prefer official `.pt`; YOLOE segmentation weights such as `yoloe-26s-seg.pt`, or detection weights such as `yoloe-26s.pt`.
- **CLIP resources**: prefer official MobileCLIP2-b weights `mobileclip2_b.pt` and `tokenizer.json`. The caller provides them when constructing `ClipTextEncoder` (they are not inside the YOLOE checkpoint and have no default path).

## Rust Example

```rust
use yolo26_rs::{FilterOption, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;
use yolo26_rs::yoloe::prompt::text_encoder::ClipTextEncoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Model constructs RepRTA from model.23.reprta during loading.
    let model = Model::from_file("yoloe-26n-seg.pt")?;
    let image = yolo26_rs::Image::from_file("examples/bus.jpg")?;

    // Construct the CLIP encoder once and reuse it across sessions.
    let encoder = ClipTextEncoder::from_files(
        "mobileclip2_b.pt",
        "tokenizer.json",
    )?;

    // Class names -> CLIP embeddings -> RepRTA alignment -> session.
    let session = Session::text(&encoder, model.reprta(), ["person", "bus", "car"])?;

    let segs = model.predict(
        &image, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    for s in &segs {
        println!("class={} conf={:.2}", s.detection.class_id, s.detection.confidence);
    }
    Ok(())
}
```

The detection (non-segmentation) path uses `yoloe::detect::Model` with signature `predict(&image, &session, &filter)` (no `mask` parameter).

## When External Embeddings Already Exist

If class-name embeddings are already encoded externally (for example, produced by official CLIP), use `Session::text_with_embeddings` to inject them directly and skip CLIP encoding:

```rust
use yolo26_rs::yoloe::{EmbeddingTable, prompt::session::Session};

let table = EmbeddingTable::new(embeddings, class_names)?;
let session = Session::text_with_embeddings(table)?;
```

## Matching Official Python

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")
model.set_classes(["person", "bus", "car"], model.get_text_pe(["person", "bus", "car"]))
results = model.predict("examples/bus.jpg")
```

## API Quick Reference

### Construction and Prediction

| API | Description |
| --- | --- |
| `Model::from_file(path)` | Loads a YOLOE seg model and constructs RepRTA (`model.reprta()`) at the same time. |
| `ClipTextEncoder::from_files(weights, tokenizer)` | Constructs a reusable CLIP encoder (also supports `from_bytes` and `new`). |
| `Session::text(&encoder, model.reprta(), classes)` | CLIP encoding + RepRTA alignment to construct a text-prompt session. `classes` accepts an `AsRef<str>` iterator (`["a","b"]`, `Vec<&str>`, and so on). |
| `Session::text_with_embeddings(table)` | Injects external embeddings and skips CLIP encoding. |
| `model.predict(&image, &session, &filter, &mask)` | seg prediction; detect uses `predict(&image, &session, &filter)`. |

### Config builder

| API | Description |
| --- | --- |
| `yoloe::config_builder()` | Returns a `Config` builder. |
| `.with_rep_rta_enabled(bool)` | Enables/disables RepRTA alignment (enabled by default). |
| `.with_image_size(size)` | Input resolution. |
| `.with_scale(scale)` | Model scale (N/S/M/L/X). |
| `.build()` | Constructs `Config`, passed to `Session::text_with_config`. |

## Differences from Official / Notes

- Official `set_classes` calls CLIP on the Python side; this crate calls MobileCLIP2-b CLIP through the `mobileclip2-rs` dependency (`ClipTextEncoder`), and `Session::text` encodes class names into `[classes, 512]` L2-normalized embeddings.
- The official inference path is CLIP -> RepRTA -> score. RepRTA is held by `Model` after loading from `model.23.reprta` (`Model::reprta()`), and `Session::text` borrows it for alignment without manual loading or checkpoint paths.
- The caller constructs the CLIP encoder once (`ClipTextEncoder::from_files` / `from_bytes` / `new`) and reuses it by passing `&encoder`. CLIP resources are not inside the YOLOE checkpoint and have no default path.
- YOLOE one-to-one heads do not need NMS by design; `FilterOption.agnostic_nms` is currently a no-op.
- Converted `.safetensors` weights remain usable, but documentation and examples default to official `.pt`.
