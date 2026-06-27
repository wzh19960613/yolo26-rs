# YOLOE Prompt-Free Inference (Built-In LRPC Open Vocabulary)

The checkpoint contains a fixed 4585-class LRPC open vocabulary. Inference needs no text/visual prompt and directly predicts class ids. Readable class names are provided by the built-in `LRPC_VOCAB`.

Full API reference: [yoloe.md](../yoloe.md).

## When to Use

- You do not want to manually specify classes and want to use the checkpoint's built-in 4585-class open vocabulary directly.
- Your class set is fixed and aligned with the LRPC vocabulary.

## Preparation

- **Feature**: `--features yoloe-pf` (included by aggregate `yoloe`). The built-in 4585-name vocabulary `LRPC_VOCAB` belongs to `yoloe-pf`.
- **Weights**: prefer official prompt-free `.pt` checkpoints, such as `yoloe-26s-seg-pf.pt` (`-pf` suffix, contains LRPC head).

## Rust Example

```rust
use yolo26_rs::{FilterOption, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_file("yoloe-26s-seg-pf.pt")?;
    let image = yolo26_rs::Image::from_file("examples/bus.jpg")?;

    // Build a session with the built-in 4585 vocabulary names (requires yoloe-pf).
    let session = Session::prompt_free_default()?;

    let segs = model.predict(
        &image, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    for s in &segs {
        let name = yolo26_rs::default_labels::LRPC_VOCAB
            .get(s.detection.class_id as usize)
            .copied()
            .unwrap_or("unknown");
        println!("{} conf={:.2}", name, s.detection.confidence);
    }
    Ok(())
}
```

The detection (non-segmentation) path uses `yoloe::detect::Model` with signature `predict(&image, &session, &filter)` (no `mask` parameter).

## Without yoloe-pf

`LRPC_VOCAB` is only compiled under the `yoloe-pf` feature. Without it, use `Session::prompt_free()` (class names are `pf_XXXX` placeholders, while scoring still comes from the checkpoint's LRPC head):

```rust
let session = Session::prompt_free()?; // class names are pf_0, pf_1, ...
```

Or, with `.pt` weights, read class names directly from checkpoint metadata: `pt_loader::load_pt_metadata(path).names`.

## Matching Official Python

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26s-seg-pf.pt")
results = model.predict("examples/bus.jpg")
for r in results:
    for b in r.boxes:
        print(model.names[b.cls.item()], b.conf.item())
```

## API Quick Reference

| API | Description |
| --- | --- |
| `Session::prompt_free_default()` | Constructs a session with built-in `LRPC_VOCAB` (4585 names; requires `yoloe-pf`). Prediction class ids directly index this table for readable names. |
| `Session::prompt_free()` | Fallback entry without the built-in vocabulary; class names are `pf_XXXX` placeholders. |
| `Session::prompt_free_with_embeddings(table)` | Constructs from external LRPC embeddings (advanced). |
| `Model::from_file(path)` | Loads a prompt-free checkpoint. |
| `default_labels::LRPC_VOCAB` | Built-in 4585 class names `&[&str; 4585]`, row order aligned with checkpoint `vocab.weight`. |

## Differences from Official / Notes

- Official `YOLOE("*-pf.pt").predict(...)` uses checkpoint `model.names`; with `.pt` weights, this crate can also read the same table directly via `pt_loader::load_pt_metadata(path).names`. `LRPC_VOCAB` is a built-in copy of the same table (aligned row order), useful when metadata is not read or when using `.safetensors`.
- Prompt-free checkpoint `model.23.lrpc.*.vocab.weight` has a fixed 4585 rows, not COCO 80. Prediction class ids directly index `LRPC_VOCAB` for readable names.
- Prompt-free needs no prompt (table/visual); it scores directly with checkpoint-internal LRPC `loc`/`vocab` outputs.
