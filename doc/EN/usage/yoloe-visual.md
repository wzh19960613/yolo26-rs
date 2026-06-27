# YOLOE Visual Prompt Inference (Box/Mask Example Prompts)

Provide boxes (or masks) for a few example objects in the current image, and the model generates predictions for same-class objects. Internally it performs letterbox, SAVPE encoding, and postprocessing.

Full API reference: [yoloe.md](../yoloe.md).

## When to Use

- The same image contains multiple objects of the same class, and one or two annotated boxes can retrieve all of them.
- The class is difficult to describe accurately with text and is better specified by example.
- You need per-image image-specific visual prompt inference.

## Preparation

- **Feature**: `--features yoloe-visual` (included by aggregate `yoloe`).
- **Weights**: requires a `-seg` `.pt` checkpoint containing official SAVPE weights, such as `yoloe-26s-seg.pt`. **Pure detection weights (`yoloe-26s.pt`, or matching `.safetensors`) do not contain SAVPE** and will fail at runtime if used with visual prompts.

## Rust Example (Box Visual Prompt Segmentation)

```rust
use yolo26_rs::{FilterOption, Image, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;
use yolo26_rs::yoloe::prompt::visual::Visual;
use yolo26_rs::yoloe::visuals::VisualSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = Model::from_file("yoloe-26s-seg.pt")?;

    // Visual prompts are per-image: create a new session for each image.
    let prompts = vec![
        Visual::from_box(0, [10.0, 20.0, 90.0, 160.0])?,
        Visual::from_box(1, [120.0, 30.0, 220.0, 180.0])?,
    ];
    let session = Session::visual(prompts.clone())?;

    let segs = model.predict_visual_prompts(
        &image, &prompts, VisualSource::Boxes, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    for s in &segs {
        println!("class={} conf={:.2}", s.detection.class_id, s.detection.confidence);
    }
    Ok(())
}
```

Mask form: replace `Visual::from_box(...)` with `Visual::from_mask(...)`, and replace `VisualSource::Boxes` with `VisualSource::Masks(&source_masks)`, where `source_masks` is a `[prompts, H, W]` mask tensor in source-image coordinates.

The detection (non-segmentation) path uses `yoloe::detect::Model` with signature `predict_visual_prompts(&image, &prompts, source, &session, &filter)` (no `mask` parameter).

## Cross-Image Visual Prompt (Reference Image -> Target Image)

`predict_visual_prompts` is a **single-image** path. To "give an example on image A and use it to recognize image B", use the two-step API aligned with official `predictor.get_vpe()` + `set_classes(names, vpe)`:

```rust
use yolo26_rs::{FilterOption, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;
use yolo26_rs::yoloe::prompt::visual::Visual;
use yolo26_rs::yoloe::visuals::VisualSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_file("yoloe-26n-seg.pt")?;

    // Step 1: reference image A + one box on A -> reusable vpe (image-agnostic embedding).
    let image_a = yolo26_rs::Image::from_file("examples/bus.jpg")?;
    let prompts_a = vec![Visual::from_box(0, [49.0, 399.0, 247.0, 902.0])?];
    let vpe = model.encode_visual_prompts(&image_a, &prompts_a, VisualSource::Boxes)?;

    // Step 2: use vpe as the classifier to identify any image B (normal predict path).
    let session = Session::text_with_embeddings(vpe)?;
    let image_b = yolo26_rs::Image::from_file("examples/boats.jpg")?;
    let segs = model.predict(
        &image_b, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    Ok(())
}
```

Matching official Python: `vpe = model.predictor.get_vpe(image_a, visuals); model.set_classes(names, vpe); model.predict("boats.jpg")`.

## API Quick Reference

### Visual Prompt Types

| API | Description |
| --- | --- |
| `Visual::from_box(class_id, [x1,y1,x2,y2])` | Box prompt element (source-image coordinates). |
| `Visual::from_mask(class_id, [x1,y1,x2,y2])` | Mask prompt element (source-image coordinates; actual mask comes from `VisualSource::Masks`). |
| `VisualSource::Boxes` / `VisualSource::Masks(&tensor)` | Source selector for `predict_visual_prompts`, deciding whether boxes are rasterized or source masks are consumed. |
| `BatchVisuals::from_boxes(items, target_size, scale, device)` | Multi-image batch helper returning `tensor` + `class_ids`. |

### Model and Prediction

| API | Description |
| --- | --- |
| `Model::predict_visual_prompts(&image, &prompts, source, &session, &filter, &mask)` | Single-image box or mask visual-prompt seg. |
| `Model::encode_visual_prompts(&reference_image, &prompts, source)` | Cross-image: encode a reusable `EmbeddingTable` (official `vpe`) on a reference image. |
| `Session::visual(prompts)` | Visual prompt session (created per image). |

> `predict_visual_prompts` already performs letterbox + SAVPE automatically. The session does not precompute SAVPE embeddings because SAVPE requires backbone features and can only run during forward.

## Differences from Official / Notes

- Official `predict(visual_prompts=...)` accepts images/boxes/masks directly; this crate provides a typed `predict_visual_prompts` single-image entry point, with box/mask selected by `VisualSource`.
- **Pure detection weights (`yoloe-26s.pt`, or matching `.safetensors`) do not contain SAVPE**: visual prompt calls on them fail at runtime. To get visual-prompt detection, load `-seg` weights as `yoloe::detect::Model`.
- Visual prompt class names are placeholders `visual_class_{original_id}`; original ids are provided by the `class_ids` returned from `BatchVisuals::from_boxes()` for multi-image mapping.
- vpe is an image-agnostic embedding, but **encoding still requires backbone features from reference image A** (by SAVPE design). One encoded vpe can identify many target images, but the prompt class set is fixed at encoding time.
