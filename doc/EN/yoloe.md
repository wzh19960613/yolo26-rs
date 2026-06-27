# YOLOE

Official reference: [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/).

The YOLOE module covers YOLOE-26 text prompt, visual prompt, prompt-free vocabulary, RepRTA, SAVPE, LRPC, and open-vocabulary detect/segment heads. The implementation references Ultralytics YOLOE/YOLO26 documentation and `ultralytics/models/yolo/yoloe/*`, `ultralytics/data/augment.py`, and `ultralytics/nn/modules/head.py`; YOLOE weights and examples outside the YOLO26 family are background only and are not part of this crate's feature scope.

Official `.pt` is the primary YOLOE weight format in this crate (default `pt` feature): `Segment::from_pt_file(...)` / `Detect::from_pt_file(...)` load official `yoloe-26*-seg.pt` / `-seg-pf.pt` directly (zero-argument `from_file(path)` can also infer automatically), and after training `Seg::save_pt` writes official PyTorch-readable `.pt` files (verified with `torch.load`). Converted `.safetensors` weights are also supported: `from_file(path)` / `from_bytes(bytes)` / `from_safetensors(weights, config)` infer scale and head layout from checkpoint shapes. `Checkpoint::parse()` and `Session::from_checkpoint()` parse scale, segmentation, and prompt-free semantics from the file name, accepting both `.pt` and `.safetensors` suffixes.

> Naming: `Detect` / `Segment` align with the `Model` shape of the six task roots. `Session` is YOLOE's immutable prompt state.

## Quick Start

### Text prompt

`Session::text(&encoder, model.reprta(), classes)` uses the MobileCLIP2-b CLIP text encoder from the `mobileclip2-rs` dependency to encode class names into `[classes, 512]` L2-normalized embeddings (requires the `yoloe-text` feature). `encoder` is a `ClipTextEncoder` constructed once and borrowed repeatedly; `model.reprta()` returns the `Option<&RepRta>` held by `Model` after loading `model.23.reprta`; `classes` accepts any `AsRef<str>` (`&str`, `String`, `&&str`, and so on, without per-item `.into()` calls):

The text encoder uses [wzh19960613/mobileclip2-b-rs](https://github.com/wzh19960613/mobileclip2-b-rs). Recommended official resources:

- YOLOE segment checkpoint: [yoloe-26s-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt).
- MobileCLIP2-B text encoder weights: [mobileclip2_b.pt](https://huggingface.co/apple/MobileCLIP2-B/resolve/main/mobileclip2_b.pt).
- CLIP BPE tokenizer: [tokenizer.json](https://huggingface.co/openai/clip-vit-base-patch32/raw/main/tokenizer.json).

```rust
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::Session;
use yolo26_rs::yoloe::prompt::text_encoder::ClipTextEncoder;

let model = Model::from_file("yoloe-26s-seg.pt")?; // constructs RepRTA during loading

// Construct the CLIP encoder once and reuse it.
let encoder = ClipTextEncoder::from_files(
    "mobileclip2_b.pt",
    "tokenizer.json",
)?;

// Construct an immutable text-prompt session (reusable across images).
let session = Session::text(&encoder, model.reprta(), ["person", "bus"])?;
```

When external embeddings already exist, inject them directly with `Session::text_with_embeddings`:

```rust
use candle_core::{Device, Tensor};
use yolo26_rs::yoloe::{EmbeddingTable, Session};

let embeddings = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu)?;
let table = EmbeddingTable::new(embeddings, vec!["person".into(), "bus".into()])?;
let session = Session::text_with_embeddings(table)?;
```

### Visual prompt

```rust
use candle_core::Device;
use yolo26_rs::{
    ImageSize, Scale,
    yoloe::{Config, Session, VisualPrompt},
};

// Visual prompts are per-image: create a new session for each image; SAVPE runs during forward.
let session = Session::visual(vec![
    VisualPrompt::from_box(0, [10.0, 20.0, 90.0, 160.0])?,
])?;

// Low-level helper aligned with official visuals tensor construction.
let visuals = yolo26_rs::yoloe::Visuals::from_boxes(
    &[VisualPrompt::from_box(0, [10.0, 20.0, 90.0, 160.0])?],
    ImageSize::new(image_width, image_height),
    0.25,
    &Device::Cpu,
)?;
```

Official documentation recommends running image-specific prompts one image at a time. The official source also has a bbox-only batch branch that pads each image's bbox visual prompts into `visuals`. This crate keeps a low-level helper aligned with that YOLOE-26 source path, but it is not the preferred high-level API:

```rust
use candle_core::Device;
use yolo26_rs::{
    ImageSize,
    yoloe::{BatchVisuals, VisualPrompt, VisualBatchItem},
};

let batch = BatchVisuals::from_boxes(
    &[
        VisualBatchItem::new(
            ImageSize::new(1280, 720),
            vec![VisualPrompt::from_box(0, [100.0, 80.0, 320.0, 360.0])?],
        )?,
        VisualBatchItem::new(
            ImageSize::new(640, 640),
            vec![
                VisualPrompt::from_box(1, [20.0, 20.0, 200.0, 260.0])?,
                VisualPrompt::from_box(2, [260.0, 80.0, 520.0, 520.0])?,
            ],
        )?,
    ],
    ImageSize::square(640),
    0.125,
    &Device::Cpu,
)?;
// batch.tensor is [batch, max_prompts, h, w]; batch.class_ids[b][p] is the original class id.
```

### Prompt-free

```rust
use yolo26_rs::yoloe::Session;

let session = Session::prompt_free(vec!["person".into(), "bus".into()])?;
```

## Full API

### Configuration and State

| API | Description |
| --- | --- |
| `yoloe::config_builder()` | Returns the default YOLOE `Config` builder (aligned with task-root `config_builder()`). |
| `yoloe::Config::default()` | Text/visual prompt YOLOE config (device/dtype/image_size/max_predictions, aligned with task-root `Base`). |
| `Config::segmentation(scale)` | Segmentation-first YOLOE config. |
| `Config::prompt_free(scale)` | Prompt-free + LRPC config. |
| `Checkpoint::parse(name)` | Parses YOLOE checkpoint names such as `yoloe-26s-seg-pf.pt`; accepts `.pt` and `.safetensors` suffixes. |
| `Usage` | `TextPrompt`/`VisualPrompt`/`PromptFree` for inference prompt sources; `FineTune`/`LinearProbe`/`Validate` for fine-tuning, linear probing, and validation scenarios. |
| `State` | `Empty`, `Text`, `Visual`, `PromptFree`. |
| `Controller` | Manages prompt state. |
| `Session` | Immutable prompt state fixed at construction time (text/prompt_free can be reused across images; visual is created per image). Contains prompt, prediction config, scorer, and prompt table. |

### Session Constructors (Recommended Entry Points)

| Constructor | Description |
| --- | --- |
| `Session::text(&encoder, model.reprta(), classes)` | Uses reusable `ClipTextEncoder` to generate an `EmbeddingTable` from class names and borrows the `RepRta` held by `Model` for alignment (requires `yoloe-text`; `classes` accepts an `AsRef<str>` iterator). |
| `Session::text_with_embeddings(table)` | Activates externally provided embeddings, such as official CLIP embeddings. |
| `Session::text_with_reprta(reprta, table, config)` | Manually specifies RepRTA + external embeddings (advanced; `Session::text` already loads RepRTA automatically). |
| `Session::prompt_free(classes)` | Activates a prompt-free vocabulary. |
| `Session::prompt_free_default()` | Activates the built-in `default_labels::LRPC_VOCAB` (4585 names; requires `yoloe-pf`, included in aggregate `yoloe`; prediction class ids directly index this table for readable names). |
| `Session::prompt_free_with_embeddings(table)` | Activates static LRPC embeddings. |
| `Session::visual(prompts)` | Per-image visual prompt session (box/mask is selected by `VisualSource` in `predict_visual_prompts`; SAVPE runs during forward). |
| `Session::new(config)` / `from_checkpoint(name)` | Low-level constructors, used with the `set_*` methods below. |

> Low-level `set_*` methods (`set_classes_with_clip_embeddings`, `set_text_prompt_embeddings`, `set_visual_prompts`, `set_visual_prompt_embeddings`, `set_prompt_free_vocabulary`, and others) remain available for advanced users who need staged construction or precomputed embeddings.

### Text prompt (Low-Level `set_*` Methods)

| API | Description |
| --- | --- |
| `set_classes(classes)` | Records classes and embedding space only; it does not generate embeddings. You must then activate one of `set_classes_with_clip_embeddings` / `set_text_prompt_embeddings` / `set_text_prompt_embeddings_with_reprta`, otherwise text-prompt scoring returns an error. |
| `set_classes_with_clip_embeddings(&encoder, reprta, classes)` | Uses reusable `ClipTextEncoder` to generate an `EmbeddingTable` from class names and borrows `Option<&RepRta>` for alignment (requires `yoloe-text`). |
| `EmbeddingTable::new(embeddings, classes)` | Creates a `[classes, dim]` prompt embedding table. |
| `set_text_prompt_embeddings(table)` | Activates text prompt embeddings. |
| `RepRta::load(vb)` | Loads RepRTA from official `reprta` weights. |
| `RepRta::load_optional(vb)` | Loads RepRTA if `vb` contains `m.w12.weight`, otherwise returns `None` (used when loading `Model`). |
| `Model::reprta()` | Returns the `Option<&RepRta>` held by `Model` during loading, for borrowing by `Session::text`. |
| `set_text_prompt_embeddings_with_reprta(reprta, table)` | Applies RepRTA before activating embeddings and stores reprta in the session. |
| `set_reprta(reprta)` | Loads RepRTA; later `set_text_prompt_embeddings` automatically applies RepRTA when `config.rep_rta.enabled`, matching the official default inference path. |
| `score_region_features(features)` | Scores `[regions, dim]` or batched region features. |
| `score_feature_map(feature_map)` | Scores dense feature maps. |

### Visual prompt

| API | Description |
| --- | --- |
| `VisualPrompt::from_box(class_id, xyxy)` | Box prompt metadata. |
| `VisualPrompt::from_mask(class_id, xyxy)` | Mask prompt metadata. |
| `VisualKind::{Box, Mask}` | Prompt source type. |
| `Visuals::from_boxes(prompts, source_size, scale_factor, device)` | Constructs single-image official-style `visuals` tensor from box prompts (`[1, classes, h, w]`, merged by same class). |
| `Visuals::from_masks(prompts, masks, scale_factor)` | Constructs single-image official-style `visuals` tensor from segmentation masks. |
| `VisualBatchItem::new(source_size, prompts)` | Declares one source image and its bbox visual prompts; used to align with the official bbox-only batch branch. |
| `BatchVisuals::from_boxes(items, target_size, scale_factor, device)` | Aligns with the official bbox-only batch branch, letterboxes boxes, then outputs padded `[batch, max_prompts, h, w]` tensor plus per-image `class_ids` for the prompt dimension. |
| `BatchVisuals::from_tensor(tensor, class_ids)` | Wraps a caller-provided batch tensor directly (common in training paths). |
| `VisualSource::Boxes` / `VisualSource::Masks(&tensor)` | Source selector for `predict_visual_prompts`: rasterize boxes or consume source masks. |
| `set_visual_prompts(prompts)` | Activates visual prompt state. |
| `set_visual_prompt_embeddings(prompts, table)` | Sets visual prompts and precomputed SAVPE embeddings together. |
| `encode_visuals(embedding_map, &visuals)` | Fallback pooler that aggregates visual embeddings from an embedding map. |
| `Encoder::load(...)` | Loads official SAVPE. |
| `encode_visuals_with_savpe(encoder, features, &visuals)` | Uses official SAVPE to encode prompts from three feature-map levels. |

Visual prompt mask rules:

- `Visuals.tensor` has shape `[1, classes, height, width]`; same-class prompts are merged and sorted by class id according to official `LoadVisualPrompt.get_visuals()` semantics.
- `Visuals::from_boxes()` / `from_masks()` merge each prompt mask and sort into `[1, classes, mask_h, mask_w]`, which can be passed directly to `encode_visuals()` or `encode_visuals_with_savpe()`.
- Session `encode_visuals*` merges incoming `Visuals` again according to the current prompt state; if `Visuals` is already an official class-merged `visuals` tensor, it is passed through unchanged.
- `BatchVisuals::from_boxes()` only handles bbox prompts; this corresponds to the `len(img) > 1` bbox-only branch in official source. Official documentation still recommends running image-specific prompts one image at a time.
- The batch helper applies official-style letterbox bbox mapping, same-class merging, and prompt-dimension padding for each image, making it suitable for batch tensor entries such as `Pooler::encode()` or `Encoder::forward()`.
- `class_ids[b][p]` returned by `BatchVisuals::from_boxes()` is the original class id for prompt/class index `p` before padding in batch image `b`, which helps map predictions back to user-provided classes.

### Prompt-free / LRPC

| API | Description |
| --- | --- |
| `set_prompt_free_vocabulary(classes)` | Activates prompt-free class vocabulary. |
| `set_prompt_free_embeddings(table)` | Sets prompt-free vocabulary embeddings. |
| `LrpcHead` | Lightweight proposal filtering + prompt scoring. |
| `Official` | Loads official single-scale LRPC head. |
| `Pyramid` | Three-scale official prompt-free pyramid head. |
| `forward_official_lrpc_head(...)` | Forward for prompt-free detect head. |
| `forward_official_lrpc_segment_head(...)` | Forward for prompt-free segment head. |

### Open-Vocabulary Model/Head

| API | Description |
| --- | --- |
| `detect::head::Head` | YOLOE detect head, accepts prompt embeddings. |
| `Detect::from_file(path)` / `from_file_with(path, config)` / `from_pt_file(path, config)` / `from_bytes(bytes)` / `from_safetensors(weights, config)` | YOLOE detect-only model loading entry points; `.pt` is preferred, scale/head layout are inferred from checkpoint, and device/dtype/image_size/max_predictions are controlled by `yoloe::Config`. |
| `segment::head::Head` | YOLOE segment head, outputs predictions + proto masks. |
| `Segment::from_file(path)` / `from_file_with(path, config)` / `from_pt_file(path, config)` / `from_bytes(bytes)` / `from_safetensors(weights, config)` | YOLOE segment model loading entry points; `.pt` is preferred. |
| `forward_tensor(input, session)` | Runs raw forward according to session state (text/visual prompt or prompt-free); detect returns `[batch, det, 6]`, segment returns `([batch, det, 6 + mask_dim], proto)`. |
| `forward_prompt_free_tensor(input, session)` | Explicit prompt-free LRPC raw forward. |
| `predict(image, session, filter)` | Detect-only typed prediction entry point, returning `Vec<detect::Prediction>`. |
| `predict(image, session, filter, mask)` | Segment typed prediction entry point, returning `Vec<segment::Prediction>`. |
| `predict_visual_prompts(image, prompts, source, session, filter[, mask])` | Visual-prompt typed prediction entry point (supported by detect and segment); `source` uses `VisualSource::Boxes` for box rasterization or `VisualSource::Masks(&tensor)` for original-coordinate masks shaped `[prompts, H, W]`/`[1, prompts, H, W]`. Internally performs letterbox, SAVPE encoding, and postprocessing. |
| `encode_visual_prompts(reference_image, prompts, source)` | Cross-image two-step API: encodes reusable `EmbeddingTable` (official `vpe`) on a reference image through SAVPE, then feeds it to `Session::text_with_embeddings` and uses normal `predict` to identify any target image. Supported by detect and segment. |
| `predict_prompt_free(image, session, filter[, mask])` | Prompt-free typed prediction entry point; supported by detect and segment. |

Open-vocabulary head top-k postprocessing supports `[batch, anchors, channels]`. Text-prompt or prompt-free batch inference with the same prompt table sorts and retains predictions independently per batch sample. Multi-image visual prompt per-image class-id mapping is still handled by the `class_ids` returned from `BatchVisuals::from_boxes()`.

### Validate / Train

| API | Description |
| --- | --- |
| `validation_request()` | Generates a validation request for the current prompt state. |
| `train::yoloe::Session` | YOLOE training session. The three training steps are listed below. |

**Training steps** (see [usage/train-yoloe.md](usage/train-yoloe.md)):

| Prompt mode | Method | Prompt source |
| --- | --- | --- |
| text-prompt | `text_batch(&input, &target, &prompts)` | CLIP-encoded `EmbeddingTable` |
| visual-prompt | `visual_batch(&input, &target, &visuals, classes)` | Generated from boxes/masks by SAVPE |
| prompt-free | `prompt_free_batch(&input, &target)` | Checkpoint-internal LRPC; no parameters |

## Differences from Official Ultralytics

- Official `YOLOE("*.pt").set_classes([...])` processes text embeddings with a CLIP text encoder on the Python side. Under the `yoloe-text` feature, this crate calls the MobileCLIP2-b CLIP text encoder through `mobileclip2-rs` (`ClipTextEncoder`, reusable value type). `Session::text(&encoder, model.reprta(), classes)` (via low-level `set_classes_with_clip_embeddings`) encodes class names into `[classes, 512]` L2-normalized embeddings and borrows the RepRTA held by `Model` at load time for alignment, matching the official CLIP -> RepRTA -> score path. The caller constructs the CLIP encoder; it is not inside the YOLOE checkpoint.
- Official visual prompt can pass images, boxes, and masks directly to `predict(..., visual_prompts=...)`; this crate's `Segment` / `Detect` provide typed single-image box and mask visual-prompt prediction through `predict_visual_prompts`, selected by `VisualSource`, and the low-level `Visuals` type can construct official `visuals` tensors directly.
- Official documentation says batch inference supports normal inputs, but image-specific prompts should run one image at a time. Official source also has an internal `len(img) > 1`, bbox-only, `pad_sequence` padded path. This crate's `BatchVisuals::from_boxes()` only aligns with that YOLOE-26 source path and does not extend multi-image mask prompts.
- Official `LoadVisualPrompt` merges same-class prompt masks; this crate aligns with that behavior in `Visuals` construction and session mask encoding entries.
- Official YOLOE training is implemented by PyTorch trainer classes; this crate's `train::yoloe::Session` provides native batch steps (text/visual/prompt-free) and expresses corresponding trainers and freeze recipes through `Mode` + `PromptMode`. It currently does not reproduce official DDP/AMP/callback behavior; YOLOE validation records native loss, box mAP, and mask mAP. See [usage/train-yoloe.md](usage/train-yoloe.md).
- Official RepRTA-enabled checkpoints automatically enable RepRTA by default. This crate's `Model` automatically constructs and holds RepRTA through `RepRta::load_optional` from `model.23.reprta` during `from_file`/`from_pt_file`/`from_safetensors` (RepRTA is only needed for the text path, not visual/prompt-free), and `Session::text` borrows it through `model.reprta()` without manual loading.
- Official visual prompt returns the original category id. This crate uses placeholder visual prompt class names `visual_class_{original_id}` because `EmbeddingTable.classes` requires non-empty strings; original ids are provided by `BatchVisuals::from_boxes().class_ids`.
- The YOLOE one-to-one head does not need NMS by design (one anchor per GT). `postprocess_topk` only performs argmax + top-k sorting; `PredictConfig.agnostic_nms` is currently a no-op and is kept for future non-end2end fallback paths.
- Fallback `Contrastive` uses empirical `logit_scale=1.0` / `bias=-10.0` values (no checkpoint values are available), so they are not numerically comparable to official `BnContrastive` learnable `logit_scale` / `bias`. Official weights use the `BnContrastiveFeatureHead` path and correctly read checkpoint `logit_scale` / `bias`.
- Official YOLOE detect (for example `yoloe-26s.pt`) and segment are both first-class inference paths. This crate provides end-to-end `Detect` and `Segment` wrappers. Detect-only returns typed `detect::Prediction`; segment returns typed `segment::Prediction`. Official `.pt` is the preferred weight format, while converted `.safetensors` weights remain supported.
- **Pure detection weights (`yoloe-26s.pt`, or matching `.safetensors`) do not contain SAVPE**: calling `predict_visual_prompts` on them returns a runtime error telling you to use `-seg` weights. To get visual-prompt detection results, load `-seg` weights as `Detect` and discard masks.
- `Checkpoint::parse()` accepts any name with a `yoloe-` prefix, but only YOLOE-26 has been verified; `yoloe-11*` can parse but is outside this crate's supported scope.
- Official SAVPE is a three-scale convolution + multi-head spatial attention module. This crate's `encode_visuals()` fallback pooler only performs masked average pooling + L2 normalization (single embedding-map scenario) and is not numerically equivalent to official SAVPE. The official SAVPE main path `Encoder` is fully implemented; the single-scale fallback is only a compatibility entry.
