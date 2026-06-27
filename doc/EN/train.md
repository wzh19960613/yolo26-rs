# Training

> **Training has not been rigorously tested. Use it carefully; training quality, convergence, and numeric equivalence with the official PyTorch Trainer are not guaranteed. Validate the full flow on a small dataset and a short run before using it for real training.**

Training lives under the `train` feature and is built on Candle autograd, `VarMap`, and optimizers. It provides Rust-native YOLO26 training infrastructure instead of wrapping the Ultralytics Python Trainer directly; trainers and parameters for non-YOLO26 families are outside this crate's implementation scope.

Training depends on Candle autograd and backend convolution implementations. Two Candle 0.10.2 paths can affect gradient correctness:

- `upsample_nearest1d/2d` backward should accumulate convolution-sum results when an existing gradient is present, rather than overwriting the existing gradient.
- Metal `conv1d/conv2d` should materialize potentially non-contiguous kernels into contiguous buffers before matmul and use a zero-offset contiguous layout, avoiding wrong offset reads in convolution-gradient paths with non-contiguous kernels.

For training, especially on Metal, first use a local patched `candle-core` as described in the [Candle 0.10.2 Patch Guide](candle-0.10.2-patch-guide.md), or wait for an upstream version containing the fixes.

End-to-end guides: [standard segmentation training](usage/train-seg.md) / [YOLOE training](usage/train-yoloe.md).

## Quick Start

```rust
use yolo26_rs::{Scale, detect, train};

let yaml = train::dataset::ultralytics::Yaml::from_file("data.yaml")?;
let config = train::ModelConfig::Detect(
    detect::config_builder()
        .with_scale(Scale::S)
        .with_labels_count(yaml.names.len())
        .build(),
);
let model = train::Model::new_with_class_names(config, yaml.names.clone())?;
let mut session = train::Session::new(
    model,
    train::OptimizerConfig::Sgd { learning_rate: 1e-4 },
)?;

let report = session.train_batch(&input, &target)?;
let eval = session.eval_batch(&input, &target)?;
```

A classification directory tree can also be used directly as a dataset:

```text
dataset/
  train/cat/*.jpg
  train/dog/*.jpg
  val/cat/*.jpg
  val/dog/*.jpg
```

## Full API

### Core Types

| API | Description |
| --- | --- |
| `train::Task` | `Detect`, `Classify`, `Segment`, `Pose`, `Semantic`, `Obb`. |
| `train::ModelConfig` | Enum wrapper around each task config. |
| `train::Model::new(config)` | Creates a trainable model and variables. |
| `train::Model::new_with_class_names(config, names)` | Creates a trainable model and writes official `model.names` in later `.pt` exports. `names.len()` must equal `labels_count`. |
| `Model::set_class_names(names)` | Sets `.pt` export class names for an already constructed model. |
| `Model::from_pt_file(path, config)` | Initializes a trainable model from an official `.pt`. |
| `Model::load_pt_file(path)` | Loads matching variables from an official `.pt` and returns `LoadReport`. |
| `Model::save_pt(path)` | Saves an official `.pt` and writes current class metadata. |
| `Model::save_pt_with_names(path, names)` | Saves an official `.pt` with explicit class names. |
| `Model::from_safetensors(weights, config)` | Initializes a trainable model from safetensors (optional compatibility path). |
| `Model::load_safetensors(weights)` | Loads matching variables from safetensors and returns `LoadReport`. |
| `Model::save_safetensors(path)` | Saves current training variables as safetensors. |
| `Model::forward_raw(input)` | Raw forward in training mode. |
| `Model::forward_raw_eval(input)` | Raw forward in eval mode. |
| `train::Output::DetectE2e` | Normal detection training raw output; contains one-to-many and one-to-one detection heads. Eval raw output and inference still use one-to-one. |
| `train::Output::SegmentE2e` | Instance segmentation training raw output; contains one-to-many and one-to-one detect/mask heads and shared proto. Eval raw output and inference still use one-to-one. |
| `train::Output::PoseE2e` | Pose training raw output; contains one-to-many and one-to-one detect/keypoint heads. Eval raw output and inference still use one-to-one. |
| `train::Output::ObbE2e` | OBB training raw output; contains one-to-many and one-to-one detect/angle heads. Eval raw output and inference still use one-to-one. |
| `train::Session` | Training/evaluation session. |
| `Session::new(model, optimizer)` | Trains all variables. |
| `Session::new_with_variable_filter(...)` | Trains only matching variables, such as head-only training. |
| `Session::save_optimizer_state_safetensors(path)` | Saves AdamW/MuSGD optimizer internal state; returns `false` for SGD when there is no internal state. |
| `Session::load_optimizer_state_safetensors(path)` | Restores current optimizer internal state from a safetensors sidecar. |
| `train::OptimizerConfig` | `Sgd { learning_rate }`, `AdamW { params }`, or `MuSgd { params }`; local `Sgd` only has `learning_rate` and does not support weight decay (known difference), while AdamW/MuSGD support weight decay. |
| `train::AutoOptimizerSelection::ultralytics(classes, iterations)` | Matches official `optimizer=auto` selection rules. Short runs use AdamW; long runs use MuSGD. |
| `train::ParamsMuSgd` | Native MuSGD parameters, including Muon/SGD mixed weights, momentum, nesterov, and weight decay. |
| `train::SampleOrder::ultralytics(seed, deterministic)` | Local sample-order config aligned with official `seed` / `deterministic`; a nonzero deterministic seed yields a stable index permutation. |
| `train::DetectionLossConfig` | Detection-style supervised loss parameters, including box/class/dfl/pose/keypoint-objectness/angle gains, `tal_topk`, `tal_topk2`, `tal_alpha`, and `tal_beta`; `tal_topk2` is used by official STAL/one-to-one two-stage candidate filtering. Small-object candidates follow official STAL by expanding too-small width/height to the second stride level, and classification target scores are normalized by official `align_metric * pos_overlaps / pos_align_metrics`. |
| `train::ProgressiveLossSchedule` | YOLO26 E2E loss schedule for one-to-many / one-to-one weights; defaults from `0.8/0.2` down to `0.1/0.9`. |
| `train::SegmentationMaskEncoding` | Segment target mask encoding; `Overlap` is the official `overlap_mask=True` single-channel instance-index map, while `PerInstance` is one binary mask per object. |
| `train::ClassFilter::new(single_class, classes)` | Training-target filtering and class-id remapping aligned with official `single_cls` / `classes`; unselected semantic-segmentation pixels enter ignore loss. |
| `train::Freeze::first_layers(n)` / `Freeze::layers([...])` | Aligned with official `freeze`; freezes the first N `model.{idx}.` layers or specified layer indices, and recognizes official always-frozen `.dfl` variables. |
| `train::CheckpointReport` | Checkpoint summary in `RunnerReport.checkpoints`, recording `last.pt`, `best.pt`, best epoch/loss, and periodic checkpoints. |
| `train::ResumeState` | Lightweight training state in checkpoint sidecars, recording completed epoch, completed step, and best loss. |

### Targets and Samples

| API | Description |
| --- | --- |
| `train::Sample` | One training sample containing `input` and `target`. |
| `train::Target` | Enum of classification, detection, segmentation, pose, semantic, and OBB targets. |
| `train::DetectionTargets::new(boxes_xyxy, class_ids, valid)` | Detection supervision target. |
| `train::SegmentationTargets::new(detection, masks)` | Instance segmentation supervision target. |
| `train::PoseTargets::new(detection, keypoints, keypoint_valid)` | Pose supervision target. |
| `train::ObbTargets::new(detection, angles)` | Oriented box supervision target. |
| `train::collate_*_samples` | Merges samples into batches by task. |

### Dataset Adapters

| API | Description |
| --- | --- |
| `Yaml::from_file(path)` | Parses YOLO YAML. |
| `detection_dataset_from_file(...)` | Detection dataset. |
| `segmentation_dataset_from_file(...)` | Instance segmentation dataset. |
| `segmentation_dataset_from_file_with_overlap_mask(...)` | Explicitly selects segment target mask encoding; the default entry uses official `overlap_mask=True`. |
| `pose_dataset_from_file(...)` | Pose dataset; reads `kpt_shape`. |
| `semantic_dataset_from_file(...)` | Semantic segmentation dataset. |
| `obb_dataset_from_file(...)` | OBB dataset. |
| `ClassificationDataset::from_dir(...)` | Classification directory dataset. |

YAML splits support image directories, image-list `.txt` files, single images, and common globs including recursive `**` directories.

For custom-class training, read the official dataset YAML with `Yaml::from_file(data_yaml)`, then pass `yaml.names.len()` to the task config's `with_labels_count(...)`. After setting class names with `Model::new_with_class_names(config, yaml.names.clone())` or `Model::set_class_names(...)`, `Model::save_pt("best.pt")` also rewrites head tensor metadata, `model.nc`, and `model.names` in the `.pt`; official Python `YOLO("best.pt")` can read those names directly. If class names are not set, `.pt` export generates placeholders such as `class_0`, `class_1`, avoiding mismatches between `nc` and `names`.

### Training and Evaluation

`RunnerConfig` controls `epochs`, `batch_size`, `steps_per_epoch`, `accumulate_steps`, `sample_fraction`, `sample_order`, `time_limit_hours`, `loss_config`, `class_filter`, `early_stopping`, `resume_state`, `learning_rate_schedule`, `learning_rate_warmup`, `bias_learning_rate_warmup`, `momentum_warmup`, `log_every_steps`, `checkpoint_dir`, `checkpoint_every_steps`, and `checkpoint_every_epochs`. When `checkpoint_dir` is set, the training loop writes official-style `last.pt` and `best.pt`, plus `*.train-state.json` and `*.optimizer.safetensors` sidecars for each checkpoint. Default `train_dataset()` selects `best` by epoch mean training loss, while `train_dataset_with_epoch_fitness()` accepts per-epoch validation fitness and selects `best` in the official direction. `loss_config` matches official detection-style loss gains, defaulting to `box=7.5`, `cls=0.5`, `dfl=1.5`, `pose=12.0`, `kobj=1.0`, `angle=1.0`. `sample_order` comes from `SampleOrder::ultralytics(seed, deterministic)` and controls local train/eval sample index order. `time_limit_hours` matches the official `time` training duration limit in hours; when triggered, `RunnerReport` records `elapsed_seconds` and `time_limit_reached`. `EvalLoopConfig` controls `batch_size`, `steps`, `max_detections`, `confidence_threshold`, `iou_threshold`, `sample_order`, `loss_config`, and `class_filter`.

### Training Parameters (Aligned with Official Semantics)

Library APIs express official training parameter semantics through the config types above. Common official parameters and their crate mappings:

| Official parameter | Crate API | Description |
| --- | --- | --- |
| `epochs` / `batch` / `imgsz` | `RunnerConfig` | Epoch count, batch size, input size. |
| `lr0` / `lrf` / `cos_lr` | `RunnerConfig` learning-rate schedule | Initial LR, final LR fraction, cosine one-cycle. |
| `optimizer` | `OptimizerConfig` / `AutoOptimizerSelection` | `auto`/`sgd`/`adamw`/`musgd`. |
| `box` / `cls` / `dfl` / `pose` / `kobj` / `angle` | `DetectionLossConfig` | Loss gains, default `7.5/0.5/1.5/12.0/1.0/1.0`. |
| `augment` / HSV / flip / mosaic / mixup | `AugmentConfig` + `AugmentingDataset` | Main augmentation switch is off by default; enabling uses official defaults. |
| `freeze` | `Freeze` | Freezes the first N layers or specified layer indices. |
| `single_cls` / `classes` | `ClassFilter` | Training-target filtering and class-id remapping. |
| `close_mosaic` | Shared epoch counter in `AugmentingDataset` | Disables mosaic/mixup for the final N epochs. |
| `ema` | `ModelEma` (builder field equivalent to `--ema-decay`) | Shadow tensor lerp. |
| `patience` | `EarlyStopping` | Early-stop patience. |
| `nbs` / `weight_decay` | `RunnerConfig` accumulate + optimizer params | Nominal batch size and weight-decay scaling. |
| `seed` / `deterministic` | `SampleOrder::ultralytics` | Sample order. |
| `time` | `RunnerConfig::time_limit_hours` | Training time limit in hours. |
| `mask_ratio` / `overlap_mask` | `segmentation_dataset_from_file*` | Segment target mask size and overlap encoding. |

## YOLOE Training

YOLOE training adds prompt alignment on top of the common training types. **For full examples and the distinction between the two "mode" concepts, see [usage/train-yoloe.md](usage/train-yoloe.md).** This section is an API quick reference.

YOLOE training has two independent mode enums; do not confuse them:

- **`PromptMode`** (`yoloe::segment::model::train_config::PromptMode`): the prompt-alignment method selected when constructing the network. Values are `TextPrompt` / `Visual` / `PromptFree`. Pass it to `ModelConfig::new`.
- **`train::yoloe::Mode`**: the official trainer recipe, which decides which parameters to freeze. Values are `FineTune` / `LinearProbe` / `FromScratch` / `Visual` / `PromptFree`.

| API | Description |
| --- | --- |
| `ModelConfig::new(scale, device, dtype, prompt_mode)` | Builds trainable YOLOE seg config; `prompt_mode` selects the network layer's prompt-alignment path. |
| `Model::new(config)` / `Model::from_pt_file(config, path)` / `Model::from_safetensors(config, path)` | Creates/loads a trainable model; `.pt` is the preferred initialization path. |
| `train::yoloe::Session::new(model, optimizer)` | YOLOE training session bound to model and optimizer. |
| `Session::new_with_variable_filter(model, optimizer, filter)` | Trains only matching variables (implements freeze recipes). |
| `Session::text_batch(&input, &target, &prompts)` | One text-prompt training step; `prompts` is a CLIP-encoded `EmbeddingTable`. |
| `Session::visual_batch(&input, &target, &visuals, classes)` | One visual-prompt training step; `visuals` is `BatchVisuals` (box/mask). |
| `Session::prompt_free_batch(&input, &target)` | One prompt-free training step; no prompt parameters. |
| `train::yoloe::Mode` | Official trainer recipe: `FineTune`/`LinearProbe` (-> `YOLOEPESegTrainer`), `FromScratch` (-> `YOLOESegTrainerFromScratch`), `Visual` (-> `YOLOESegVPTrainer`), `PromptFree` (-> `YOLOEPEFreeTrainer`). |
| `Mode::official_segment_trainer()` | Returns the corresponding official trainer name. |
| `Session::save_pt(path)` / `save_safetensors(path)` | Saves trained weights; `.pt` is the preferred output format. |

## Differences from Official Ultralytics

- Official `model.train(...)` is a PyTorch Trainer with full augmentation, schedulers, validators, AMP/DDP, resume, and callback ecosystems; this crate is Candle-native training infrastructure.
- Official YOLOE Trainer has classes such as `YOLOEPETrainer`, `YOLOEPESegTrainer`, `YOLOESegTrainerFromScratch`, `YOLOESegVPTrainer`, and `YOLOEPEFreeTrainer`; this crate expresses the corresponding trainer, official freeze recipe, grounding-data requirements, and `single_cls` semantics through `Session::train_config()`. Both the library layer and examples use the `train::yoloe::Session` batch API to cover text, visual, and prompt-free train/eval steps.
- When a training checkpoint path ends in `.pt`, export uses the `pt_loader::save_pt` / `Model::save_pt` / `Seg::save_pt` template method to write official PyTorch-readable `.pt` files (`pt` feature, default; embeds 40 official `(task, scale)` `data.pkl` templates, solid-compressed around 950 KiB, supporting 6 standard tasks + 3 YOLOE segment modes). Standard-task export rewrites `data.pkl` according to current model tensor shapes and supports arbitrary `labels_count`; when class names are set, official `model.names` is synchronized. EMA/optimizer/`proto.semseg` non-model storages and `num_batches_tracked` are zero-filled. `.safetensors` remains optional (with `.train-state.json` + `.optimizer.safetensors` sidecars).
- YOLOE text-prompt training class-name embeddings are generated through the CLIP encoder under the `yoloe-text` feature (`mobileclip2-rs` dependency) as `[classes, 512]` embeddings; prompt-free training needs no prompt table and uses checkpoint-internal LRPC outputs directly. See [usage/train-yoloe.md](usage/train-yoloe.md).
- This crate currently does not reproduce official DDP/AMP/callback support, PyTorch scheduler-object restoration, or the full multi-dataset grounding pipeline. Native validation records loss, box mAP, and mask mAP. See the training-loss alignment and difference notes below.
