//! Identity-driven inference for [`Scale`] and class count (`nc`).
//!
//! A YOLO26 checkpoint encodes its model scale in the backbone's first conv
//! output width (= `width * 64`) and its depth in the number of `model.2.m`
//! repeat blocks (= `depth`-scaled C3k2 count). The `(conv_out, depth_repeat)`
//! pair is unique across all five scales:
//!
//! | scale | first conv out | `model.2.m` blocks |
//! |-------|----------------|--------------------|
//! | n     | 16             | 1                  |
//! | s     | 32             | 1                  |
//! | m     | 64             | 1                  |
//! | l     | 64             | 2                  |
//! | x     | 96             | 2                  |
//!
//! The class count (`nc`) is the output-channel dim of the head's final
//! classification projection, whose tensor key differs per task.

use std::collections::HashMap;

use crate::Result;
use crate::model::Scale;

/// YOLO26 task family, selecting the head tensor key used to infer `nc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferredTask {
    /// Object detection (head prefix `model.23`).
    Detect,
    /// Image classification (head at `model.10`).
    Classify,
    /// Instance segmentation (head prefix `model.23`).
    Segment,
    /// Pose/keypoint estimation (head prefix `model.23`).
    Pose,
    /// Semantic segmentation (head at `model.17`).
    Semantic,
    /// Oriented bounding-box detection (head prefix `model.23`).
    Obb,
}

/// Infers the model [`Scale`] from checkpoint tensor shapes by reading the
/// backbone's first conv output width and the `model.2.m` repeat-block count.
pub(crate) fn infer_scale_from_shapes(shapes: &HashMap<String, Vec<usize>>) -> Result<Scale> {
    let conv_out = first_conv_out(shapes)?;
    let depth_repeat = model2_repeat_count(shapes);
    match (conv_out, depth_repeat) {
        (16, _) => Ok(Scale::N),
        (32, _) => Ok(Scale::S),
        (64, 1) => Ok(Scale::M),
        (64, _) => Ok(Scale::L),
        (96, _) => Ok(Scale::X),
        other => Err(crate::Error::InvalidConfig(format!(
            "cannot infer YOLO26 scale from checkpoint: first conv out={}, model.2.m blocks={}",
            other.0, other.1
        ))),
    }
}

/// Infers the class count (`nc`) from the head's final classification projection.
pub(crate) fn infer_labels_count_from_shapes(
    task: InferredTask,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<usize> {
    let key = head_nc_key(task);
    match shapes.get(key) {
        Some(dims) => dims.first().copied().ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "head tensor '{key}' has no dimensions to infer labels_count"
            ))
        }),
        None => Err(crate::Error::InvalidConfig(format!(
            "checkpoint has no '{key}' tensor to infer labels_count"
        ))),
    }
}

/// Infers the pose keypoint count from `model.23.cv4_kpts.0.2.weight` (dim 0
/// divided by the keypoint dims, assumed 3 for COCO-style x/y/visibility).
pub(crate) fn infer_keypoints_count(shapes: &HashMap<String, Vec<usize>>) -> Result<usize> {
    let key = "model.23.cv4_kpts.2.weight";
    match shapes.get(key) {
        Some(dims) => {
            let out = dims.first().copied().unwrap_or(0);
            Ok(out.div_ceil(3))
        }
        None => Err(crate::Error::InvalidConfig(format!(
            "checkpoint has no '{key}' tensor to infer keypoints_count"
        ))),
    }
}

/// Returns the head tensor key whose dim 0 is the class count for a task.
fn head_nc_key(task: InferredTask) -> &'static str {
    match task {
        InferredTask::Detect | InferredTask::Segment | InferredTask::Obb => {
            "model.23.cv3.0.2.weight"
        }
        InferredTask::Pose => "model.23.cv3.0.2.weight",
        InferredTask::Classify => "model.10.linear.weight",
        InferredTask::Semantic => "model.17.classifier.1.weight",
    }
}

/// Reads the backbone's first conv output width from `model.0.conv.weight` (or
/// the fused `model.0.0.conv.weight` variant).
fn first_conv_out(shapes: &HashMap<String, Vec<usize>>) -> Result<usize> {
    for key in ["model.0.conv.weight", "model.0.0.conv.weight"] {
        if let Some(dims) = shapes.get(key) {
            return dims.first().copied().ok_or_else(|| {
                crate::Error::InvalidConfig(format!(
                    "backbone tensor '{key}' has no dimensions to infer scale"
                ))
            });
        }
    }
    Err(crate::Error::InvalidConfig(
        "checkpoint has no 'model.0.conv.weight' to infer scale".to_string(),
    ))
}

/// Counts `model.2.m.{i}.` repeat blocks (the C3k2 depth-scaled sub-modules).
fn model2_repeat_count(shapes: &HashMap<String, Vec<usize>>) -> usize {
    let mut max_idx: i32 = -1;
    for key in shapes.keys() {
        if let Some(rest) = key.strip_prefix("model.2.m.")
            && let Some(idx_str) = rest.split('.').next()
            && let Ok(idx) = idx_str.parse::<i32>()
        {
            max_idx = max_idx.max(idx);
        }
    }
    (max_idx + 1).max(0) as usize
}

/// Returns true when `path` has a `.pt` extension (case-insensitive).
pub(crate) fn is_pt_path(path: &std::path::Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pt"))
}

/// Reads tensor shapes from a `.pt` path or a `.safetensors` path, dispatching
/// on the file extension. The `pt` feature gates `.pt` reading.
pub(crate) fn checkpoint_shapes(path: &std::path::Path) -> Result<HashMap<String, Vec<usize>>> {
    if is_pt_path(path) {
        #[cfg(feature = "pt")]
        {
            shapes_from_pt(path)
        }
        #[cfg(not(feature = "pt"))]
        {
            Err(crate::Error::InvalidConfig(
                "loading .pt requires the 'pt' feature".to_string(),
            ))
        }
    } else {
        let weights = std::fs::read(path)?;
        shapes_from_safetensors(&weights)
    }
}

/// Reads tensor shapes from an in-memory checkpoint byte buffer, auto-sniffing
/// the format by its magic bytes. `.pt` checkpoints are ZIP archives starting
/// with `PK\x03\x04`; safetensors starts with an 8-byte little-endian JSON
/// header length followed by `{`. The `pt` feature gates `.pt` reading.
pub(crate) fn checkpoint_shapes_from_bytes(bytes: &[u8]) -> Result<HashMap<String, Vec<usize>>> {
    if is_pt_bytes(bytes) {
        #[cfg(feature = "pt")]
        {
            shapes_from_pt_bytes(bytes)
        }
        #[cfg(not(feature = "pt"))]
        {
            Err(crate::Error::InvalidConfig(
                "loading .pt requires the 'pt' feature".to_string(),
            ))
        }
    } else {
        shapes_from_safetensors(bytes)
    }
}

/// Returns true when the byte buffer starts with the ZIP local-file signature,
/// i.e. it is a `.pt` (PyTorch zip) checkpoint rather than safetensors.
pub(crate) fn is_pt_bytes(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
}

/// Reads tensor shapes from a `.pt` checkpoint path (requires `pt` feature).
#[cfg(feature = "pt")]
pub(crate) fn shapes_from_pt(path: &std::path::Path) -> Result<HashMap<String, Vec<usize>>> {
    let tensors = crate::pt_loader::load_pt_to_tensors(path, &candle_core::Device::Cpu)?;
    Ok(tensors
        .into_iter()
        .map(|(n, t)| (n, t.shape().dims().to_vec()))
        .collect())
}

/// Reads tensor shapes from an in-memory `.pt` byte buffer (requires `pt`).
#[cfg(feature = "pt")]
pub(crate) fn shapes_from_pt_bytes(bytes: &[u8]) -> Result<HashMap<String, Vec<usize>>> {
    let tensors =
        crate::pt_loader::load_pt_to_tensors_from_bytes(bytes, &candle_core::Device::Cpu)?;
    Ok(tensors
        .into_iter()
        .map(|(n, t)| (n, t.shape().dims().to_vec()))
        .collect())
}

/// Reads tensor shapes from a `.safetensors` byte buffer.
pub(crate) fn shapes_from_safetensors(weights: &[u8]) -> Result<HashMap<String, Vec<usize>>> {
    let safetensors = candle_core::safetensors::SliceSafetensors::new(weights)?;
    Ok(safetensors
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect())
}
