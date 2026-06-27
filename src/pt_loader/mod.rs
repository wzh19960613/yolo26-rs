//! Pure-Rust loader for official Ultralytics `.pt` checkpoints.
//!
//! Official YOLO26 / YOLOE-26 weights ship as PyTorch `.pt` files (the new
//! zip container with a pickle `data.pkl` plus per-storage `data/<key>` blobs).
//! This module reads such a file entirely in Rust and returns a flat
//! `HashMap<String, Tensor>` keyed by the same dotted paths the rest of the
//! crate already consumes from safetensors (`model.0.conv.weight`, ...), so the
//! existing `network::load` / `load_tensor_map` paths work unchanged.
//!
//! Only YOLO26 / YOLOE-26 checkpoints are supported. The loader reads the model
//! (or EMA) module tree, class names and basic metadata; it intentionally does
//! not deserialize the optimizer / scaler / scheduler state, which the native
//! training loop stores separately.

mod metadata;
mod reader;
mod tensor_build;

mod archive;
mod flatten;

mod blob_gen;
mod checkpoint_patch;
mod pickle_write;
mod storage_scan;
mod templates;
mod writer;

#[cfg(feature = "train")]
pub(crate) use checkpoint_patch::ClassMetadata as PtClassMetadata;
#[cfg(feature = "train")]
pub(crate) use writer::save_pt_with_class_metadata;
pub use writer::{save_pt, save_pt_with_names, save_pt_with_template_file};

pub use metadata::PtCheckpointMetadata;
#[allow(unused_imports)]
pub use metadata::PtNames;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

use crate::Result;

/// Loads a `.pt` checkpoint into a flat tensor map keyed by dotted state-dict
/// paths (e.g. `model.0.conv.weight`).
///
/// `device` controls where the resulting tensors live; dtypes follow the
/// storage types declared in the checkpoint (commonly F16 for YOLO26 weights).
pub fn load_pt_to_tensors(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let parsed = reader::read_checkpoint(path.as_ref())?;
    tensor_build::build_tensor_map(&parsed, device)
}

/// Loads a `.pt` checkpoint from an in-memory byte buffer, mirroring
/// [`load_pt_to_tensors`] but reading the zip container straight out of memory.
/// Used by the wasm API (no filesystem) and any other caller that already holds
/// the bytes.
pub fn load_pt_to_tensors_from_bytes(
    bytes: &[u8],
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let parsed = reader::read_checkpoint_from_zip_bytes(bytes)?;
    tensor_build::build_tensor_map(&parsed, device)
}

/// Reads checkpoint metadata (class names, epoch, best fitness) without
/// materializing every tensor.
pub fn load_pt_metadata(path: impl AsRef<Path>) -> Result<PtCheckpointMetadata> {
    let parsed = reader::read_checkpoint(path.as_ref())?;
    metadata::extract(&parsed)
}

/// Reads checkpoint metadata from an in-memory byte buffer, mirroring
/// [`load_pt_metadata`].
pub fn load_pt_metadata_from_bytes(bytes: &[u8]) -> Result<PtCheckpointMetadata> {
    let parsed = reader::read_checkpoint_from_zip_bytes(bytes)?;
    metadata::extract(&parsed)
}

/// Builds a [`VarBuilder`] from an official `.pt` checkpoint so the existing
/// `network::load` paths can consume official weights directly. The returned
/// builder owns the tensor map; lookups are keyed by the same dotted paths as
/// safetensors, so callers prefix it the same way (e.g. `vb.pp("model")`).
pub(crate) fn var_builder_from_pt_file(
    path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let tensors = load_pt_to_tensors(path, device)?;
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}

/// Builds a [`VarBuilder`] from an in-memory `.pt` byte buffer, mirroring
/// [`var_builder_from_pt_file`]. The returned builder owns the tensor map;
/// lookups are keyed by the same dotted paths as safetensors.
pub(crate) fn var_builder_from_pt_bytes(
    bytes: &[u8],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let tensors = load_pt_to_tensors_from_bytes(bytes, device)?;
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}
