//! Template-based `.pt` writer: reuses an official checkpoint object graph and
//! regenerates `data.pkl` plus the `data/<N>` storage blob bytes.
//!
//! This guarantees official `torch.load` compatibility because the pickle
//! object graph (module tree, tensor metadata, storage references) is copied
//! verbatim from a real PyTorch checkpoint template. Only the raw weight bytes
//! are rewritten, and the storage layout (blob path, offset, dtype, shape) is
//! read back from the template so regenerated blobs land at the right offsets.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::Result;
use crate::model::Scale;

use super::archive::{read_data_pkl, write_pt_zip};
use super::blob_gen::{build_blobs, build_blobs_with_fallback};
use super::checkpoint_patch::{self, ClassMetadata};
use super::pickle_write;
use super::reader::{read_checkpoint, read_checkpoint_from_bytes};
use super::storage_scan::collect_all;
use super::templates::resolve;

/// Writes a `.pt` checkpoint that official `torch.load` can read.
///
/// `task` and `scale` select the embedded official `data.pkl` template, which
/// fixes the storage layout (blob path, byte offset, dtype, shape) and the
/// archive root that prefixes every entry. The destination zip stores a patched
/// copy of that template plus regenerated `data/<N>` blobs built from `tensors`.
/// Every storage the pickle references gets a blob: model tensors present in
/// `tensors` carry their real bytes; the rest are zero-filled so the storage
/// layout stays intact.
pub fn save_pt(
    dest: impl AsRef<Path>,
    tensors: &HashMap<String, Tensor>,
    task: &str,
    scale: Scale,
) -> Result<()> {
    save_pt_with_class_metadata(dest, tensors, task, scale, None)
}

/// Writes a `.pt` checkpoint and embeds class names as official
/// `model.names` metadata.
///
/// The class count is derived from `names.len()`, and the head tensor metadata
/// in `data.pkl` is resized to match the tensors being saved.
pub fn save_pt_with_names(
    dest: impl AsRef<Path>,
    tensors: &HashMap<String, Tensor>,
    task: &str,
    scale: Scale,
    names: &[String],
) -> Result<()> {
    if names.is_empty() {
        return Err(crate::Error::InvalidConfig(
            "pt class names must not be empty".to_string(),
        ));
    }
    let metadata = ClassMetadata {
        labels_count: names.len(),
        names: Some(names),
    };
    save_pt_with_class_metadata(dest, tensors, task, scale, Some(&metadata))
}

pub(crate) fn save_pt_with_class_metadata(
    dest: impl AsRef<Path>,
    tensors: &HashMap<String, Tensor>,
    task: &str,
    scale: Scale,
    metadata: Option<&ClassMetadata<'_>>,
) -> Result<()> {
    let template = resolve(task, scale)?;
    let tensors = tensors_with_runtime_tensors(tensors, task)?;
    let mut parsed = read_checkpoint_from_bytes(template.pkl_bytes, &template.dir_name)?;
    let named_infos = checkpoint_patch::apply(
        &mut parsed.full_object,
        &parsed.dir_name,
        &tensors,
        metadata,
    )?;
    let all_infos = collect_all(&parsed.full_object, &parsed.dir_name)?;
    let pkl_bytes = pickle_write::to_vec(&parsed.full_object)?;
    let blobs = build_blobs(&named_infos, &all_infos, &tensors)?;
    write_pt_zip(dest.as_ref(), &pkl_bytes, &blobs, &template.dir_name)
}

/// Writes a `.pt` checkpoint using another `.pt` file as the complete storage
/// template. Trainable model tensors present in `tensors` are replaced; storage
/// blobs that are not represented in the native model are copied from
/// `template_path` instead of being zero-filled.
pub fn save_pt_with_template_file(
    dest: impl AsRef<Path>,
    tensors: &HashMap<String, Tensor>,
    template_path: impl AsRef<Path>,
) -> Result<()> {
    let template_path = template_path.as_ref();
    let parsed = read_checkpoint(template_path)?;
    let pkl_bytes = read_data_pkl(template_path)?;
    let all_infos = collect_all(&parsed.full_object, &parsed.dir_name)?;
    let blobs = build_blobs_with_fallback(&parsed.tensor_infos, &all_infos, tensors, &parsed)?;
    write_pt_zip(dest.as_ref(), &pkl_bytes, &blobs, &parsed.dir_name)
}

fn tensors_with_runtime_tensors(
    tensors: &HashMap<String, Tensor>,
    task: &str,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = tensors.clone();
    if matches!(task, "detect" | "segment" | "pose" | "obb") {
        let stride = Tensor::from_vec(vec![8.0f32, 16.0, 32.0], 3, &Device::Cpu)?;
        tensors.insert("stride".to_string(), stride.clone());
        tensors.insert("model.stride".to_string(), stride.clone());
        tensors.insert("23.stride".to_string(), stride.clone());
        tensors.insert("model.23.stride".to_string(), stride);
    }
    Ok(tensors)
}
